import numpy as np
from typing import Any, Counter, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
from dpu_utils.mlutils import Vocabulary
import torch
from torch import nn
import annoy
from dpu_utils.utils import RichPath
import logging
import tempfile
from collections import defaultdict

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.neuralmodels.gnn import GnnOutput, GraphData
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetwork, GraphNeuralNetworkModel
from ptgnn.implementations.typilus.graph2class import TypilusGraph, Prediction, TensorizedGraph2ClassSample, IGNORED_TYPES, Graph2ClassModule

from utils import load_data_from_sample

class Graph2MetricModule(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork, margin: float):
        super().__init__()
        self.__gnn = gnn
        self.__margin = margin

    def _reset_module_metrics(self) -> None:
        self.__num_samples = 0
        self.__sum_accuracy = 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {"Accuracy": int(self.__sum_accuracy) / int(self.__num_samples)}
    
    def _get_supernode_representations(self, graph_mb_data):
        graph_output: GnnOutput = self.__gnn(**graph_mb_data)
        # Gather the output representation of the nodes of interest
        supernode_idxs = graph_output.reference_nodes_idx["supernodes"]
        supernode_graph_idx = graph_output.reference_nodes_graph_idx["supernodes"]
        supernode_representations = graph_output.output_node_representations[
            supernode_idxs
        ]  # [num_supernodes_in_mb, D]
        return supernode_representations, supernode_graph_idx

    def forward(self, graph_mb_data, typed_annotation_pairs_are_equal, original_supernode_idxs):
        # print(graph_mb_data['adjacency_lists'])
        target_representations, _ = self._get_supernode_representations(graph_mb_data)
        target_representations_1 = target_representations.unsqueeze(0)  # 1 x N x D
        target_representations_2 = target_representations.unsqueeze(1)  # N x 1 x D

        distances = torch.norm(target_representations_1 - target_representations_2, dim=-1, p=1)  # N x N

        max_positive_distance = (distances * typed_annotation_pairs_are_equal).max(dim=-1)[0]  # N
        neg_dist_filter = distances <= (max_positive_distance + self.__margin).unsqueeze(-1)
        pos_mask = typed_annotation_pairs_are_equal + torch.eye(distances.size(0), device=distances.device)
        neg_dist_filter = neg_dist_filter.float() * (1 - pos_mask)
        mean_negative_distances = (distances * neg_dist_filter).sum(dim=-1) / (neg_dist_filter.sum(dim=-1) + 1e-10)  # N

        min_negative_distance = (distances + pos_mask * 3000).min(dim=-1)[0]
        pos_dist_filter = (distances >= (min_negative_distance - self.__margin).unsqueeze(-1)).float()
        pos_dist_filter *= typed_annotation_pairs_are_equal
        mean_positive_distances = (distances * pos_dist_filter).sum(dim=-1) / (pos_dist_filter.sum(dim=-1) + 1e-10)

        triplet_loss = 0.5 * torch.nn.functional.relu(mean_positive_distances - min_negative_distance + self.__margin)
        triplet_loss += 0.5 * torch.nn.functional.relu(max_positive_distance - mean_negative_distances + self.__margin)

        return triplet_loss.mean()
    

class Graph2HybridMetricModule(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork, num_target_classes: int, margin: float):
        super().__init__()
        self._graph2class = Graph2ClassModule(gnn, num_target_classes)
        self._graph2metric = Graph2MetricModule(gnn, margin)

    def predict(self, graph_mb_data, metadata):
        with torch.no_grad():
            target_representations, supernode_graph_idx = self._graph2metric._get_supernode_representations(graph_mb_data)
            if target_representations.shape[0] > 10000:
                return
            
            if not isinstance(metadata['index'], annoy.AnnoyIndex):
                with tempfile.NamedTemporaryFile() as f:
                    with open(f.name, 'wb') as fout:
                        fout.write(metadata['index'])
                    metadata['index'] = annoy.AnnoyIndex(self.__type_representation_size, 'manhattan')
                    metadata['index'].load(f.name)

            # This is also classification-specific due to class_id_to_class
            for representation in target_representations:
                nn_idx, distance =  metadata['index'].get_nns_by_vector(representation, n=10, include_distances=True)
                distances = 1 / (np.array(distance) + 1e-10) ** 2
                distances /= np.sum(distances)
                rel_types = defaultdict(int)
                for n, p in zip(nn_idx, distances):
                    rel_types[metadata['indexed_element_types'][n]] += p
                predicted_annotation_logprob_dist={t: np.log(v) for t, v in rel_types.items()}
                yield (
                    torch.stack((torch.tensor(list(predicted_annotation_logprob_dist.keys())), torch.tensor(list(predicted_annotation_logprob_dist.values()))), dim=0),
                    torch.tensor(max(predicted_annotation_logprob_dist, key=predicted_annotation_logprob_dist.get)),
                    supernode_graph_idx
                )
        
    def forward(self, graph_mb_data, target_classes, typed_annotation_pairs_are_equal, original_supernode_idxs):
        return self._graph2class(graph_mb_data, target_classes, original_supernode_idxs) + self._graph2metric(graph_mb_data, typed_annotation_pairs_are_equal, original_supernode_idxs)


class Graph2HybridMetric(
    AbstractNeuralModel[TypilusGraph, TensorizedGraph2ClassSample, Graph2HybridMetricModule]
):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel,
        max_num_classes: int = 100,
        try_simplify_unks: bool = True,
        margin: float = 2, # same as typilus graph2hybridmetric
    ):
        super().__init__()
        self.__gnn_model = gnn_model
        self.max_num_classes = max_num_classes
        self.__try_simplify_unks = try_simplify_unks
        self.__tensorize_samples_with_no_annotation = False
        self.__tensorize_keep_original_supernode_idx = False
        self.margin = margin

    # converts TypilusGraph into GraphData, which is the format that GraphNeuralNetworkModels accept
    def __convert(self, typilus_graph: TypilusGraph) -> Tuple[GraphData[str, None], List[str]]:
        # returns a list of edges with format (from_node_idx, to_node_idx)
        def get_adj_list(adjacency_dict):
            for from_node_idx, to_node_idxs in adjacency_dict.items():
                from_node_idx = int(from_node_idx)
                for to_idx in to_node_idxs:
                    yield (from_node_idx, to_idx)

        edges = {}
        for edge_type, adj_dict in typilus_graph["edges"].items():
            adj_list: List[Tuple[int, int]] = list(get_adj_list(adj_dict))
            if len(adj_list) > 0:
                edges[edge_type] = np.array(adj_list, dtype=np.int32)
            else:
                edges[edge_type] = np.zeros((0, 2), dtype=np.int32)

        # find all supernodes with proper annotations
        supernode_idxs_with_ground_truth: List[int] = []
        supernode_annotations: List[str] = []
        for supernode_idx, supernode_data in typilus_graph["supernodes"].items():
            if supernode_data["annotation"] in IGNORED_TYPES:
                continue
            if (
                not self.__tensorize_samples_with_no_annotation
                and supernode_data["annotation"] is None
            ):
                continue
            elif supernode_data["annotation"] is None:
                supernode_data["annotation"] = "??"
            supernode_idxs_with_ground_truth.append(int(supernode_idx))
            supernode_annotations.append(enforce_not_None(supernode_data["annotation"]))

        return (
            GraphData[str, None](
                node_information=typilus_graph["nodes"],
                edges=edges,
                reference_nodes={
                    "token-sequence": typilus_graph["token-sequence"],
                    "supernodes": supernode_idxs_with_ground_truth,
                },
            ),
            supernode_annotations,
        )

    # region Metadata Loading
    '''
    Pseudo code:
    metadata = initialize_metadata()
    for each training_sample:
        update_metadata_from(training_sample)
    metadata = finalize_metadata(metadata)
    '''

    def initialize_metadata(self) -> None:
        self.__target_class_counter = Counter[str]()

    def update_metadata_from(self, datapoint: TypilusGraph) -> None:
        # target_classes is a list of str
        graph_data, target_classes = self.__convert(datapoint)
        self.__gnn_model.update_metadata_from(graph_data)
        self.__target_class_counter.update(target_classes)

    def finalize_metadata(self) -> None:
        # create a close vocabulary of top-n types
        self.__target_vocab = Vocabulary.create_vocabulary(
            self.__target_class_counter,
            max_size=self.max_num_classes + 1,
        )
        del self.__target_class_counter

    # endregion

    def build_neural_module(self) -> Graph2HybridMetricModule:
        return Graph2HybridMetricModule(
            gnn=self.__gnn_model.build_neural_module(), 
            num_target_classes=len(self.__target_vocab),
            margin=self.margin
        )

    # Tensorization is the process where we convert the raw data into tensors that can be fed into our neural module. 
    # The tensorize() will be called for each sample in our dataset.
    def tensorize(self, datapoint: TypilusGraph) -> Optional[TensorizedGraph2ClassSample]:
        graph_data, target_classes = self.__convert(datapoint)
        if len(target_classes) == 0:
            return None  # Sample contains no ground-truth annotations.

        graph_tensorized_data = self.__gnn_model.tensorize(graph_data)

        if graph_tensorized_data is None:
            return None  # Sample rejected by the GNN

        target_class_ids = []
        for target_cls in target_classes:
            if self.__try_simplify_unks and self.__target_vocab.is_unk(target_cls):
                generic_start = target_cls.find("[")
                if generic_start != -1:
                    target_cls = target_cls[:generic_start]
            target_class_ids.append(self.__target_vocab.get_id_or_unk(target_cls))

        return TensorizedGraph2ClassSample(
            graph=graph_tensorized_data, 
            supernode_target_classes=target_class_ids
        )


    # region Minibatching
    '''
    Pseudo code:
    mb_data = initalize_minibatch()
    for datapoint in some_samples:
        extend_minibatch_with(tensorized_datapoint, mb_data)
    mb_data = finalize_minibatch(mb_data)

    # Compute the output of a neural module on the minibatch data
    neural_module(**mb_data)
    '''

    # initialize_minibatch creates a dictionary where we accumulate the minibatch data. 
    # It explicitly invokes the GNN model and asks it to initialize its portion of the minibatch.
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "graph_mb_data": self.__gnn_model.initialize_minibatch(),
            "target_classes": [],
            "original_supernode_idxs": [],
        }

    # extend_minibatch_with accepts a single tensorized datapoint (as returned by (tensorize()) 
    # and extends the partial_minibatch with that sample.
    # We unpack the tensorized_datapoint and pass the graph-related data to the GNN model along with the graph-related partial minibatch.
    # We extend target_classes by appending all the target class indices. Note that this behavior is different from common minibatching where tensors are stacked together using a different "batch" dimension. This is necessary, as graphs have different numbers of supernodes.
    def extend_minibatch_with(
        self, 
        tensorized_datapoint: TensorizedGraph2ClassSample, 
        partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["target_classes"].extend(tensorized_datapoint.supernode_target_classes)
        if self.__tensorize_keep_original_supernode_idx:
            partial_minibatch["original_supernode_idxs"].extend(
                tensorized_datapoint.graph.reference_nodes["supernodes"]
            )
        return self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.graph, partial_minibatch["graph_mb_data"]
        )

    # finalize_minibatch, unpacks the GNN-related data and invokes finalize_minibatch for the child GNN model. 
    # It also creates a PyTorch Tensor for the target classes. 
    # The keys of the returned dictionary are the names of the arguments in the forward() of Graph2HybridMetricModule.
    def finalize_minibatch(
        self, 
        accumulated_minibatch_data: Dict[str, Any], 
        device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        supernode_target_classes = accumulated_minibatch_data['target_classes']
        types_are_equal  = np.zeros((len(supernode_target_classes), len(supernode_target_classes)), dtype=bool)
        for i in range(len(supernode_target_classes)):
            for j in range(len(supernode_target_classes)):
                if supernode_target_classes[i] == supernode_target_classes[j]:
                    types_are_equal [i][j] = True
                    types_are_equal [j][i] = True
        return {
            "graph_mb_data": self.__gnn_model.finalize_minibatch(
                accumulated_minibatch_data["graph_mb_data"], device
            ),
            "target_classes": torch.tensor(
                accumulated_minibatch_data["target_classes"], dtype=torch.int64, device=device
            ),
            "typed_annotation_pairs_are_equal": types_are_equal ,
            "original_supernode_idxs": accumulated_minibatch_data["original_supernode_idxs"],
        }

    # endregion

    def create_index(self, data_paths: List[RichPath], metadata: Dict[str, Any], device: Union[str, torch.device]) -> None:
        representation_size = 0
        def representation_iter():
            data_chunk_iterator = (r.read_by_file_suffix() for r in data_paths)
            with torch.no_grad():
                for raw_data_chunk in data_chunk_iterator:
                    for raw_sample in raw_data_chunk:
                        loaded_sample = {}
                        use_example = load_data_from_sample(
                            raw_sample=raw_sample,
                            result_holder=loaded_sample,
                            vocab=self.__target_vocab,
                            is_train=False
                        )
                        if not use_example:
                            continue

                        # TODO
                        assert isinstance(raw_sample, TypilusGraph)
                        mb_data = self.initialize_minibatch()
                        self.extend_minibatch_with(tensorized_datapoint=self.tensorize(raw_sample))
                        mb_data = self.finalize_minibatch(mb_data, device)
                        target_representations = Graph2MetricModule._get_supernode_representations(mb_data['graph_mb_data'])

                        if representation_size == 0:
                            representation_size = target_representations.size()[1]

                        idx = 0
                        for node_idx, annotation_data in raw_sample['supernodes'].items():
                            node_idx = int(node_idx)
                            if 'ignored_supernodes' in loaded_sample and node_idx in loaded_sample['ignored_supernodes']:
                                continue

                            annotation = annotation_data['annotation']
                            if annotation in IGNORED_TYPES:
                                idx += 1
                                continue

                            yield target_representations[idx], annotation
                            idx += 1

        index = annoy.AnnoyIndex(representation_size, 'manhattan')
        indexed_element_types = []
        logging.info('Creating index...')
        for i, (representation, type) in enumerate(representation_iter()):
            index.add_item(i, representation)
            indexed_element_types.append(type)
        logging.info('Indexing...')
        index.build(20)
        logging.info('Index Created.')

        with tempfile.NamedTemporaryFile() as f:
            index.save(f.name)
            with open(f.name, 'rb') as fout:
                metadata['index'] = fout.read()
        metadata['indexed_element_types'] = indexed_element_types

    def report_accuracy(
        self,
        dataset: Iterator[TypilusGraph],
        trained_network: Graph2HybridMetricModule,
        device: Union[str, torch.device],
    ) -> float:
        trained_network.eval()
        unk_class_id = self.__target_vocab.get_id_or_unk(self.__target_vocab.get_unk())

        num_correct, num_elements = 0, 0
        for mb_data, _ in self.minibatch_iterator(
            self.tensorize_dataset(dataset), device, max_minibatch_size=50
        ):
            for target_idx, (_, prediction, _) in zip(mb_data["target_classes"], trained_network.predict(mb_data["graph_mb_data"])):
                num_elements += 1
                if target_idx == prediction and target_idx != unk_class_id:
                    num_correct += 1
        return num_correct / num_elements
    
    def predict(
        self,
        data: Iterator[TypilusGraph],
        trained_network: Graph2HybridMetricModule,
        device: Union[str, torch.device],
    ) -> Iterator[Prediction]:
        trained_network.eval()
        with torch.no_grad():
            try:
                self.__tensorize_samples_with_no_annotation = True
                self.__tensorize_keep_original_supernode_idx = True

                for mb_data, original_datapoints in self.minibatch_iterator(
                    self.tensorize_dataset(data, return_input_data=True, parallelize=False),
                    device,
                    max_minibatch_size=50,
                    parallelize=False,
                ):
                    current_graph_idx = 0
                    graph_preds: Dict[int, Tuple[str, float]] = {}

                    # TODO
                    probs, predictions, graph_idxs = trained_network.predict(
                        mb_data["graph_mb_data"]
                    )
                    supernode_idxs = mb_data["original_supernode_idxs"]
                    for graph_idx, prediction_prob, prediction_id, supernode_idx in zip(
                        graph_idxs, probs, predictions, supernode_idxs
                    ):
                        if graph_idx != current_graph_idx:
                            yield original_datapoints[current_graph_idx], graph_preds
                            current_graph_idx = graph_idx
                            graph_preds: Dict[int, Tuple[str, float]] = {}

                        predicted_type = self.__target_vocab.get_name_for_id(prediction_id)
                        graph_preds[supernode_idx] = predicted_type, float(prediction_prob)
                    yield original_datapoints[current_graph_idx], graph_preds
            finally:
                self.__tensorize_samples_with_no_annotation = False
                self.__tensorize_keep_original_supernode_idx = False
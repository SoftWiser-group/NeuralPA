import numpy as np
from dpu_utils.mlutils import Vocabulary
from ptgnn.implementations.typilus.graph2class import IGNORED_TYPES

def load_data_from_sample(raw_sample, result_holder, vocab: Vocabulary, is_train=False):
    target_node_idxs, target_class, target_class_id = [], [], []
    for node_idx, annotation_data in raw_sample['supernodes'].items():
        node_idx = int(node_idx)
        annotation = annotation_data['annotation']
        if is_train and annotation in IGNORED_TYPES:
            continue
        target_node_idxs.append(node_idx)
        target_class.append(annotation)
        target_class_id.append(vocab.get_id_or_unk(annotation))

    result_holder['target_node_idxs'] = np.array(target_node_idxs, dtype=np.uint16)
    result_holder['target_type'] = target_class
    result_holder['variable_target_class'] = np.array(target_class_id, dtype=np.uint16)
    return len(target_node_idxs) > 0
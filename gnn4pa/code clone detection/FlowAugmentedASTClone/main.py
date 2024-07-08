import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
from create_clone import createast, creategmndata, createseparategraph, get_datalist
import models
from torch_geometric.data import Data, DataLoader

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default="GCJ")
parser.add_argument("--graphmode", default="astandnext")
parser.add_argument("--nextsib", default=False)
parser.add_argument("--ifedge", default=False)
parser.add_argument("--whileedge", default=False)
parser.add_argument("--foredge", default=False)
parser.add_argument("--blockedge", default=False)
parser.add_argument("--nexttoken", default=False)
parser.add_argument("--nextuse", default=False)
parser.add_argument("--data_setting", default="0")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", default=1e-3)
parser.add_argument("--weight_decay", default=1e-3)
parser.add_argument("--threshold", default=0)
parser.add_argument("--resume_train", default=False)
parser.add_argument("--checkpoint_path", default="")
args = parser.parse_args()


def save_checkpoint(
    model, optimizer, epoch, save_dir="checkpoints", filename="checkpoint.pth"
):
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 构建保存路径
    save_path = os.path.join(save_dir, filename)

    # 构建checkpoint字典
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # 保存checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
        return epoch
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}

def create_batches(data):
    # random.shuffle(data)
    batches = [
        data[graph : graph + args.batch_size]
        for graph in range(0, len(data), args.batch_size)
    ]
    return batches

def test(dataset):
    # model.eval()
    with torch.no_grad():
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data, label in dataset:
            label = torch.tensor(label, dtype=torch.float, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 is not None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
            data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            prediction = model(data)
            output = F.cosine_similarity(prediction[0], prediction[1])
            # results.append(output.item())
            prediction = torch.sign(output).item()

            if prediction > args.threshold and label.item() == 1:
                tp += 1
                # print('tp')
            if prediction <= args.threshold and label.item() == -1:
                tn += 1
                # print('tn')
            if prediction > args.threshold and label.item() == -1:
                fp += 1
                # print('fp')
            if prediction <= args.threshold and label.item() == 1:
                fn += 1
                # print('fn')
        print(tp, tn, fp, fn)
        p = 0.0
        r = 0.0
        f1 = 0.0
        if tp + fp == 0:
            print("precision is none")
            return
        p = tp / (tp + fp)
        if tp + fn == 0:
            print("recall is none")
            return
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print(f"Acc:{acc}")
        print(f"P:{p}")
        print(f"R:{r}")
        print(f"F1:{f1}")
        return (acc, p, r, f1)


device = torch.device("cuda:0")
# device=torch.device('cpu')

dataset = args.dataset
datalist = get_datalist(dataset, args.data_setting)
astdict, vocablen, vocabdict = createast(dataset, datalist)
treedict = createseparategraph(
    astdict,
    vocablen,
    vocabdict,
    device,
    mode=args.graphmode,
    nextsib=args.nextsib,
    ifedge=args.ifedge,
    whileedge=args.whileedge,
    foredge=args.foredge,
    blockedge=args.blockedge,
    nexttoken=args.nexttoken,
    nextuse=args.nextuse,
)
traindata, validdata, testdata = creategmndata(
    dataset, args.data_setting, treedict, vocablen, vocabdict, device
)
# trainloder=DataLoader(traindata,batch_size=1)
num_layers = int(args.num_layers)
model = models.GMNnet(
    vocablen, embedding_dim=100, num_layers=num_layers, device=device
)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CosineEmbeddingLoss()
criterion2 = nn.MSELoss()
print(f"Total parameter: {get_parameter_number(model)}")

resume_train = args.resume_train
start_epoch = 0
if resume_train:
    checkpoint_path = args.checkpoint_path
    checkpoint_path = 'checkpoints/checkpoint.pth'
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

epochs = trange(start_epoch, args.num_epochs, leave=True, desc="Epoch")
losses = []
for epoch in epochs:  # without batching
    batches = create_batches(traindata)
    totalloss = 0.0
    main_index = 0.0

    train_acc = []

    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        batch_acc = []
        optimizer.zero_grad()
        batchloss = 0
        total = 0
        correct = 0
        for data, label in batch:
            label = torch.tensor(label, dtype=torch.float, device=device)

            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 != None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
                
            data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            # print(x1, flush=True)
            prediction = model(data)
            # batchloss=batchloss+criterion(prediction[0],prediction[1],label)
            cossim = F.cosine_similarity(prediction[0], prediction[1])
            batchloss = batchloss + criterion2(cossim[0], label)
            total += 1
            if torch.sign(cossim[0]).item() == label:
                correct += 1
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        batch_acc.append(1.0 * correct / total)
        train_acc.append(batch_acc)
        losses.append(loss)
        epochs.set_description("Epoch %d (Loss=%g)" % (epoch+1, round(loss, 5)))

    if not os.path.exists(f"{dataset}_{args.data_setting}_res"):
        os.makedirs(f"{dataset}_{args.data_setting}_res")
    with open(f"{dataset}_{args.data_setting}_res/val.txt", "a+") as f:
        res = test(validdata)
        f.write(" ".join(
            [str(epoch + 1),
            # str(train_acc),
            str(totalloss / main_index),
            "\n",
            str(res),
            "\n"])
        )
    with open(f"{dataset}_{args.data_setting}_res/test.txt", "a+") as f:
        res = test(testdata)
        f.write(" ".join([str(epoch + 1), str(res), "\n"]))

    save_checkpoint(model, optimizer, epoch, filename=f"{dataset}_GMN_{args.data_setting}_{str(epoch + 1)}.pth")
    
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
plt.show()
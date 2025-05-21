import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
#from models.rqvae import RQVAE
from models.hyper_rqvae import HypRQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

# dataset = "Games"
# ckpt_path = "/zhengbowen/rqvae_ckpt/xxxx"
# output_dir = f"/zhengbowen/data/{dataset}/"

#dataset = "Arts"
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Arts/Jan-23-2025_12-50-51/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Tiger_Arts/Jan-23-2025_15-31-56/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Arts_v5/Feb-06-2025_11-41-41/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Arts/Apr-27-2025_12-27-19/best_collision_model.pth'

#dataset = "Instruments"
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Instruments/Jan-23-2025_12-54-54/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Tiger_Instruments/Jan-23-2025_15-30-38/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v2/Jan-31-2025_11-24-09/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v3/Feb-04-2025_16-49-52/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v5/Feb-05-2025_11-45-13/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v7/Feb-05-2025_16-41-33/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v8/Feb-05-2025_16-46-22/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v9/Feb-07-2025_10-58-15/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v10/Feb-07-2025_13-27-12/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v11/Mar-06-2025_01-14-08/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v12_3/Mar-07-2025_00-54-13/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v12_4/Mar-07-2025_00-56-26/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_v14_4/Mar-07-2025_17-00-49/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_4_32/Mar-31-2025_13-16-54/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Instruments_LcRec/Apr-24-2025_16-07-54/best_collision_model.pth'


#dataset = "Games"
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Games/Jan-23-2025_12-40-19/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Tiger_Games/Jan-23-2025_14-40-53/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Games_v5/Feb-06-2025_11-45-56/best_collision_model.pth'

#dataset = "Yelp"
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Yelp/Jan-23-2025_13-37-29/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Tiger_Yelp/Jan-23-2025_14-42-27/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Yelp_v5/Feb-06-2025_12-59-21/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Yelp/Apr-27-2025_22-34-16/best_collision_model.pth'

#dataset = "Beauty"
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Beauty/Jan-23-2025_13-34-26/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Tiger_Beauty/Jan-23-2025_15-29-40/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Beauty_v5/Feb-06-2025_11-35-01/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_Beauty/Apr-27-2025_00-50-18/best_collision_model.pth'

dataset = "ml_1m"
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_4_ml_1m_kmeans/Mar-28-2025_22-14-00/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_3_ml_1m/Apr-02-2025_15-34-17/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_ml_1m_4layer_128/Apr-25-2025_13-15-31/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_ml_1m_6layer/May-11-2025_14-46-10/best_collision_model.pth'
#ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_ml_1m_7layer/May-14-2025_16-46-55/best_collision_model.pth'
ckpt_path = '/home/longfeng/projects/GenRec/LC-Rec/index/ckpt/Hyp_ml_1m_7layer/May-14-2025_16-48-21/best_collision_model.pth'


output_dir = f"/home/longfeng/projects/GenRec/LC-Rec/data/{dataset}/"
output_file = f"{dataset}.index_hyper_8layer.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)

model = HypRQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>", "<g_{}>", "<h_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)



with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)

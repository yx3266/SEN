'''
Perform semactic clustering for textual evidence of sentence type.
Save as caption_cluster.npz in the inverse_search path.
'''
import os
import json
import argparse
import torch
import torchvision
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cosine
import dataset_compute_emb

parser = argparse.ArgumentParser(description='Perform semantic clustering for textual evidence of sentence type')
parser.add_argument('--queries_dataset_root', type=str, default='../evidence/queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')

parser.add_argument('--visual_news_root', type=str, default='../../news_clippings/visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')

parser.add_argument('--news_clip_root', type=str,
                    default='../../news_clippings/news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')

parser.add_argument('--dataset_items_file', type=str, default='../evidence/queries_dataset/dataset_items_',
                    help='location to the dataset items file')

parser.add_argument('--domains_file', type=str, default='../evidence/queries_dataset/domain_to_idx_dict.json',
                    help='location to the domains to idx file')

parser.add_argument('--split', type=str, default='test',
                    help='which split to compute the embeddings for')

parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be start from 0')

parser.add_argument('--threshold', type=float, default=0.5, help='the clustering threshold')

args = parser.parse_args()

args.dataset_items_file = args.dataset_items_file + args.split+".json"

print("Performing semantic clustering for: " + args.split)
print("Reading items from: " + args.dataset_items_file)

# solve the situation in which there are more than one max_clusters at the same time
def solve_max(all_feature, clusters, max_clusters):
    cos_result = []
    for i in range(len(max_clusters)):
        for j in range(len(clusters)):
            if clusters[j] == max_clusters[i]:
                cos_result.append(1 - cosine(all_feature[j], all_feature[-1]))
                break
    return max_clusters[cos_result.index(max(cos_result))]


# find the three class cluster
def find_cluster(all_feature, clusters):
    count = np.bincount(clusters[:-1])
    sim_clusters = clusters[-1]
    max_clusters = [i for i, j in enumerate(count) if j == max(count)]
    if len(max_clusters) > 1:
        max_clusters = solve_max(all_feature, clusters, max_clusters)
    sim_class = []
    max_class = []
    other_class = []

    # find sim
    for i in range(clusters.shape[0] - 1):
        if clusters[i] == sim_clusters:
            sim_class.append(i)
        if clusters[i] == max_clusters:
            max_class.append(i)
        if (clusters[i] != sim_clusters) and (clusters[i] != max_clusters):
            other_class.append(i)
    return np.array(sim_class), np.array(max_class), np.array(other_class)

# load items file
context_data_items_dict = json.load(open(args.dataset_items_file))

# transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset file
dataset = dataset_compute_emb.NewsContextDataset(context_data_items_dict, args.visual_news_root,
                                                 args.queries_dataset_root, args.news_clip_root, args.domains_file,
                                                 args.split, transform)

if args.start_idx != -1:
    start_idx = args.start_idx
else:
    start_idx = 0

for i in range(start_idx, len(dataset)):
    key = list(context_data_items_dict.keys())[i]
    print("item number: " + str(i) + ", key: " + str(key))
    inverse_dir = os.path.join(args.queries_dataset_root, context_data_items_dict[key]['inv_path'])
    if not os.path.isdir(inverse_dir):
        print("########## dir not exists ##########")
        continue
    # captions may not exist
    if not os.path.isfile(os.path.join(inverse_dir, "caption_features2.npz")):
        print("********** captions not exist **********")
        continue
    else:
        try:
            # load features
            caption_feature = np.load(os.path.join(inverse_dir, "caption_features2.npz"))["arr_0"]
            query_feature = np.load(os.path.join(inverse_dir, "qCap_sbert_features.npz"))["arr_0"]
            all_feature = np.concatenate((caption_feature, query_feature), axis=0)
            # clustering
            clusters = fclusterdata(X=all_feature, t=args.threshold, criterion='distance', metric='cosine')
            sim_class, max_class, other_class = find_cluster(all_feature, clusters)
            # that is, sim_class: Supporting Cluster, max_class: Representative Cluster, other_class: Complementary Cluster
            np.savez(os.path.join(inverse_dir, "caption_cluster.npz"), sim_class=sim_class, max_class=max_class,
                     other_class=other_class)
        except:
            pass
'''
Compute support-refutation score for each piece of textual evidence.
Save as entities/captions_supporting_score.npz in the inverse_search path.
'''
import os
import json
import argparse
import numpy as np
import math
from collections import Counter
from rapidfuzz import fuzz, process
import torch
import torch.nn as nn
import torchvision
import dataset_compute_emb

parser = argparse.ArgumentParser(description='Compute support-refutation score')
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

parser.add_argument('--fuzz_ratio', type=float, default=90.0,
                    help='fuzz ratio threshold for treating an entity and another entity as one entity')

parser.add_argument('--entity_threshold', type=int, default=2,
                    help='if the number of the occurrences of an entity is less than the entity_threshold, it is not considered')

args = parser.parse_args()

args.dataset_items_file = args.dataset_items_file + args.split+".json"
print("Precomputing support score features for: " + args.split)
print("Reading items from: " + args.dataset_items_file)

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

def get_entity_list(entity_keyword):
    query_entity_list = []
    for ent in entity_keyword['query_entity']:
        if len(ent) > 0:
            query_entity_list.append(ent[0])
    entities_entity_list = []
    if 'entities_entity' in entity_keyword.keys():
        for ent_ in entity_keyword['entities_entity']:
            entities_entity_list.append([])
            for ent in ent_:
                if len(ent) > 0:
                    entities_entity_list[-1].append(ent[0])
    captions_entity_list = []
    if 'captions_entity' in entity_keyword.keys():
        for ent_ in entity_keyword['captions_entity']:
            captions_entity_list.append([])
            for ent in ent_:
                if len(ent) > 0:
                    captions_entity_list[-1].append(ent[0])
    return query_entity_list, captions_entity_list, entities_entity_list

def compute_score(query_entity_list, entity_list, counter, count, is_entities=False):
    sort_list = sorted(counter.items(), key = lambda s: (-s[-1]))
    sort_entity_list = [ent[0] for ent in sort_list]
    # print(sort_list)
    # print(sort_entity_list)
    score_list = [0] * len(entity_list)
    frequency_threshold = min(2, len(entity_list))   # or frequency_threshold = 2 if is_entities else min(2, len(entity_list)), almost no difference in the results
    for i in range(len(entity_list)):   # the i-th caption's entity list
        if len(entity_list) == 0:
            continue
        result = process.cdist(query_entity_list, entity_list[i], scorer=fuzz.partial_ratio)
        # determine the scale and compute the score
        for j in range(len(entity_list[i])):
            if (result[:, j] > 90.0).any() == True:
                score_list[i] += 1
        scale = 2 if score_list[i] >= 1 else 1
        # or the following code, almost no difference in the results
        # scale = 1
        # if np.sum(result>=90.0):
        #     score_list[i] = np.sum(result>=90.0)
        #     scale = 2
        for j in range(len(entity_list[i])):
            if (result[:, j] > 90.0).any() == True:
                pass
            else:
                # deduct score according to the word frequency
                if counter[entity_list[i][j]] >= frequency_threshold:
                    score_list[i] -= math.exp(-math.sqrt(sort_entity_list.index(entity_list[i][j])))/scale
    return np.array(score_list)

for i in range(start_idx, len(dataset)):
    key = list(context_data_items_dict.keys())[i]
    print("item number: " + str(i) + ", key: " + str(key))
    inverse_dir = os.path.join(args.queries_dataset_root, context_data_items_dict[key]['inv_path'])
    if not os.path.isdir(inverse_dir):
        print("########## dir not exists ##########")
        continue

    entity_keyword_path = os.path.join(inverse_dir, "entity_keyword.json")
    if not os.path.isfile(entity_keyword_path):
        print("********** file not exists **********")
        continue

    # get the three entity list
    entity_keyword = json.load(open(entity_keyword_path, 'r'))
    query_entity_list, captions_entity_list, entities_entity_list = get_entity_list(entity_keyword)

    # if entities entity list is not empty
    if entities_entity_list != []:
        entities_counter = Counter([ent for ent_ in entities_entity_list for ent in ent_])
        entities_count = len([ent for ent_ in entities_entity_list for ent in ent_])
        entities_score = compute_score(query_entity_list, entities_entity_list, entities_counter, entities_count, True)
        # print(entities_score)
        np.savez(os.path.join(inverse_dir, "entities_supporting_score.npz"), entities_score=entities_score)
    else:
        print("entities not exist")
    # if captions is not empty
    if captions_entity_list != []:
        captions_counter = Counter([ent for ent_ in captions_entity_list for ent in ent_])
        captions_count = len([ent for ent_ in captions_entity_list for ent in ent_])
        captions_score = compute_score(query_entity_list, captions_entity_list, captions_counter, captions_count)
        # print(captions_score)
        np.savez(os.path.join(inverse_dir, "captions_supporting_score.npz"), captions_score=captions_score)
    else:
        print("captions not exist")
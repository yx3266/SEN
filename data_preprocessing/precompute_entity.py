'''
Perform named entity recognition and keyword extraction for caption claim and textual evidence.
Save as entity_keyword.json in the inverse_search path.
The extracted keywords are not used.
'''
import numpy as np
import json
import torch
import torch.nn as nn
import dataset_compute_emb
import torchvision
import os
import argparse
import io
import spacy
import pytextrank

NER = spacy.load("en_core_web_sm")
NER.add_pipe('textrank')

parser = argparse.ArgumentParser(description='Precompute entities')
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

parser.add_argument('--split', type=str, default='train',
                    help='which split to compute the embeddings for')

parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be start from 0')


def compare_entities(text1, text2):
    for word in text1.ents:
        for word2 in text2.ents:
            if word.text.lower() == word2.text.lower() and word.label_ == word2.label_:
                return 1
    return 0


def find_entities_overlap(text_list, query_ner):
    overlap = []
    for item in text_list:
        item_ner = NER(item)
        overlap.append(compare_entities(item_ner, query_ner))
    return np.asarray(overlap)


args = parser.parse_args()
args.dataset_items_file = args.dataset_items_file + args.split+".json"
print("Precomputing entities for: " + args.split)
print("Reading items from: " + args.dataset_items_file)


def save_features(features, full_file_path):
    if torch.is_tensor(features):
        np.savez_compressed(full_file_path, features.detach().cpu().numpy())
    else:
        np.savez_compressed(full_file_path, features)


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
    sample = dataset.__getitemNoimgs__(i)

    query_text = sample['qCap']
    query_ner = NER(query_text)
    inv_path_item = os.path.join(args.queries_dataset_root, context_data_items_dict[key]['inv_path'])
    save_path = os.path.join(args.queries_dataset_root, context_data_items_dict[key]['inv_path'], "entity_keyword.json")
    f = open(save_path, "w")
    entity_keyword_dict = dict()
    entity_keyword_dict["query_entity"] = [(ent.text, ent.label_) for ent in query_ner.ents]
    entity_keyword_dict["query_keyword"] = [phrase.text for phrase in query_ner._.phrases]

    if sample['entities']:
        entity_keyword_dict["entities_entity"] = []
        for entity in sample['entities']:
            try:
                entity_ner = NER(entity)
                entity_keyword_dict["entities_entity"].append([(ent.text, ent.label_) for ent in entity_ner.ents])
            except:
                entity_keyword_dict["entities_entity"].append([])

    if sample['caption']:
        entity_keyword_dict["captions_entity"] = []
        entity_keyword_dict["captions_keyword"] = []
        for caption in sample['caption']:
            try:
                caption_ner = NER(caption)
                entity_keyword_dict["captions_entity"].append([(ent.text, ent.label_) for ent in caption_ner.ents])
                entity_keyword_dict["captions_keyword"].append([phrase.text for phrase in caption_ner._.phrases])
            except:
                entity_keyword_dict["captions_entity"].append([])
                entity_keyword_dict["captions_keyword"].append([])

    json.dump(entity_keyword_dict, f)
    f.close()




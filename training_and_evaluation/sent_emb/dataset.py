import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import json
import os
    
class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, visual_news_root_dir, queries_root_dir, news_clip_root_dir, domain_to_idx_dict, split, sent_emb_dim, load_clip_for_queries=False, load_clip_for_all=False, ner_ent='', ner_cap='', filter_duplicates=False,labels_overlap=False, img_cluster=False, places_cluster=False, cap_cluster=False, ent_cluster=False ):
        self.context_data_items_dict = context_data_items_dict
        self.visual_news_root_dir = visual_news_root_dir
        self.queries_root_dir = queries_root_dir
        self.news_clip_root_dir = news_clip_root_dir
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.domain_to_idx_dict = domain_to_idx_dict
        self.news_clip_data_dict = json.load(open(os.path.join(self.news_clip_root_dir,split+".json")))["annotations"]
        self.sent_emb_dim = sent_emb_dim        
        self.load_clip_for_queries = load_clip_for_queries #whether to load precomputed clip embeddings for queries only 
        self.load_clip_for_all = load_clip_for_all #whether to load precomputed clip embeddings for queries only
        # support-refutation score or binary feature
        self.ner_cap = ner_cap
        self.ner_ent = ner_ent
        self.filter_duplicates = filter_duplicates
        self.labels_overlap = labels_overlap

        # stance extraction score
        self.img_cluster = img_cluster
        self.places_cluster = places_cluster
        self.cap_cluster = cap_cluster
        self.ent_cluster = ent_cluster

    def __len__(self):
        return len(self.context_data_items_dict)   

    def load_entities(self,ent_file_path,ent_ner_path):
        if os.path.isfile(ent_file_path): 
            entities_file = np.load(ent_file_path)
            entities = torch.from_numpy(entities_file['arr_0'])   # num * 768
            if self.ner_ent != '':
                ent_ner_file = np.load(ent_ner_path)
                ent_ner = torch.from_numpy(ent_ner_file['arr_0' if self.ner_ent == 'binary' else 'entities_score'])   # num * 1
                entities = torch.cat((entities, ent_ner.unsqueeze(1)),dim=1)   # num * (768+1)
        else:
            entities = torch.empty( (0,self.sent_emb_dim) ) if self.ner_ent == '' else torch.empty( (0,self.sent_emb_dim+1) )
        return entities 
    
    def get_domain_idx(self,domain):
        if domain in self.domain_to_idx_dict.keys(): 
            domain_idx = self.domain_to_idx_dict[domain]
        else:
            domain_idx = self.domain_to_idx_dict['UNK']
        return domain_idx
            
    def load_imgs_direct_search(self,imgs_file_path, metadata_file_path, imgs_to_keep_path, labels_overlap_file, label,places=False, cluster_path=None):
        img_dim = (2048+1) if self.labels_overlap else 2048
        # may not be used
        if not os.path.isfile(imgs_file_path):
            imgs = torch.empty((0, img_dim)) if places else torch.empty((0, img_dim, 7, 7))
            domains = torch.empty((0))
            if places: return imgs
            return imgs, domains, [0, 0, 0]
        #
        if self.filter_duplicates and label==0:
            imgs_to_keep_idx = json.load(open(imgs_to_keep_path))['index_of_images_tokeep']
            if len(imgs_to_keep_idx) == 0:
                imgs = torch.empty((0,img_dim)) if places else torch.empty((0,img_dim,7,7)) 
                domains = torch.empty((0))
                if places: return imgs
                return imgs, domains, [0, 0, 0]
            imgs_to_keep_idx = np.asarray([int(i) for i in imgs_to_keep_idx])   
        imgs = np.load(imgs_file_path)['arr_0']
        # not img_cluster, places or imgs
        if self.img_cluster == False:
            if self.filter_duplicates and label==0: imgs = imgs[imgs_to_keep_idx,:]
            imgs = torch.from_numpy(imgs)   # num * 2048 * 7 * 7 or num * 2048
            if self.labels_overlap:
                labels_overlap_feature = np.load(labels_overlap_file)['arr_0']
                if self.filter_duplicates and label==0: labels_overlap_feature=labels_overlap_feature[imgs_to_keep_idx]
                labels_overlap_feature = torch.from_numpy(labels_overlap_feature).float()   # num
                if places:
                    imgs = torch.cat( (imgs,labels_overlap_feature.unsqueeze(1)), dim=1)   # num * 2049
                    #print(imgs.shape)
                    #print('***')
                else:
                    labels_overlap_feature = labels_overlap_feature.unsqueeze(1).unsqueeze(2).unsqueeze(3)   # num * 1 * 1 * 1
                    labels_overlap_feature = labels_overlap_feature.expand(-1,-1,7,7)   # num * 1 * 7 * 7
                    imgs = torch.cat((imgs,labels_overlap_feature),dim=1)    # num * 2049 * 7 * 7
            if places: return imgs
            imgs_metadata = json.load(open(metadata_file_path))
            domains = []
            for i in range(0, len(imgs_metadata)):
                if self.filter_duplicates and label==0:
                    if i not in imgs_to_keep_idx: continue
                domain_idx = self.get_domain_idx(imgs_metadata[str(i)]['domain'])
                domains.append(domain_idx)
            domains = torch.as_tensor(domains)   # num
            return imgs, domains, [0, 0, 0]
        else:   # img_cluster == True, places or imgs
            if not os.path.isfile(cluster_path):
                imgs = torch.empty((0, img_dim)) if places else torch.empty((0, img_dim, 7, 7))
                domains = torch.empty((0))
                if places: return imgs
                return imgs, domains, [0, 0, 0]
            sim_class = np.load(cluster_path)['sim_class']
            max_class = np.load(cluster_path)['max_class']
            other_class = np.load(cluster_path)['other_class']
            if self.filter_duplicates and label == 0:
                sim_class = np.intersect1d(sim_class, imgs_to_keep_idx)
                max_class = np.intersect1d(max_class, imgs_to_keep_idx)
                other_class = np.intersect1d(other_class, imgs_to_keep_idx)
            sim_cluster = torch.from_numpy(imgs[sim_class, ]) if len(sim_class) != 0 else torch.empty((0, 2048, 7, 7)) if not places else torch.empty((0, 2048))
            max_cluster = torch.from_numpy(imgs[max_class, ]) if len(max_class) != 0 else torch.empty((0, 2048, 7, 7)) if not places else torch.empty((0, 2048))
            other_cluster = torch.from_numpy(imgs[other_class, ]) if len(other_class) != 0 else torch.empty((0, 2048, 7, 7)) if not places else torch.empty((0, 2048))
            if self.labels_overlap:
                labels_overlap_feature = torch.from_numpy(np.load(labels_overlap_file)['arr_0']).float()
                if places:
                    sim_class_overlap = labels_overlap_feature[sim_class, ] if len(sim_class) != 0 else torch.empty((0))
                    max_class_overlap = labels_overlap_feature[max_class, ] if len(max_class) != 0 else torch.empty((0))
                    other_class_overlap = labels_overlap_feature[other_class, ] if len(other_class) != 0 else torch.empty((0))
                    sim_cluster = torch.cat((sim_cluster, sim_class_overlap.unsqueeze(1)), dim=1)
                    max_cluster = torch.cat((max_cluster, max_class_overlap.unsqueeze(1)), dim=1)
                    other_cluster = torch.cat((other_cluster, other_class_overlap.unsqueeze(1)), dim=1)
                else:
                    labels_overlap_feature = labels_overlap_feature.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    labels_overlap_feature = labels_overlap_feature.expand(-1, -1, 7, 7)
                    sim_class_overlap = labels_overlap_feature[sim_class, ] if len(sim_class) != 0 else torch.empty((0, 1, 7, 7))
                    max_class_overlap = labels_overlap_feature[max_class, ] if len(max_class) != 0 else torch.empty((0, 1, 7, 7))
                    other_class_overlap = labels_overlap_feature[other_class, ] if len(other_class) != 0 else torch.empty((0, 1, 7, 7))
                    sim_cluster = torch.cat((sim_cluster, sim_class_overlap), dim=1)
                    max_cluster = torch.cat((max_cluster, max_class_overlap), dim=1)
                    other_cluster = torch.cat((other_cluster, other_class_overlap), dim=1)
            imgs = torch.cat((sim_cluster, max_cluster, other_cluster), dim=0)
            if places:
                return imgs

            imgs_metadata = json.load(open(metadata_file_path))
            domains = []
            for i in range(0, len(imgs_metadata)):
                domain_idx = self.get_domain_idx(imgs_metadata[str(i)]['domain'])
                domains.append(domain_idx)
            domains = torch.as_tensor(domains)  # num
            sim_domains = domains[sim_class] if len(sim_class) != 0 else torch.empty(0)
            max_domains = domains[max_class] if len(max_class) != 0 else torch.empty(0)
            other_domains = domains[other_class] if len(other_class) != 0 else torch.empty(0)
            domains = torch.cat((sim_domains, max_domains, other_domains), dim=0)
            return imgs, domains, [sim_class.shape[0], max_class.shape[0], other_class.shape[0]]



    def load_queries(self,qcap_path, qimg_path, qplace_path=None):
        caption_file = np.load(qcap_path)
        caption = torch.from_numpy(caption_file['arr_0'])

        img_file = np.load(qimg_path)
        img = torch.from_numpy(img_file['arr_0'])
        if qplace_path:
            places_file = np.load(qplace_path)
            qplaces = torch.from_numpy(places_file['arr_0'])
            return caption, img, qplaces
        return caption, img
        
    def load_captions(self,captions_emb_path, captions_info_path, captions_ner_path, captions_to_keep_path, captions_cluster_path):
        if not os.path.isfile(captions_emb_path):
            captions_emb = torch.empty((0,self.sent_emb_dim)) if self.ner_cap == '' else torch.empty((0,self.sent_emb_dim+1))
            domains = torch.empty((0))
            return captions_emb, domains, [0, 0, 0]
        if self.filter_duplicates:
            captions_to_keep_idx = json.load(open(captions_to_keep_path))['index_of_captions_tokeep']
            if len(captions_to_keep_idx) == 0:            
                captions_emb = torch.empty((0,self.sent_emb_dim)) if self.ner_cap == '' else torch.empty((0,self.sent_emb_dim+1))
                domains = torch.empty((0))
                return captions_emb, domains, [0, 0, 0]
            captions_to_keep_idx = np.asarray([int(i) for i in captions_to_keep_idx])                
        captions_emb = np.load(captions_emb_path)['arr_0']   # num * 768
        if not self.cap_cluster:
            try:
                if self.filter_duplicates: captions_emb = captions_emb[captions_to_keep_idx,:]
            except:
                print(captions_emb_path)
            captions_emb = torch.from_numpy(captions_emb)
            if self.ner_cap != '':
                captions_ner = np.load(captions_ner_path)['arr_0' if self.ner_cap == 'binary' else 'captions_score']
                if self.filter_duplicates: captions_ner=captions_ner[captions_to_keep_idx]
                captions_ner = torch.from_numpy(captions_ner)   # num
                captions_emb = torch.cat((captions_emb, captions_ner.unsqueeze(1)),dim=1)

            domains = []
            cap_domains = json.load(open(captions_info_path))['domains']
            for i in range(0,len(cap_domains)):
                if self.filter_duplicates and i not in captions_to_keep_idx: continue
                domain_idx = self.get_domain_idx(cap_domains[str(i)])
                domains.append(domain_idx)
            domains = torch.as_tensor(domains)   # num
            return captions_emb, domains, [0, 0, 0]
        else:
            if not os.path.isfile(captions_cluster_path):
                captions_emb = torch.empty((0, self.sent_emb_dim)) if self.ner_cap == '' else torch.empty((0, self.sent_emb_dim + 1))
                domains = torch.empty((0))
                return captions_emb, domains, [0, 0, 0]
            sim_class = np.load(captions_cluster_path)['sim_class']
            max_class = np.load(captions_cluster_path)['max_class']
            other_class = np.load(captions_cluster_path)['other_class']
            if self.filter_duplicates:
                sim_class = np.intersect1d(sim_class, captions_to_keep_idx)
                max_class = np.intersect1d(max_class, captions_to_keep_idx)
                other_class = np.intersect1d(other_class, captions_to_keep_idx)
            sim_cluster = torch.from_numpy(captions_emb[sim_class, :]) if len(sim_class) != 0 else torch.empty((0, self.sent_emb_dim))
            max_cluster = torch.from_numpy(captions_emb[max_class, :]) if len(max_class) != 0 else torch.empty((0, self.sent_emb_dim))
            other_cluster = torch.from_numpy(captions_emb[other_class, :]) if len(other_class) != 0 else torch.empty((0, self.sent_emb_dim))
            if self.ner_cap != '':   # if self.filter_duplicates, the code gets the three filter_class, and then get the three filter_binary
                captions_ner = np.load(captions_ner_path)['arr_0' if self.ner_cap == 'binary' else 'captions_score']
                sim_class_ner = torch.from_numpy(captions_ner[sim_class]) if len(sim_class) != 0 else torch.empty((0))
                max_class_ner = torch.from_numpy(captions_ner[max_class]) if len(max_class) != 0 else torch.empty((0))
                other_class_ner = torch.from_numpy(captions_ner[other_class]) if len(other_class) != 0 else torch.empty((0))
                sim_cluster = torch.cat((sim_cluster, sim_class_ner.unsqueeze(1)), dim=1)
                max_cluster = torch.cat((max_cluster, max_class_ner.unsqueeze(1)), dim=1)
                other_cluster = torch.cat((other_cluster, other_class_ner.unsqueeze(1)), dim=1)
            captions_emb = torch.cat((sim_cluster, max_cluster, other_cluster), dim=0)

            domains = []
            cap_domains = json.load(open(captions_info_path))['domains']
            for i in range(0, len(cap_domains)):
                # if self.filter_duplicates and i not in captions_to_keep_idx: continue
                domain_idx = self.get_domain_idx(cap_domains[str(i)])
                domains.append(domain_idx)
            domains = torch.as_tensor(domains)  # num
            sim_domains = domains[sim_class] if len(sim_class) != 0 else torch.empty(0)
            max_domains = domains[max_class] if len(max_class) != 0 else torch.empty(0)
            other_domains = domains[other_class] if len(other_class) != 0 else torch.empty(0)
            domains = torch.cat((sim_domains, max_domains, other_domains), dim=0)
            return captions_emb, domains, [sim_class.shape[0], max_class.shape[0], other_class.shape[0]]

    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()               
        key = self.idx_to_keys[idx]
        label = torch.as_tensor(1) if self.news_clip_data_dict[int(key)]['falsified'] else torch.as_tensor(0)
        direct_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['direct_path'])
        inverse_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['inv_path'])

        #load entities        
        entities_file_path = os.path.join(inverse_path_item, 'entities_clip_features.npz' if self.load_clip_for_all else 'entities_features2.npz')
        entities_ner_file_path = os.path.join(inverse_path_item, 'entities_binary_feature2.npz' if self.ner_ent == 'binary' else 'entities_supporting_score.npz')
        entities = self.load_entities(entities_file_path,entities_ner_file_path)
        
        #load captions 
        cap_file_path = os.path.join(inverse_path_item, 'captions_clip_features.npz' if self.load_clip_for_all else 'caption_features2.npz')
        cap_info_file_path = os.path.join(inverse_path_item, 'captions_info')
        cap_ner_file_path = os.path.join(inverse_path_item, 'captions_binary_feature2.npz' if self.ner_cap == 'binary' else 'captions_supporting_score.npz')
        cap_filter_mask = os.path.join(inverse_path_item, 'captions_to_keep_idx')
        cap_cluster_file_path = os.path.join(inverse_path_item, 'caption_cluster.npz')
        captions,captions_domains, captions_cluster_num = self.load_captions(cap_file_path, cap_info_file_path,cap_ner_file_path,cap_filter_mask, cap_cluster_file_path)

        #load imgs   
        imgs_emb_file_path = os.path.join(direct_path_item, 'imgs_clip_features.npz' if self.load_clip_for_all else 'resnet_features.npz') 
        imgs_info_file_path = os.path.join(direct_path_item, 'metadata_of_features') 
        imgs_filter_mask = os.path.join(direct_path_item, 'imgs_to_keep_idx') 
        labels_overlap_file = os.path.join(inverse_path_item, 'labels_overlap.npz')
        imgs_cluster_file_path = os.path.join(inverse_path_item, "image_cluster.npz")   # used in both visual sens
        imgs,imgs_domains, imgs_cluster_num = self.load_imgs_direct_search(imgs_emb_file_path,imgs_info_file_path,imgs_filter_mask,labels_overlap_file,label, cluster_path=imgs_cluster_file_path)

        #load resnet_places 
        places_emb_file_path = os.path.join(direct_path_item, 'places_resnet_features.npz') 
        places_emb = self.load_imgs_direct_search(places_emb_file_path,imgs_info_file_path,imgs_filter_mask,labels_overlap_file,label,places=True, cluster_path=imgs_cluster_file_path)
        #load query         
        qImg_path = os.path.join(inverse_path_item, 'qImg_clip_features.npz' if self.load_clip_for_all else 'qImg_resnet_features.npz')
        qImg_places_path = os.path.join(inverse_path_item, 'qImg_places_resnet_features.npz')

        qCap_path = os.path.join(inverse_path_item, 'qCap_clip_features.npz' if self.load_clip_for_all else 'qCap_sbert_features.npz')
        qCap, qImg, qPlaces =  self.load_queries(qCap_path, qImg_path, qImg_places_path)
            
        if self.load_clip_for_queries:
            qImg_path_clip = os.path.join(inverse_path_item, 'qImg_clip_features.npz')
            qCap_path_clip = os.path.join(inverse_path_item, 'qCap_clip_features.npz')
            qCap_clip, qImg_clip =  self.load_queries(qCap_path_clip, qImg_path_clip)

        sample = {'label': label, 'entities':entities.float(), 'caption': captions.float(), 'caption_domains': captions_domains, 'cap_cluster': self.cap_cluster, 'caption_cluster_num': captions_cluster_num, 'imgs': imgs.float(), 'imgs_domains': imgs_domains, 'img_cluster': self.img_cluster, 'img_cluster_num': imgs_cluster_num, 'places_mem': places_emb, 'qImg': qImg.float(), 'qCap': qCap.float(), 'qPlaces': qPlaces}
        if self.load_clip_for_queries:   
            sample['qImg_clip'] = qImg_clip.float() 
            sample['qCap_clip'] = qCap_clip.float()             
        return sample, entities.size(0), captions.size(0), imgs.size(0),  captions_cluster_num, imgs_cluster_num


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,padding_idx, embed.max_norm, embed.norm_type,embed.scale_grad_by_freq,embed.sparse)
    return X
  
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class ContextSEN(nn.Module):
    def __init__(self, settings, clip_model=None):
        super(ContextSEN, self).__init__()
        self.use_img_memory = False
        self.use_places_memory = False
        self.use_ent_memory = False
        self.use_cap_memory = False
        self.consistency = False 
        self.use_src = False 
        self.ner_cap = settings['ner_cap']
        self.ner_ent = settings['ner_ent']

        self.fusion = settings['fusion']        
        self.pdrop = settings['pdrop']
        self.pdrop_mem = settings['pdrop_mem']
        self.inp_pdrop = settings['inp_pdrop']
        self.emb_pdrop = settings['emb_pdrop']
        self.lockdrop = LockedDropout()
        self.use_clip_for_all = settings['use_clip_for_all']
        
        #source embeddings  
        if settings['use_src'] == True:
            self.use_src = True 
            self.domains_num = settings['domains_num']
            self.domains_dim = settings['domains_dim']
            self.W_domains_emb = nn.Embedding(self.domains_num,self.domains_dim)

        #cluster   ########################################
        self.img_cluster = settings['img_cluster']
        self.places_cluster = settings['places_cluster']
        self.cap_cluster = settings['cap_cluster']
        self.ent_cluster = settings['ent_cluster']


        #Image memory 
        self.img_dim_in = settings['img_dim_in']
        self.img_dim_out = settings['img_dim_out']
        if settings['use_img_memory']==True: 
            self.use_img_memory = True 
            self.img_mem_hops = settings['img_mem_hops']
            if self.img_mem_hops > 1:
                self.Wqimg_hop = nn.Linear(self.img_dim_out,self.img_dim_out)
            img_dim_in = (self.img_dim_in+self.domains_dim) if settings['use_src'] else self.img_dim_in
            img_dim_in = (img_dim_in+1) if settings['labels_overlap'] else img_dim_in
            self.W_im_c = nn.Linear(img_dim_in,self.img_dim_out) 
            self.W_im_a = nn.Linear(img_dim_in,self.img_dim_out)

            # cluster   ########################################
            if self.img_cluster:
                self.W_im_cluster = nn.Linear(self.img_dim_out * 3, self.img_dim_out)
                self.bn_img_cluster = torch.nn.ModuleList([nn.BatchNorm1d(num_features=self.img_dim_out) for _ in range(3)])
            
            if settings['fusion'] == 'byDecision':
                self.W_dec_img_mem = nn.Linear(self.img_dim_out,1)
            self.bn_imgMem = nn.BatchNorm1d(num_features=self.img_dim_out)

        #Resnet places memory
        self.places_dim_in = settings['places_dim_in']
        self.places_dim_out = settings['places_dim_out']
        if settings['use_places_memory']==True: 
            self.use_places_memory = True 
            self.places_mem_hops = settings['places_mem_hops']
            if self.places_mem_hops > 1:
                self.Wqplaces_hop = nn.Linear(self.places_dim_in,self.places_dim_out)
            places_dim_in = (self.places_dim_in+self.domains_dim) if settings['use_src'] else self.places_dim_in
            places_dim_in = (places_dim_in+1) if settings['labels_overlap'] else places_dim_in

            self.W_places_c = nn.Linear(places_dim_in,self.places_dim_out) 
            self.W_places_a = nn.Linear(places_dim_in,self.places_dim_out)

            # cluster   ########################################
            if self.places_cluster:
                self.W_places_cluster = nn.Linear(self.places_dim_out * 3, self.places_dim_out)
                self.bn_places_cluster = torch.nn.ModuleList([nn.BatchNorm1d(num_features=self.places_dim_out) for _ in range(3)])
            
            if settings['fusion'] == 'byDecision':
                self.W_dec_places_mem = nn.Linear(self.places_dim_out,1)
            self.bn_placesMem = nn.BatchNorm1d(num_features=self.places_dim_out)
               
        #Captions memory - sentences embeddings
        self.sent_emb_dim_in = settings['sent_emb_dim_in']
        self.sent_emb_dim_out = settings['sent_emb_dim_out']
        if settings['use_cap_memory']==True: 
            self.use_cap_memory = True 
            self.cap_mem_hops = settings['cap_mem_hops']
            if self.cap_mem_hops > 1:
                self.Wqcap_hop = nn.Linear(self.sent_emb_dim_out,self.sent_emb_dim_out)
            sent_emb_dim_in = self.sent_emb_dim_in if not self.ner_cap else (self.sent_emb_dim_in+1)
            sent_emb_dim_in = (sent_emb_dim_in+self.domains_dim) if settings['use_src'] else sent_emb_dim_in           
            self.W_sent_c = nn.Linear(sent_emb_dim_in, self.sent_emb_dim_out)
            self.W_sent_a = nn.Linear(sent_emb_dim_in, self.sent_emb_dim_out)

            # cluster   ########################################
            if self.cap_cluster:
                self.W_sent_cluster = nn.Linear(self.sent_emb_dim_out * 3, self.sent_emb_dim_out)
                self.bn_cap_cluster = torch.nn.ModuleList([nn.BatchNorm1d(num_features=self.sent_emb_dim_out) for _ in range(3)])

            if settings['fusion'] == 'byDecision':                
                self.W_dec_cap_mem = nn.Linear(self.sent_emb_dim_out,1)   
            self.bn_capMem = nn.BatchNorm1d(num_features=self.sent_emb_dim_out)
          
        #Entities memory 
        self.ent_dim_in = settings['ent_dim_in']
        self.ent_dim_out = settings['ent_dim_out']
        if settings['use_ent_memory']==True:
            self.use_ent_memory = True 
            self.ent_mem_hops = settings['ent_mem_hops']
            ent_dim_in = self.ent_dim_in if not self.ner_ent else self.ent_dim_in+1
            #mapping was not defined before 
            if self.ent_mem_hops > 1:
                self.Wqent_hop = nn.Linear(self.ent_dim_out,self.ent_dim_out)
                
            self.W_ent_c = nn.Linear(ent_dim_in, self.ent_dim_out)
            self.W_ent_a = nn.Linear(ent_dim_in, self.ent_dim_out)

            # cluster   ########################################
            if self.ent_cluster:
                self.W_ent_cluster = nn.Linear(self.ent_dim_out * 3, self.ent_dim_out)
                self.bn_ent_cluster = torch.nn.ModuleList([nn.BatchNorm1d(num_features=self.ent_dim_out) for _ in range(3)])

            if settings['fusion'] == 'byDecision':  
                self.W_dec_ent_mem = nn.Linear(self.ent_dim_out,1) 
            self.bn_entMem = nn.BatchNorm1d(num_features=self.ent_dim_out)

        #Query embeddings (for memory modules)
        if self.use_img_memory: self.W_qImg = nn.Linear(self.img_dim_in,self.img_dim_out)
        if self.use_places_memory: self.W_qPlaces = nn.Linear(self.places_dim_in,self.places_dim_out)
        
            
        if settings['consistency'] == 'clip':
            #use CLIP for classification 
            self.consistency = 'clip'
            self.bn_clip = nn.BatchNorm1d(num_features=512)

            if settings['fusion'] == 'byDecision':
                self.W_consis_clip = nn.Linear(512,1)

        #final classification layer         
        if settings['fusion'] == 'byDecision': 
            count_branches = 0
            if self.use_img_memory:  count_branches += 1  
            if self.use_cap_memory:  count_branches += 1
            if self.use_ent_memory:  count_branches += 1
            if self.use_places_memory:  count_branches += 1
            if self.consistency:  count_branches += 1
            self.W_final_dec = nn.Linear(count_branches,1)
        elif settings['fusion'] == 'byFeatures': 
            count_dim = 0 
            if self.use_img_memory: count_dim += self.img_dim_out
            if self.use_cap_memory: count_dim += self.sent_emb_dim_out
            if self.use_ent_memory: count_dim += self.ent_dim_out
            if self.use_places_memory: count_dim += self.places_dim_out
            if self.consistency == 'clip': count_dim += 512 

            self.nlayers = settings['nlayers']
            self.fc_dims = settings['fc_dims']           
            self.W_final_dec =  [nn.Linear(count_dim if l == 0 else self.fc_dims[l-1], self.fc_dims[l] if l != self.nlayers - 1 else 1) for l in range(self.nlayers)]
            if self.nlayers>1: self.bn1 = torch.nn.ModuleList([nn.BatchNorm1d(num_features=self.fc_dims[l]) for l in range(self.nlayers-1)])
            print(self.W_final_dec)
            self.W_final_dec = torch.nn.ModuleList(self.W_final_dec)
            
    def encode_andAdd_domains(self,domains, results):
        #forward domains embeddings. concatenate the domain to the results.  
        domains_emb = embedded_dropout(self.W_domains_emb, domains, dropout=self.emb_pdrop if self.training else 0)
        domains_emb = self.lockdrop(domains_emb, self.emb_pdrop)
        results_andDomains = torch.cat( (results,domains_emb), dim=-1) 
        return results_andDomains
        
    def forward_query_mem(self,query_img,query_places=None):  
        #projection of query that will be used with the memory modules - should be the same size as the memory 
        query_img_proj = F.relu(self.W_qImg(query_img)) if self.use_img_memory else None 
        query_places_proj = F.relu(self.W_qPlaces(query_places)) if self.use_places_memory else None 
        return query_img_proj,query_places_proj

    def generic_memory(self,query_proj, results,mem_a_weights,mem_c_weights, bn_mem, mem_hops=1, query_hop_weight=None, W_dec_mem=None, cluster=False, max_class_len=None, mem_cluster_weights=None, bn_cluster=None):
        # we add the parameter "max_class_len=None", which should be a tensor containing 3 elements in order like torch.tensor([max_sim_len, max_max_len, max_other_len]).
        # in one batch, the three elements in max_class_len is specific
        u = query_proj   # batch * query_dim
        for i in range(0,mem_hops):
            u = F.dropout(u, self.pdrop_mem)
     
            mem_a = F.relu(mem_a_weights(results))
            mem_a = F.dropout(mem_a, self.pdrop_mem)   # matrix A & C don't need to change

            mem_c = F.relu(mem_c_weights(results)) 
            mem_c = F.dropout(mem_c, self.pdrop_mem)
            # mem_a/c: batch * num * mem_[class]_dim_out
            # we need to use "gather" function to extract
            # print(f"cluster: {cluster}")
            if not cluster:
                # print("no cluster")
                P = F.softmax(torch.sum(mem_a * u.unsqueeze(1), 2), dim=1)
                mem_out = torch.sum(P.unsqueeze(2).expand_as(mem_c) * mem_c, 1)
                mem_out = mem_out + u
            else:
                # print(cluster)
                sim_ = max_class_len[0]   # a specific value
                max_ = max_class_len[1]
                other_ = max_class_len[2]
                # print(f"sim_:{sim_}, max_:{max_}, other_:{other_}")
                # after gather, mem_dim is (batch * select_num * mem_[class]_dim_out
                sim_P = F.softmax(torch.sum(mem_a[:, :sim_, :]*u.unsqueeze(1), 2), dim=1)   # batch_size * sim_
                max_P = F.softmax(torch.sum(mem_a[:, sim_: sim_+max_, :]*u.unsqueeze(1), 2), dim=1)
                other_P = F.softmax(torch.sum(mem_a[:, sim_+max_: sim_+max_+other_, :]*u.unsqueeze(1), 2), dim=1)
                # print(f"sim_P:{sim_P.shape}, max_P:{max_P.shape}, other_P:{other_P.shape}")
                P = torch.cat((sim_P, max_P, other_P), dim=1)
                # print(f"P:{P.shape}")
                # batch * sim_num
                sim_mem_out = torch.sum(sim_P.unsqueeze(2).expand_as(mem_c[:, :sim_, :]) * mem_c[:, :sim_, :], 1) + u   # batch_size * dim
                max_mem_out = torch.sum(max_P.unsqueeze(2).expand_as(mem_c[:, sim_: sim_+max_, :]) * mem_c[:, sim_: sim_+max_, :], 1) + u
                other_mem_out = torch.sum(other_P.unsqueeze(2).expand_as(mem_c[:, sim_+max_: sim_+max_+other_, :]) * mem_c[:, sim_+max_: sim_+max_+other_, :], 1) + u
                sim_mem_out = bn_cluster[0](sim_mem_out)
                max_mem_out = bn_cluster[1](max_mem_out)
                other_mem_out = bn_cluster[2](other_mem_out)
                # print(f"sim_mem_out:{sim_mem_out.shape}, max_mem_out:{max_mem_out.shape}, other_mem_out:{other_mem_out.shape}")
                # the dim of select_mem_out is (batch * mem_[class]_dim_out)
                mid_mem_out = torch.cat((sim_mem_out, max_mem_out, other_mem_out), dim=1) # batch_size * (3*mem_[class]_dim_out)
                # print(f"mid_mem_out:{mid_mem_out.shape}")
                # next is fc+relu+dropout
                mem_out = F.dropout(F.relu(mem_cluster_weights(mid_mem_out)), self.pdrop_mem)
                # print(f"mem_out:{mem_out.shape}")
                mem_out = mem_out + u
                # print(f"mem_out:{mem_out.shape}")

            u = mem_out 
        mem_out = bn_mem(mem_out)    
        
        if self.fusion == 'byDecision':
            mem_out = F.dropout(mem_out, p=self.pdrop)
            mem_dec_out = F.relu(W_dec_mem(mem_out))
            return mem_dec_out
        return mem_out,P 

    def consistency_clip(self,qimage_clip, qtext_clip):
        ### assume that we will read precomputed embeddings 
        qtext_clip = F.dropout(qtext_clip, p=self.inp_pdrop)    
        qimage_clip = F.dropout(qimage_clip, p=self.inp_pdrop)     
        
        encoded_img = qimage_clip / qimage_clip.norm(dim=-1, keepdim=True) 
        encoded_text = qtext_clip / qtext_clip.norm(dim=-1, keepdim=True)
        joint_features = encoded_img*encoded_text
        joint_features = self.bn_clip(joint_features)
        if self.fusion == 'byDecision':
            joint_features = F.dropout(joint_features, p=self.pdrop).float()
            consis_out = F.relu(self.W_consis_clip(joint_features))
            return consis_out
        return joint_features
        

    def forward(self,query_img, query_captions, query_places=None, entities=None, results_images=None, results_places=None, images_domains=None, results_captions=None, captions_domains=None, qimage_clip=None, qtext_clip=None, query_img_regions=None, cap_max_class_len=None, img_max_class_len=None):
        # print(f"forward cap cluster: {self.cap_cluster}")
        #dropout to the input
        query_img = F.dropout(query_img, self.inp_pdrop)
        query_captions = F.dropout(query_captions, self.inp_pdrop) 
        if self.use_places_memory: query_places = F.dropout(query_places, self.inp_pdrop) 
        #project queries to the memory dimension space.
        query_projections = self.forward_query_mem(query_img,query_places)
        
        #add domains embeddings to the search results.        
        if self.use_src:
            if self.use_img_memory: results_images = self.encode_andAdd_domains(images_domains, results_images)
            if self.use_cap_memory: results_captions = self.encode_andAdd_domains(captions_domains, results_captions)
            if self.use_places_memory: results_places = self.encode_andAdd_domains(images_domains, results_places)
            
        #apply dropout to the results 
        if self.use_img_memory: results_images = F.dropout(results_images, self.inp_pdrop)
        if self.use_cap_memory: results_captions = F.dropout(results_captions, self.inp_pdrop) 
        if self.use_ent_memory: entities = F.dropout(entities, self.inp_pdrop) 
        if self.use_places_memory: results_places = F.dropout(results_places, self.inp_pdrop) 

        #generic_memory(self,query_proj, results,mem_a_weights,mem_c_weights, bn_mem, mem_hops=1, query_hop_weight=None, W_dec_mem=None)
        #empty array 
        out_features = torch.empty((query_img.size(0),0)).cuda()            
        if self.use_img_memory:
            # print("img")
            img_mem_out,P_img = self.generic_memory(query_projections[0], results_images, self.W_im_a, self.W_im_c,self.bn_imgMem, self.img_mem_hops,self.Wqimg_hop if self.img_mem_hops>1 else None,self.W_dec_img_mem if self.fusion=='byDecision' else None, self.img_cluster, img_max_class_len, self.W_im_cluster if self.img_cluster else None, self.bn_img_cluster if self.img_cluster else None)
            out_features = img_mem_out
        if self.use_cap_memory:
            # print("cap")
            cap_mem_out,P_cap = self.generic_memory(query_captions, results_captions, self.W_sent_a, self.W_sent_c, self.bn_capMem, self.cap_mem_hops,self.Wqcap_hop if self.cap_mem_hops>1 else None,self.W_dec_cap_mem if self.fusion=='byDecision' else None, self.cap_cluster, cap_max_class_len, self.W_sent_cluster if self.cap_cluster else None, self.bn_cap_cluster if self.cap_cluster else None)
            out_features = torch.cat((out_features, cap_mem_out),dim=-1)
        if self.use_ent_memory:
            # print("ent")
            ent_mem_out,P_ent = self.generic_memory(query_captions, entities, self.W_ent_a, self.W_ent_c, self.bn_entMem, self.ent_mem_hops,self.Wqent_hop if self.ent_mem_hops>1 else None,self.W_dec_ent_mem if self.fusion=='byDecision' else None, self.ent_cluster, None, self.W_ent_cluster if self.ent_cluster else None, self.bn_ent_cluster if self.ent_cluster else None)
            out_features = torch.cat((out_features, ent_mem_out),dim=-1)
        if self.use_places_memory:
            # print("places")
            places_mem_out,P_places = self.generic_memory(query_projections[1], results_places, self.W_places_a, self.W_places_c, self.bn_placesMem, self.places_mem_hops,self.Wqplaces_hop if self.places_mem_hops>1 else None,self.W_dec_places_mem if self.fusion=='byDecision' else None, self.places_cluster, img_max_class_len, self.W_places_cluster if self.places_cluster else None, self.bn_places_cluster if self.places_cluster else None)
            out_features = torch.cat((out_features, places_mem_out),dim=-1)
            
        if self.consistency == 'clip':                 
            consistency_out = self.consistency_clip(query_img if self.use_clip_for_all else qimage_clip , query_captions if self.use_clip_for_all else qtext_clip)
            out_features = torch.cat((out_features, consistency_out),dim=-1)
  
        if self.fusion == 'byFeatures': 
            out_features = F.dropout(out_features,p=self.pdrop)
        for i in range(0,self.nlayers):
            out_features = self.W_final_dec[i](out_features)
            if i != (self.nlayers-1):
                out_features = F.relu(out_features)
                out_features = self.bn1[i](out_features)
                out_features = F.dropout(out_features,p=self.pdrop)
        return out_features, P_img, P_cap, P_ent, P_places    
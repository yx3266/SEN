'''
The file used for training the model in the repo.
The repo and the code are built on the top of CCN.
'''
from collate import collate_padding
import dataset
import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import model
import torch.nn.parallel
import torch.optim
import time 

parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
# datasets' and model's location
parser.add_argument('--queries_dataset_root', type=str, default='../../evidence/queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')   
parser.add_argument('--visual_news_root', type=str, default='../../../news_clippings/visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')
parser.add_argument('--news_clip_root', type=str, default='../../../news_clippings/news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')               
parser.add_argument('--dataset_items_file', type=str, default='../../evidence/queries_dataset/dataset_items_',
                    help='location to the dataset items file')
parser.add_argument('--domains_file', type=str, default='../../evidence/queries_dataset/domain_to_idx_dict.json',
                    help='location to the domains to idx file')
parser.add_argument('--exp_folder', type=str, default='./exp_optional_cluster_ss/',
                    help='path to the folder to log the output and save the models')
                    
# model details
parser.add_argument('--domains_dim', type=int, default=20,
                    help='dimension of domains embeddings')        
parser.add_argument('--mem_img_dim_out', type=int, default=1024,
                    help='dimension of image memory')
parser.add_argument('--mem_places_dim_out', type=int, default=1024,
                    help='dimension of resnet places memory')
parser.add_argument('--mem_sent_dim_out', type=int, default=768,
                    help='projection dimension of sentence emb in the memory network module')
parser.add_argument('--mem_ent_dim_out', type=int, default=768,
                    help='projection dimension of entities emb in the memory network module')
parser.add_argument('--consistency', type=str, default='clip',
                    help='which method to use for mismatch between queries, options: clip, san, embeddings. otherwise, will not be used')
parser.add_argument('--san_emb', type=int, default=1024,
                    help='projection dimension of san')
parser.add_argument('--consis_proj_dim', type=int, default=1024,
                    help='projection dimension of using the embeddings directly for consistency prediction')
parser.add_argument('--pdrop', type=float, default=0,
                    help='dropout probability')           # using for mem_out and joint_features when byDecision, and out_features before and after the fc
parser.add_argument('--pdrop_mem', type=float, default=0.1,
                    help='dropout probability')           # using for query and evidence when in mem
parser.add_argument('--inp_pdrop', type=float, default=0.05,
                    help='dropout probability for input features')   # using for input features
parser.add_argument('--emb_pdrop', type=float, default=0.25,
                    help='dropout probability for the embeddings of domains')   # using for the embeddings of domains
parser.add_argument('--img_mem_hops', type=int, default=1,
                    help='number of hops for the img memory') 
parser.add_argument('--cap_mem_hops', type=int, default=1,
                    help='number of hops for the cap memory') 
parser.add_argument('--ent_mem_hops', type=int, default=1,
                    help='number of hops for the ent memory')
parser.add_argument('--places_mem_hops', type=int, default=1,
                    help='number of hops for the places memory')
                    
                    
parser.add_argument('--fusion', type=str, default='byFeatures',
                    help='how to fuse the different component, options: byFeatures and byDecision')  
parser.add_argument('--nlayers', type=int, default='2',
                    help='number of fc layers of the final classifier') 
parser.add_argument("--fc_dims",nargs="*",type=int,default=[1024],
                    help='the dimensions of the fully connected classifier layers') 
parser.add_argument('--img_rep', type=str, default='pool',
                    help='how to represent images in the memory, options: pool or regions')                      
                    
parser.add_argument('--use_src', action='store_true', help='whether to use domain embeddings in the network')
parser.add_argument('--use_img_memory', action='store_true', help='whether to use img memory')
parser.add_argument('--use_ent_memory', action='store_true', help='whether to use ent memory')
parser.add_argument('--use_cap_memory', action='store_true', help='whether to use cap memory')
parser.add_argument('--use_places_memory', action='store_true', help='whether to use resnet places memory')

parser.add_argument('--shared_cap_proj', action='store_true', help='whether to use the same project for the caption in the caption and entities memories')
parser.add_argument('--shared_cap_ent_mem', action='store_true', help='whether to use the share the caption and entities memories')
parser.add_argument('--use_clip_for_all', action='store_true', help='whether to use clip for all embeddings')
parser.add_argument('--labels_overlap', action='store_true', help='whether to load labels overlap between images')
parser.add_argument('--filter_dup', action='store_true', help='whether to filter out evidence that exactly match the query')

# support-refutation score
parser.add_argument('--ner_ent', default='',
                    help='which method to use for measuring the correlation between the entities evidence and query caption. Option: none, binary, srscore')
parser.add_argument('--ner_cap', default='',
                    help='which method to use for measuring the correlation between the captions evidence and query caption. Option: none, binary, srscore')
# stance extraction network
parser.add_argument('--img_cluster', action='store_true', help='whether to use sen for visual evidence in general feature')
parser.add_argument('--places_cluster', action='store_true', help='whether to use sen for visual evidence in places feature')
parser.add_argument('--cap_cluster', action='store_true', help='whether to use sen for textual evidence of sentence type')
parser.add_argument('--ent_cluster', action='store_true', help='whether to use sen for textual evidence of entity type')

# training details
parser.add_argument('--batch_size', type=int, default=64,
                    help='bs')
parser.add_argument('--eval_batch_size', type=int, default=64,
                    help='bs for validation and test sets')                    
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of data loaders workers') 
parser.add_argument('--epochs', type=int, default=60,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--log_interval', type=int, default=200,
                    help='how many batches')
parser.add_argument('--resume', type=str, default = '', help='path to model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='which optimizer to use')
parser.add_argument('--lr', type=float, default=0.000009,
                    help='start or base learning rate') 
parser.add_argument('--lr_max', type=float, default=0.00006,
                    help='learning rate')                     
parser.add_argument('--lr_sched', type=str, default='cycle',
                        help='use cyclic scheduler')
parser.add_argument('--sgd_momentum', type=float, default=0.9,
                    help='learning rate')                      
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', default=1.2e-6, type=float,
                        help='weight decay pow (default: -5)')

parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
                    
args = parser.parse_args()
# maintain consistency between the two visual SENs
args.places_cluster = args.img_cluster
# default settings, can be changed or deleted
args.use_src = True
args.use_img_memory = True
args.use_ent_memory = True
args.use_cap_memory = True
args.use_places_memory = True
args.filter_dup = True
args.labels_overlap = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### load input files ####
data_items_train = json.load(open(args.dataset_items_file+"train.json"))
data_items_val = json.load(open(args.dataset_items_file+"val.json"))
data_items_test = json.load(open(args.dataset_items_file+"test.json"))
domain_to_idx_dict = json.load(open(args.domains_file))

#### load Datasets and DataLoader ####
sent_emb_dim = 512 if args.use_clip_for_all else 768 #sentence bert dimension 

train_dataset = dataset.NewsContextDatasetEmbs(data_items_train, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'train', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' and not args.use_clip_for_all else False, \
load_clip_for_all = args.use_clip_for_all,labels_overlap=args.labels_overlap,\
ner_ent=args.ner_ent, ner_cap=args.ner_cap,filter_duplicates=args.filter_dup, \
img_cluster=args.img_cluster, places_cluster=args.places_cluster, cap_cluster=args.cap_cluster, ent_cluster=args.ent_cluster)

val_dataset = dataset.NewsContextDatasetEmbs(data_items_val, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'val', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' and not args.use_clip_for_all else False, \
load_clip_for_all = args.use_clip_for_all,labels_overlap=args.labels_overlap,\
ner_ent=args.ner_ent, ner_cap=args.ner_cap,filter_duplicates=args.filter_dup, \
img_cluster=args.img_cluster, places_cluster=args.places_cluster, cap_cluster=args.cap_cluster, ent_cluster=args.ent_cluster)

test_dataset = dataset.NewsContextDatasetEmbs(data_items_test, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'test', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' and not args.use_clip_for_all else False, \
load_clip_for_all = args.use_clip_for_all,labels_overlap=args.labels_overlap,\
ner_ent=args.ner_ent, ner_cap=args.ner_cap,filter_duplicates=args.filter_dup, \
img_cluster=args.img_cluster, places_cluster=args.places_cluster, cap_cluster=args.cap_cluster, ent_cluster=args.ent_cluster)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_padding, num_workers=args.num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn = collate_padding, num_workers=args.num_workers,  pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn = collate_padding, num_workers=args.num_workers,  pin_memory=True)

#### settings of the model ####
img_features_dim = 512 if args.use_clip_for_all else 2048 #resnet dimension 
places_mem_dim = 2048 #resnet places dim 

model_settings = {'use_img_memory':args.use_img_memory, 'use_cap_memory':args.use_cap_memory, 'use_ent_memory': args.use_ent_memory,\
'use_src': args.use_src, 'use_places_memory':args.use_places_memory,'domains_num': len(domain_to_idx_dict), 'domains_dim': args.domains_dim, \
'img_dim_in': img_features_dim, 'img_dim_out': args.mem_img_dim_out, 'places_dim_in': places_mem_dim, 'places_dim_out': args.mem_places_dim_out, \
'ent_dim_in': sent_emb_dim,'ent_dim_out': args.mem_ent_dim_out, 'sent_emb_dim_in': sent_emb_dim,'sent_emb_dim_out': args.mem_sent_dim_out, \
'consistency': args.consistency, 'san_emb': args.san_emb, 'consis_proj_dim': args.consis_proj_dim, \
'fusion': args.fusion, 'pdrop': args.pdrop, 'inp_pdrop': args.inp_pdrop, 'pdrop_mem': args.pdrop_mem, 'emb_pdrop':args.emb_pdrop, \
'nlayers': args.nlayers, 'fc_dims': args.fc_dims, \
'img_mem_hops': args.img_mem_hops, 'cap_mem_hops': args.cap_mem_hops, 'ent_mem_hops': args.ent_mem_hops, 'places_mem_hops':args.places_mem_hops,\
'use_clip_for_all': args.use_clip_for_all,\
'ner_cap': True if args.ner_cap != '' else False, 'ner_ent': True if args.ner_ent != '' else False,'labels_overlap':args.labels_overlap,\
'img_cluster': args.img_cluster, 'places_cluster': args.places_cluster, 'cap_cluster': args.cap_cluster, 'ent_cluster': args.ent_cluster}

device = 0
model = model.ContextSEN(model_settings)
model.cuda(device)
#create exp folder if it has not been created 
if not os.path.isdir(args.exp_folder):
    os.makedirs(args.exp_folder)

print(model)

#define optimizer 
params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay, momentum=args.sgd_momentum)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    
#resume training
stored_loss = 100000000  
stored_acc = 0 
if args.resume:
    if os.path.isfile(args.resume):
        log_file_val_loss = open(os.path.join(args.exp_folder,'log_file_val.txt'),'a') 
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if 'best_val_loss' in checkpoint: stored_loss = checkpoint['best_val_loss']
        if 'best_val_acc' in checkpoint: stored_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    log_file_val_loss = open(os.path.join(args.exp_folder,'log_file_val.txt'),'w') 

if args.lr_sched == 'cycle':
    lr_tot_steps = args.epochs*len(train_dataloader)
    steps_up = lr_tot_steps/3 
    steps_down = 2*lr_tot_steps/3 
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.lr_max,step_size_up=steps_up, step_size_down=steps_down, cycle_momentum=False)
else:
    scheduler = None 
#define loss function
criterion = nn.BCEWithLogitsLoss().cuda(device)

#the saved features are not average-pooled
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

def evaluate(loader):
    total_loss = 0
    correct = 0
    total_num = 0
    model.eval()
    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda(device)
        entities_mem = batch[1].cuda(device) if args.use_ent_memory else None
        captions = batch[2].cuda(device) if args.use_cap_memory else None
        captions_domains = batch[3].cuda(device) if (args.use_src and args.use_cap_memory) else None
        img_mem = batch[4].cuda(device) if args.use_img_memory else None
        img_mem_domains = batch[5].cuda(device) if ( (args.use_img_memory or args.use_places_memory) and args.use_src) else None
        places_mem = batch[6].cuda(device) if args.use_places_memory else None
        qCap = batch[7].cuda(device)
        qImg = batch[8].cuda(device)
        qPlace = batch[9].cuda(device) if args.use_places_memory else None
        cap_max_class_len = batch[10].cuda(device) if args.cap_cluster and batch[10] != None else None
        img_max_class_len = batch[11].cuda(device) if args.img_cluster and batch[11] != None else None
        
        if args.consistency == 'clip' and not args.use_clip_for_all:     
            qCap_clip = batch[12].cuda(device)
            qImg_clip = batch[13].cuda(device)
        else:       
            qCap_clip = None      
            qImg_clip = None
        with torch.no_grad():
            img_dim = (model_settings['img_dim_in']+1) if args.labels_overlap else model_settings['img_dim_in']
            if not args.use_clip_for_all: qImg_avg = adaptive_pool(qImg).view(labels.size(0),model_settings['img_dim_in']) 
            if img_mem is not None: 
                num_img_mem = img_mem.size(1) 
                if args.img_rep == 'pool' and not args.use_clip_for_all:
                    img_mem = adaptive_pool(img_mem).view(labels.size(0),num_img_mem,img_dim)
                elif args.img_rep == 'regions': 
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7*7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7*7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem*7*7, img_dim)
                    if img_mem_domains is not None: img_mem_domains = img_mem_domains.unsqueeze(dim=2).repeat(1,1,49).view(labels.size(0),num_img_mem*49)


            #forward 
            output = model(qImg_avg if not args.use_clip_for_all else qImg, qCap, query_places=qPlace, entities=entities_mem, \
                       results_images=img_mem, results_places = places_mem, images_domains=img_mem_domains, \
                       results_captions=captions, captions_domains=captions_domains, \
                       query_img_regions = qImg if args.consistency == 'san' else None, qtext_clip = qCap_clip, qimage_clip=qImg_clip, \
                       cap_max_class_len = cap_max_class_len if args.cap_cluster else None, img_max_class_len=img_max_class_len if args.img_cluster else None)
            #compute loss     
            loss = criterion(output[0], torch.unsqueeze(labels.float(), 1))
            total_loss += loss.item()
            #compute correct predictions              
            pred = torch.sigmoid(output[0]) >= 0.5
            truth = torch.unsqueeze(labels,1)>=0.5
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0) 
    avg_loss = total_loss/len(loader)    
    acc = (correct/total_num)*100
    return avg_loss, acc


def evaluate_save(loader):
    total_loss = 0
    correct = 0
    correct_falsified = 0
    correct_match = 0
    total_num = 0
    total_num_falsified = 0
    total_num_match = 0
    model.eval()
    result = None
    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda(device)
        entities_mem = batch[1].cuda(device) if args.use_ent_memory else None
        captions = batch[2].cuda(device) if args.use_cap_memory else None
        captions_domains = batch[3].cuda(device) if (args.use_src and args.use_cap_memory) else None
        img_mem = batch[4].cuda(device) if args.use_img_memory else None
        img_mem_domains = batch[5].cuda(device) if (
                    (args.use_img_memory or args.use_places_memory) and args.use_src) else None
        places_mem = batch[6].cuda(device) if args.use_places_memory else None
        qCap = batch[7].cuda(device)
        qImg = batch[8].cuda(device)
        qPlace = batch[9].cuda(device) if args.use_places_memory else None
        cap_max_class_len = batch[10].cuda(device) if args.cap_cluster and batch[10] != None else None
        img_max_class_len = batch[11].cuda(device) if args.img_cluster and batch[11] != None else None

        if args.consistency == 'clip' and not args.use_clip_for_all:
            qCap_clip = batch[12].cuda(device)
            qImg_clip = batch[13].cuda(device)
        else:
            qCap_clip = None
            qImg_clip = None
        with torch.no_grad():
            img_dim = (model_settings['img_dim_in'] + 1) if args.labels_overlap else model_settings['img_dim_in']
            if not args.use_clip_for_all: qImg_avg = adaptive_pool(qImg).view(labels.size(0),
                                                                              model_settings['img_dim_in'])
            if img_mem is not None:
                num_img_mem = img_mem.size(1)
                if args.img_rep == 'pool' and not args.use_clip_for_all:
                    img_mem = adaptive_pool(img_mem).view(labels.size(0), num_img_mem, img_dim)
                elif args.img_rep == 'regions':
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7 * 7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7 * 7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem * 7 * 7, img_dim)
                    if img_mem_domains is not None: img_mem_domains = img_mem_domains.unsqueeze(dim=2).repeat(1, 1,
                                                                                                              49).view(
                        labels.size(0), num_img_mem * 49)

            # forward
            output = model(qImg_avg if not args.use_clip_for_all else qImg, qCap, query_places=qPlace,
                           entities=entities_mem, \
                           results_images=img_mem, results_places=places_mem, images_domains=img_mem_domains, \
                           results_captions=captions, captions_domains=captions_domains, \
                           query_img_regions=qImg if args.consistency == 'san' else None, qtext_clip=qCap_clip,
                           qimage_clip=qImg_clip, \
                           cap_max_class_len=cap_max_class_len if args.cap_cluster else None,
                           img_max_class_len=img_max_class_len if args.img_cluster else None)
            # compute loss
            loss = criterion(output[0], torch.unsqueeze(labels.float(), 1))
            total_loss += loss.item()
            # compute correct predictions
            pred = torch.sigmoid(output[0]) >= 0.5
            truth = torch.unsqueeze(labels, 1) >= 0.5
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0)

            if result == None:
                result = pred.eq(truth)
            else:
                result = torch.cat((result, pred.eq(truth)))

            # print(labels)
            index_ones = ((labels == 1).nonzero(as_tuple=True)[0])
            correct_falsified += pred[index_ones, :].eq(truth[index_ones, :]).sum().item()
            total_num_falsified += index_ones.size(0)

            index_zeros = ((labels == 0).nonzero(as_tuple=True)[0])
            correct_match += pred[index_zeros, :].eq(truth[index_zeros, :]).sum().item()
            total_num_match += index_zeros.size(0)

    avg_loss = total_loss / len(loader)
    acc = (correct / total_num) * 100
    acc_falsified = (correct_falsified / total_num_falsified) * 100
    acc_match = (correct_match / total_num_match) * 100
    print(f"correct: {correct}\ttotal_num: {total_num}")
    print(f"correct_falsified: {correct_falsified}\ttotal_num_falsified: {total_num_falsified}")
    print(f"correct_match: {correct_match}\ttotal_num_match: {total_num_match}")
    return avg_loss, acc, acc_falsified, acc_match, result, (correct, correct_falsified, correct_match), (total_num, total_num_falsified, total_num_match)

def train():
    total_loss = 0
    start_time = time.time()
    start_time2 = time.time()
    model.train()
    for (idx, batch) in enumerate(train_dataloader):
        #prepare batches 
        labels = batch[0].cuda(device)
        entities_mem = batch[1].cuda(device) if args.use_ent_memory else None
        captions = batch[2].cuda(device) if args.use_cap_memory else None
        captions_domains = batch[3].cuda(device) if (args.use_src and args.use_cap_memory) else None
        img_mem = batch[4].cuda(device) if args.use_img_memory else None
        img_mem_domains = batch[5].cuda(device) if ( (args.use_img_memory or args.use_places_memory) and args.use_src) else None
        places_mem = batch[6].cuda(device) if args.use_places_memory else None
        qCap = batch[7].cuda(device)
        qImg = batch[8].cuda(device)
        qPlace = batch[9].cuda(device) if args.use_places_memory else None
        cap_max_class_len = batch[10].cuda(device) if args.cap_cluster and batch[10] != None else None
        img_max_class_len = batch[11].cuda(device) if args.img_cluster and batch[11] != None else None


        if args.consistency == 'clip' and not args.use_clip_for_all:    
            qCap_clip = batch[12].cuda(device)
            qImg_clip = batch[13].cuda(device)
        else:       
            qCap_clip = None      
            qImg_clip = None             
        #the saved embeddings are the full feature map
        with torch.no_grad():
            img_dim = (model_settings['img_dim_in']+1) if args.labels_overlap else model_settings['img_dim_in']
            if not args.use_clip_for_all: qImg_avg = adaptive_pool(qImg).view(labels.size(0),model_settings['img_dim_in']) 
            if img_mem is not None: 
                num_img_mem = img_mem.size(1) 
                if args.img_rep == 'pool' and not args.use_clip_for_all:
                    img_mem = adaptive_pool(img_mem).view(labels.size(0),num_img_mem,img_dim)
                elif args.img_rep == 'regions': 
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7*7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7*7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem*7*7, img_dim)
                    if img_mem_domains is not None: img_mem_domains = img_mem_domains.unsqueeze(dim=2).repeat(1,1,49).view(labels.size(0),num_img_mem*49)

        #forward 
        output = model(qImg_avg if not args.use_clip_for_all else qImg, qCap, query_places=qPlace, entities=entities_mem,\
                       results_images=img_mem, results_places = places_mem, images_domains=img_mem_domains, \
                       results_captions=captions, captions_domains=captions_domains, \
                       query_img_regions = qImg if args.consistency == 'san' else None, qtext_clip = qCap_clip, qimage_clip=qImg_clip, \
                       cap_max_class_len = cap_max_class_len if args.cap_cluster else None, img_max_class_len=img_max_class_len if args.img_cluster else None)
        #compute loss     
        loss = criterion(output[0], torch.unsqueeze(labels.float(), 1))
        total_loss += loss.item()
    
        #backward and optimizer step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            #print(scheduler.get_last_lr())
        #log    
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.9f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                    epoch, idx, len(train_dataloader) , scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
     
if args.mode == 'train':
    try:
    #you can exit from training by: ctrl+c
        for epoch in range(args.start_epoch, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss, val_acc = evaluate(val_dataloader)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | '
                    'val acc {:8.2f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss , val_acc))
            print('-' * 89)
            log_file_val_loss.write(str(val_loss) + ', '+ str(val_acc)+ '\n')
            log_file_val_loss.flush()
            if val_loss < stored_loss:
                print('New best model loss')
                stored_loss = val_loss
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'best_val_loss': stored_loss,
                            'best_val_acc': stored_acc},
                            os.path.join(args.exp_folder, 'best_model_loss.pth.tar'))
            if val_acc > stored_acc:
                print('New best model acc')
                stored_acc = val_acc
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'best_val_loss': stored_loss,
                            'best_val_acc': stored_acc},
                            os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    
# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_loss.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_loss.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_loss.pth.tar')))
    # Run on test data.
    test_loss, test_acc, test_acc_false, test_acc_match, test_result, test_correct_fm, test_total_fm = evaluate_save(test_dataloader)
    print('=' * 89)
    print('| End of training | best loss model | test loss {:5.2f} | test acc {:8.3f}'.format(
    test_loss, test_acc))
    print('=' * 89)
    loss_save = args.exp_folder + "loss_result.npz"
    np.savez(loss_save, result=test_result.to('cpu').numpy(), correct_fm=test_correct_fm, total_fm=test_total_fm)

if os.path.isfile(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint: '{}')".format(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')))
    # Run on test data.
    test_loss, test_acc, test_acc_false, test_acc_match, test_result, test_correct_fm, test_total_fm = evaluate_save(test_dataloader)
    print('=' * 89)
    print('| End of training | best acc model | test loss {:5.2f} | test acc {:8.3f}'.format(
    test_loss, test_acc))
    print('=' * 89)
    acc_save = args.exp_folder + "acc_result.npz"
    np.savez(acc_save, result=test_result.to('cpu').numpy(), correct_fm=test_correct_fm, total_fm=test_total_fm)
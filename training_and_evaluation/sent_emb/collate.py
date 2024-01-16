import torch 

def collate_padding(batch):
    samples = [item[0] for item in batch]
    max_entities_len = max([item[1] for item in batch])
    max_captions_len = max([item[2] for item in batch])
    max_images_len = max([item[3] for item in batch])
    entities_batch = []
    mem_cap_batch = []
    mem_cap_domains_batch = []
    mem_img_batch = []
    mem_img_domains_batch = []
    mem_places_batch = []
    qCap_batch = []
    qImg_batch = []
    qPlaces_batch = []

    qCap_clip_batch = []
    qImg_clip_batch = []
    labels = []

    cap_cluster = samples[0]['cap_cluster']
    cap_max_sim_len = max([item[4][0] for item in batch])
    cap_max_max_len = max([item[4][1] for item in batch])
    cap_max_other_len = max([item[4][2] for item in batch])
    # print(f"all_max_sim_len:{cap_max_sim_len}")
    # print(f"all_max_max_len:{cap_max_max_len}")
    # print(f"all_max_other_len:{cap_max_other_len}")
    # print(f"sim:{cap_max_sim_len}, max:{cap_max_max_len}, other:{cap_max_other_len}")

    img_cluster = samples[0]['img_cluster']
    img_max_sim_len = max([item[5][0] for item in batch])
    img_max_max_len = max([item[5][1] for item in batch])
    img_max_other_len = max([item[5][2] for item in batch])
    # print(f"all_img_max_sim_len:{img_max_sim_len}")
    # print(f"all_img_max_max_len:{img_max_max_len}")
    # print(f"all_img_max_other_len:{img_max_other_len}")


    for j in range(0,len(samples)):
        # print(j)
        sample = samples[j]    
        labels.append(sample['label'])
        #pad entities
        entities = sample['entities']
        padding_size = (max_entities_len-sample['entities'].size(0), sample['entities'].size(1))
        padded_mem_ent = torch.cat((sample['entities'], torch.zeros(padding_size)),dim=0)   # using torch.zeros to padding
        entities_batch.append(padded_mem_ent)
        if not cap_cluster:
            #pad captions
            padding_size = (max_captions_len-sample['caption'].size(0), sample['caption'].size(1))
            padded_mem_cap = torch.cat((sample['caption'], torch.zeros(padding_size)),dim=0)
            mem_cap_batch.append(padded_mem_cap)
            #pad domains of captions
            padded_cap_domains = torch.cat( (sample['caption_domains'], torch.zeros((max_captions_len-sample['caption'].size(0)))) )
            mem_cap_domains_batch.append(padded_cap_domains)
            if sample['caption'].size(0) != sample['caption_domains'].size(0):
                print('domains mismatch - captions')
        else:
            # pad captions
            # print(f"{j}: {sample['caption_cluster_num']}")
            padding_sim_size = (cap_max_sim_len-sample['caption_cluster_num'][0], sample['caption'].size(1))
            padded_mem_cap_sim = torch.cat((sample['caption'][:sample['caption_cluster_num'][0], :], torch.zeros(padding_sim_size)), dim=0)
            padding_max_size = (cap_max_max_len - sample['caption_cluster_num'][1], sample['caption'].size(1))
            padded_mem_cap_max = torch.cat((sample['caption'][
                                            sample['caption_cluster_num'][0]: sample['caption_cluster_num'][0] +
                                                                              sample['caption_cluster_num'][1], :],
                                            torch.zeros(padding_max_size)), dim=0)
            padding_other_size = (cap_max_other_len - sample['caption_cluster_num'][2], sample['caption'].size(1))
            padded_mem_cap_other = torch.cat((sample['caption'][
                                              sample['caption_cluster_num'][0] + sample['caption_cluster_num'][1]:
                                              sample['caption_cluster_num'][0] + sample['caption_cluster_num'][1] +
                                              sample['caption_cluster_num'][2], :], torch.zeros(padding_other_size)),
                                             dim=0)
            padded_mem_cap = torch.cat((padded_mem_cap_sim, padded_mem_cap_max, padded_mem_cap_other), dim=0)
            # print(f"padded_mem_cap:{padded_mem_cap.shape}")
            mem_cap_batch.append(padded_mem_cap)
            # pad domains of captions
            padded_cap_domains_sim = torch.cat((sample['caption_domains'][:sample['caption_cluster_num'][0]],
                                                torch.zeros(cap_max_sim_len - sample['caption_cluster_num'][0])))
            padded_cap_domains_max = torch.cat((sample['caption_domains'][
                                                sample['caption_cluster_num'][0]: sample['caption_cluster_num'][0] +
                                                                                  sample['caption_cluster_num'][1]],
                                                torch.zeros(cap_max_max_len - sample['caption_cluster_num'][1])))

            padded_cap_domains_other = torch.cat((sample['caption_domains'][
                                                  sample['caption_cluster_num'][0] + sample['caption_cluster_num'][1]:
                                                  sample['caption_cluster_num'][0] + sample['caption_cluster_num'][1] +
                                                  sample['caption_cluster_num'][2]],
                                                  torch.zeros(cap_max_other_len - sample['caption_cluster_num'][2])))
            padded_cap_domains = torch.cat((padded_cap_domains_sim, padded_cap_domains_max, padded_cap_domains_other))
            # print(f"padded_cap_domains:{padded_cap_domains.shape}")
            mem_cap_domains_batch.append(padded_cap_domains)
            if sample['caption'].size(0) != sample['caption_domains'].size(0):
                print('domains mismatch - captions')


        if not img_cluster:
            #padded images
            if len(sample['imgs'].size()) > 2:
                padding_size = (max_images_len-sample['imgs'].size(0),sample['imgs'].size(1),sample['imgs'].size(2),sample['imgs'].size(3))
            else:
                padding_size = (max_images_len-sample['imgs'].size(0),sample['imgs'].size(1))
            if sample['imgs'].size(0) != sample['imgs_domains'].size(0):
                print('domains mismatch')
            padded_mem_img = torch.cat((sample['imgs'], torch.zeros(padding_size)),dim=0)
            mem_img_batch.append(padded_mem_img)
            #pad domains of images
            padded_img_domains = torch.cat( (sample['imgs_domains'], torch.zeros((max_images_len-sample['imgs'].size(0)))))
            mem_img_domains_batch.append(padded_img_domains)
            # places memory
            padding_size = (max_images_len - sample['imgs'].size(0), sample['places_mem'].size(1))
            padded_mem_places = torch.cat((sample['places_mem'], torch.zeros(padding_size)), dim=0)
            mem_places_batch.append(padded_mem_places)
        else:
            # pad images
            if len(sample['imgs'].size()) > 2:
                padding_sim_size = (img_max_sim_len-sample['img_cluster_num'][0], sample['imgs'].size(1), sample['imgs'].size(2), sample['imgs'].size(3))
                padding_max_size = (img_max_max_len-sample['img_cluster_num'][1], sample['imgs'].size(1), sample['imgs'].size(2), sample['imgs'].size(3))
                padding_other_size = (img_max_other_len-sample['img_cluster_num'][2], sample['imgs'].size(1), sample['imgs'].size(2), sample['imgs'].size(3))
            else:
                padding_sim_size = (img_max_sim_len-sample['img_cluster_num'][0], sample['imgs'].size(1))
                padding_max_size = (img_max_max_len-sample['img_cluster_num'][1], sample['imgs'].size(1))
                padding_other_size = (img_max_other_len-sample['img_cluster_num'][2], sample['imgs'].size(1))
            #print(f"padding_sim_size: {padding_sim_size}")
            #print(f"padding_max_size: {padding_max_size}")
            #print(f"padding_other_size: {padding_other_size}")
            padded_mem_img_sim = torch.cat((sample['imgs'][:sample['img_cluster_num'][0], ], torch.zeros(padding_sim_size)), dim=0)
            padded_mem_img_max = torch.cat((sample['imgs'][sample['img_cluster_num'][0]:sample['img_cluster_num'][0] +
                                                                                        sample['img_cluster_num'][1], ],
                                            torch.zeros(padding_max_size)), dim=0)
            padded_mem_img_other = torch.cat((sample['imgs'][
                                              sample['img_cluster_num'][0] + sample['img_cluster_num'][1]:
                                              sample['img_cluster_num'][0] + sample['img_cluster_num'][1] +
                                              sample['img_cluster_num'][2]], torch.zeros(padding_other_size)), dim=0)
            padded_mem_img = torch.cat((padded_mem_img_sim, padded_mem_img_max, padded_mem_img_other), dim=0)
            mem_img_batch.append(padded_mem_img)
            # pad image domains
            padded_img_domains_sim = torch.cat((sample['imgs_domains'][:sample['img_cluster_num'][0]], torch.zeros(img_max_sim_len-sample['img_cluster_num'][0])))
            padded_img_domains_max = torch.cat((sample['imgs_domains'][sample['img_cluster_num'][0]:sample['img_cluster_num'][0]+sample['img_cluster_num'][1]], torch.zeros(img_max_max_len-sample['img_cluster_num'][1])))
            padded_img_domains_other = torch.cat((sample['imgs_domains'][sample['img_cluster_num'][0]+sample['img_cluster_num'][1]:sample['img_cluster_num'][0]+sample['img_cluster_num'][1]+sample['img_cluster_num'][2]], torch.zeros(img_max_other_len-sample['img_cluster_num'][2])))
            padded_img_domains = torch.cat((padded_img_domains_sim, padded_img_domains_max, padded_img_domains_other))
            mem_img_domains_batch.append(padded_img_domains)
            if sample['imgs'].size(0) != sample['imgs_domains'].size(0):
                print('domains mismatch')
            # pad places, num * dim
            padded_mem_places_sim = torch.cat((sample['places_mem'][:sample['img_cluster_num'][0], ], torch.zeros((img_max_sim_len-sample['img_cluster_num'][0], sample['places_mem'].size(1)))), dim=0)
            padded_mem_places_max = torch.cat((sample['places_mem'][sample['img_cluster_num'][0]:sample['img_cluster_num'][0]+sample['img_cluster_num'][1], ], torch.zeros((img_max_max_len-sample['img_cluster_num'][1], sample['places_mem'].size(1)))), dim=0)
            padded_mem_places_other = torch.cat((sample['places_mem'][sample['img_cluster_num'][0]+sample['img_cluster_num'][1]:sample['img_cluster_num'][0]+sample['img_cluster_num'][1]+sample['img_cluster_num'][2], ], torch.zeros((img_max_other_len-sample['img_cluster_num'][2], sample['places_mem'].size(1)))), dim=0)
            padded_mem_places = torch.cat((padded_mem_places_sim, padded_mem_places_max, padded_mem_places_other), dim=0)
            mem_places_batch.append(padded_mem_places)

        #Query 
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        qPlaces_batch.append(sample['qPlaces'])
        
        if 'qImg_clip' in sample.keys(): qImg_clip_batch.append(sample['qImg_clip'])   
        if 'qCap_clip' in sample.keys(): qCap_clip_batch.append(sample['qCap_clip'])          
        
    #stack 
    entities_batch = torch.stack(entities_batch, dim=0)
    mem_cap_batch = torch.stack(mem_cap_batch, dim=0)
    mem_cap_domains_batch = torch.stack(mem_cap_domains_batch, dim=0).long()
    mem_img_batch = torch.stack(mem_img_batch, dim=0)
    mem_img_domains_batch = torch.stack(mem_img_domains_batch, dim=0).long()
    mem_places_batch = torch.stack(mem_places_batch, dim=0)

    qImg_batch = torch.cat(qImg_batch, dim=0)
    qCap_batch = torch.cat(qCap_batch, dim=0)
    qPlaces_batch = torch.cat(qPlaces_batch, dim=0)

    labels = torch.stack(labels, dim=0) 
    if qImg_clip_batch and qCap_clip_batch:
        qImg_clip_batch = torch.cat(qImg_clip_batch, dim=0)
        qCap_clip_batch = torch.cat(qCap_clip_batch, dim=0)
        return labels, entities_batch, mem_cap_batch, mem_cap_domains_batch, mem_img_batch, mem_img_domains_batch, mem_places_batch, qCap_batch, qImg_batch, qPlaces_batch, torch.tensor([cap_max_sim_len, cap_max_max_len, cap_max_other_len]) if cap_cluster else None, torch.tensor([img_max_sim_len, img_max_max_len, img_max_other_len]) if img_cluster else None, qCap_clip_batch, qImg_clip_batch
    return labels, entities_batch, mem_cap_batch, mem_cap_domains_batch, mem_img_batch, mem_img_domains_batch, mem_places_batch, qCap_batch, qImg_batch, qPlaces_batch, torch.tensor([cap_max_sim_len, cap_max_max_len, cap_max_other_len]) if cap_cluster else None, torch.tensor([img_max_sim_len, img_max_max_len, img_max_other_len]) if img_cluster else None
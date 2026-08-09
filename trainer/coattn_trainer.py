from glob import glob
import numpy as np
import torch

import os
from sksurv.metrics import concordance_index_censored
import torch.nn as nn

import torch
from typing import Tuple
import math
from captum.attr import IntegratedGradients

def replace_random_patches(feat_a,feat_b,ratio = 0.2):
    assert feat_a.dim() == 2 and feat_b.dim() == 2, "input must be 2-D tensor"
    assert feat_a.size(1) == feat_b.size(1), "two tensors' dimension (768) must be same"

    N1 = feat_a.size(0)
    N2 = feat_b.size(0)
    n_replace  = max(1, int(N1 * ratio))             
    if N2<n_replace:
        n_replace=N2
    
    replace_indices = np.random.choice(N1, size=n_replace, replace=False)
    donor_indices = np.random.choice(N2, size=n_replace, replace=False)
    features_aug = feat_a.clone()
    features_aug[replace_indices] = feat_b[donor_indices]
    return features_aug

def mix_omics(cur_omics, near_omics, ratio=0.2):
    
    aug_omics = []
    for cur, near in zip(cur_omics, near_omics):

        dim = cur.shape[-1]
        k = max(1, int(dim * ratio))             
        idx = torch.randperm(dim, device=cur.device)[:k]

        cur_aug = cur.clone()                    
        cur_aug[..., idx] = near[..., idx]       
        aug_omics.append(cur_aug)
    return aug_omics


def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, args=None, wsi_omic_fisher_list=None, fisher=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    train_recon_loss = 0.
    train_encode_wsi_loss = 0.
    train_kl_wsi = 0.
    flow_fit_loss = 0.

    if epoch < args.warm_epoch: 
        stage = 'warmup' 
    else: 
        stage = 'jointly' 

    
    

    
    if epoch > 0:
        aug_rate = 0.8
        aug_slide_id = []
        if fisher["wsi"] < fisher["omic"]:
            k = int(np.ceil(aug_rate * len(wsi_omic_fisher_list['wsi_fisher']))) 
            idx_sorted = np.argsort(wsi_omic_fisher_list['wsi_fisher']) 
            top_k_idx = idx_sorted[:k] 
            #  slide_id
            aug_slide_id = [wsi_omic_fisher_list['slide_id'][i] for i in top_k_idx]
        elif fisher["wsi"] >= fisher["omic"]:
            k = int(np.ceil(aug_rate * len(wsi_omic_fisher_list['omic_fisher']))) 
            idx_sorted = np.argsort(wsi_omic_fisher_list['omic_fisher']) 
            top_k_idx = idx_sorted[:k] 
            #  slide_id
            aug_slide_id = [wsi_omic_fisher_list['slide_id'][i] for i in top_k_idx]
        aug_num = len(aug_slide_id)*1
    else:
        aug_slide_id = []
        aug_num=0
    #####
    
    print('\n')
    num_samples = len(loader)
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_wsi_scores = np.zeros((len(loader)))
    
    all_path_grad = 0.0
    all_omic_grad = 0.0
    batch_idx = 0
    cur_epoch_fisher = {"wsi": 0.0, "omic": 0.0}


    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_id) in enumerate(loader):
        if epoch==0:
            wsi_omic_fisher_list['slide_id'].append(slide_id)
            wsi_omic_fisher_list['wsi_feature'].append(data_WSI)
            wsi_omic_fisher_list['omic_feature'].append((data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6))
            wsi_omic_fisher_list['label'].append(label)
            wsi_omic_fisher_list['event_time'].append(event_time)
            wsi_omic_fisher_list['wsi_fisher'].append(0)
            wsi_omic_fisher_list['omic_fisher'].append(0)

            
        if data_WSI.shape[0]>100000:   #privent out of memory
            data_WSI = data_WSI[:100000,:]
        data_WSI = data_WSI.to(device)
        
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
            
        global kld_value
        kld_value = args.annealing_agent()
        
        if args.generator:
            hazards, S, Y_hat, A, all_loss = model(stage=stage, train=True, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        else:
            hazards, S, Y_hat, A, all_loss, scores_patches  = model(stage=stage, train=True, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        
        loss = survey_loss + loss_reg
        if args.generator:
            loss += all_loss['recon_loss'] + all_loss['encode_wsi_loss']*args.beta +\
                (all_loss['kl_wsi'] + all_loss['flow_fit_loss'])*kld_value*0.5
            all_risk_wsi_scores[batch_idx] = all_loss['risk_wsi']
        else:
            loss += all_loss['encode_wsi_loss']*args.beta +\
                (all_loss['kl_wsi'] )*kld_value*0.5
            all_risk_wsi_scores[batch_idx] = all_loss['risk_wsi']

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time
        
        train_loss_surv += survey_loss.item()
        if args.generator:
            train_recon_loss += all_loss['recon_loss'].item()
            train_encode_wsi_loss += all_loss['encode_wsi_loss']
            train_kl_wsi += all_loss['kl_wsi']
            flow_fit_loss += all_loss['flow_fit_loss']

        loss = loss 
        train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss.item(), label.item(), float(event_time), float(risk))
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str+'\n')
            f.close()
            print(train_batch_str)
        loss = loss / gc + loss_reg
        loss.backward()


        #####FIMR
        scale_factor = 0.4 
        if slide_id in aug_slide_id:
            if fisher["wsi"] < fisher["omic"]:
            
                for name, param in model.named_parameters():
                    if ('omic' in name or 'sig' in name) and param.grad is not None:
                        factor= fisher["omic"]/((fisher["omic"]+fisher["wsi"])*0.5*scale_factor)
                        param.grad[:] = param.grad * factor
            if fisher["wsi"] >= fisher["omic"]:   
            # if fisher["wsi"] <= fisher["omic"]:
                for name, param in model.named_parameters():
                    if ('wsi' in name or 'path' in name) and param.grad is not None:
                        factor= fisher["wsi"]/((fisher["omic"]+fisher["wsi"])*0.5*scale_factor)
                        param.grad[:] = param.grad * factor#

        

        #####fisher compute
        cur_fisher = {"wsi": 0.0, "omic": 0.0}
        total_param_path = 0
        total_param_omic = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'path' in name:
                    cur_fisher["wsi"] += (param.grad.detach() ** 2).sum().item()/param.numel()
                    
                    total_param_path += param.numel()
                elif 'omic' in name:
                    cur_fisher["omic"] += (param.grad.detach() ** 2).sum().item()/param.numel()
                    
                    total_param_omic += param.numel()
        assert total_param_path == total_param_omic, f"Error: a ({total_param_path}) is not equal to b ({total_param_omic})"
        

        idx = wsi_omic_fisher_list['slide_id'].index(slide_id)   # slide_id index
        wsi_omic_fisher_list['omic_fisher'][idx] = cur_fisher["omic"]
        wsi_omic_fisher_list['wsi_fisher'][idx] = cur_fisher["wsi"]
        cur_epoch_fisher["wsi"] += cur_fisher["wsi"]
        cur_epoch_fisher["omic"] += cur_fisher["omic"]
        #####fisher compute

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()
            args.annealing_agent.step()

    
    fisher["wsi"] = cur_epoch_fisher["wsi"]
    fisher["omic"] = cur_epoch_fisher["omic"]
    # calculate loss and error for epoch
    train_loss_surv /= (len(loader))
    if args.generator: 
        train_recon_loss /= len(loader) 
        train_encode_wsi_loss /= len(loader) 
        train_kl_wsi /= len(loader) 
        flow_fit_loss /= len(loader) 

    train_loss /= (len(loader))
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_wsi_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_wsi_scores, tied_tol=1e-08)[0]
    if args.generator:
        train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_recon_loss: {:.4f}, train_encode_wsi_loss: {:.4f}, train_kl_wsi: {:.4f}, flow_fit_loss: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
                                epoch, train_loss_surv, train_recon_loss, train_encode_wsi_loss, train_kl_wsi, flow_fit_loss, train_loss, c_index_train)
        acc_str = 'c_index_wsi_train: {:.4f}'.format(c_index_wsi_train) 
        
    else: 
        train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
            epoch, train_loss_surv, train_loss, c_index_train)
        acc_str = ''
    print(train_epoch_str)
    print(acc_str)

    ###
    fisher_str = 'fisher: ' + str(fisher) + ', Epoch: ' + str(epoch)
    print(fisher_str)
    ###

    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
        f.write(acc_str+'\n')
        f.write(fisher_str + '\n')
        
    f.close()

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

        if args.generator:
            writer.add_scalar('train/recon_loss', train_recon_loss, epoch)
            writer.add_scalar('train/encode_wsi_loss', train_encode_wsi_loss, epoch)
            writer.add_scalar('train/kl_wsi', train_kl_wsi, epoch)
            writer.add_scalar('train/flow_fit_loss', flow_fit_loss, epoch)
    return fisher, wsi_omic_fisher_list


def validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_wsi_scores = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    merged_tensor = []
    label_merged = []

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_id) in enumerate(loader):
        
        if data_WSI.shape[0]>100000:   #privent out of memory
            data_WSI = data_WSI[:100000,:]
        data_WSI = data_WSI.cuda()
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.generator:
                hazards, S, Y_hat, A, all_loss  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            else:
                hazards, S, Y_hat, A, all_loss, scores_patches = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        
        loss = survey_loss + loss_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += survey_loss.item()

        val_loss += loss.item()

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        c_index = 0.
        
    if args.generator:
        val_epoch_str = 'Epoch: {}, c_index: {:.4f}'.format(epoch, c_index)
    else: 
        val_epoch_str = 'Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, c_index: {:.4f}'.format(
            epoch, val_loss_surv, val_loss, c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False

def validate_survival_coattn_missing(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None, averaged_sample=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_id) in enumerate(loader):

        if data_WSI.shape[0]>100000:   #privent out of memory
            data_WSI = data_WSI[:100000,:]
        data_WSI = data_WSI.cuda()
        if torch.randn(1) < args.missing_rate:
            omic_missing = True
        else: 
            omic_missing = False 

        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.generator:
                hazards, S, Y_hat, A, recon_loss  = model(omic_missing=omic_missing, train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                
            else:
                hazards, S, Y_hat, A  = model(label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        loss = survey_loss + loss_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += survey_loss.item()
        val_loss += loss.item()

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        c_index = 0
        
    val_epoch_str = "missing setting, val c-index: {:.4f}".format(c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val_missing/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val_missing/loss', val_loss, epoch)
        writer.add_scalar('val_missing/c-index', c_index, epoch)

    return patient_results, c_index, False

def validate_survival_coattn_patch_zero(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_wsi_scores = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_id) in enumerate(loader):
        
        if data_WSI.shape[0]>100000:   #privent out of memory
            data_WSI = data_WSI[:100000,:]
        data_WSI = data_WSI.cuda()
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.generator:
                hazards, S, Y_hat, A, all_loss  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            else:
                hazards, S, Y_hat, A, all_loss  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        ############patch score
        contribution = []
        with torch.no_grad():
            for i in range(data_WSI.shape[0]):
                data_WSI_mask = data_WSI.clone()
                data_WSI_mask[i,:] *= 0
                hazards, S, Y_hat, A, all_loss  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                survey_loss_patch = loss_fn(hazards=hazards, S=S, Y=label, c=c)
                contribution.append(survey_loss_patch - survey_loss)
        patch_scores = {'slide_id': slide_id, 'patch_scores': contribution}
        score_save_path = os.path.join(results_dir, 'heatmap', 'scores')
        os.makedirs(score_save_path, exist_ok=True)
        torch.save(patch_scores, os.path.join(score_save_path, slide_id + '_patch_scores.pt'))  
        ############


    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        c_index = 0.
        
    if args.generator:
        val_epoch_str = 'Epoch: {}, c_index: {:.4f}'.format(epoch, c_index)
    else: 
        val_epoch_str = 'Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, c_index: {:.4f}'.format(
            epoch, val_loss_surv, val_loss, c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False

def validate_survival_coattn_heatmap(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_wsi_scores = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    merged_tensor = []
    label_merged = []

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_id) in enumerate(loader):
        
        if data_WSI.shape[0]>100000:   #privent out of memory
            data_WSI = data_WSI[:100000,:]
        data_WSI = data_WSI.cuda()
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.generator:
                hazards, S, Y_hat, A, all_loss  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            else:
                hazards, S, Y_hat, A, all_loss, scores_patches_gene  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        gene_scores = {'slide_id': slide_id, 'gene_scores': scores_patches_gene[1]}
        score_save_gene = os.path.join(results_dir, 'heatmap', 'gene_score', '10x')
        os.makedirs(score_save_gene, exist_ok=True)
        torch.save(gene_scores, os.path.join(score_save_gene, slide_id + '_gene_scores.pt'))  
        print(slide_id,scores_patches_gene[1])

        patch_scores = {'slide_id': slide_id, 'patch_scores': scores_patches_gene[0]}
        score_save_path = os.path.join(results_dir, 'heatmap', 'patch_scores', '10x')
        os.makedirs(score_save_path, exist_ok=True)
        torch.save(patch_scores, os.path.join(score_save_path, slide_id + '_patch_scores.pt'))  

        merged_tensor.append(scores_patches_gene[0])
        label_merged.append(label)
        
        

    merged_tensor = torch.stack(merged_tensor, dim=0)
    t_SNE_data = {'merged_tensor': merged_tensor, 'label': label_merged}
    save_path_tsne = os.path.join(results_dir, 't-SNE', 'features.pt')
    os.makedirs(os.path.dirname(save_path_tsne), exist_ok=True)
    torch.save(t_SNE_data, save_path_tsne)

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        c_index = 0.
        
    if args.generator:
        val_epoch_str = 'Epoch: {}, c_index: {:.4f}'.format(epoch, c_index)
    else: 
        val_epoch_str = 'Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, c_index: {:.4f}'.format(
            epoch, val_loss_surv, val_loss, c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False

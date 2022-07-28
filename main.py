import cv2
import os
import sys
#sys.path.insert(0,"/data/hnljj/UDA_seg_package/package/")
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset
from perturbations.augmentations import augment, get_augmentation
from perturbations.fourier import fourier_mix
from perturbations.cutmix import cutmix_combine
from models import get_model
from models.ema import EMA
from utils.eval import Eval, synthia_set_16, synthia_set_13
from utils.loss import generate_ori_feature

def adentropy_ori(F1, feat, lamda,param_fc2=None, eta=1.0):
    if param_fc2 is None:
        out_t1,= F1(feat, reverse=True, eta=eta)
    else:
        out_t1,_ = F1(feat, param_fc2,reverse=True, eta=eta)
    out_t1=out_t1.reshape(out_t1.shape[0],-1).transpose(0,1).contiguous()
    out_t1 = F.softmax(out_t1)
    #print(out_t1.shape)
    #exit()
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

class Trainer():
    def __init__(self, cfg, logger, writer):

        # Args
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.logger = logger
        self.writer = writer

        # Counters
        self.epoch = 0
        self.iter = 0
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0

        # Metrics
        self.evaluator = Eval(self.cfg.data.num_classes)

        # Loss
        self.ignore_index = -1
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Model
        self.model, params = get_model(self.cfg)
        #print(params)
        #exit()
        # self.model = nn.DataParallel(self.model, device_ids=[0])  # TODO: test multi-gpu
        self.model.to(self.device)

        # EMA
        self.ema = EMA(self.model, self.cfg.ema_decay)

        # Optimizer
        if self.cfg.opt.kind == "SGD":
            self.optimizer = torch.optim.SGD(
                params, momentum=self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay)
        elif self.cfg.opt.kind == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(
                0.9, 0.99), weight_decay=self.cfg.opt.weight_decay)
        else:
            raise NotImplementedError()
        self.lr_factor = 10

        # Source
        if self.cfg.data.source.dataset == 'synthia':
            source_train_dataset = SYNTHIA_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = SYNTHIA_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'gta5':
            source_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        else:
            raise NotImplementedError()
        self.source_dataloader = DataLoader(
            source_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.source_loader.kwargs)
        self.source_val_dataloader = DataLoader(
            source_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Target
        if self.cfg.data.target.dataset == 'cityscapes':
            target_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
        else:
            raise NotImplementedError()
        self.target_dataloader = DataLoader(
            target_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.target_loader.kwargs)
        #print(self.cfg.data.loader.kwargs)
        #exit()
        self.target_val_dataloader = DataLoader(
            target_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)
        self.use_NCE=True
        self.nofalse=False
        self.criterion = nn.CrossEntropyLoss().cuda()
        # Perturbations
        if self.cfg.lam_aug > 0:
            self.aug = get_augmentation()
    
    def load_shadowdict(self):
        self.ema.shadowdict(self.shadow_dict)
    
    def train(self):

        # Loop over epochs
        self.continue_training = True
        while self.continue_training:

            # Train for a single epoch
            self.train_one_epoch()

            # Use EMA params to evaluate performance
            self.ema.apply_shadow()#use ema carry out test-phase
            self.ema.model.eval()
            self.ema.model.cuda()

            # Validate on source (if possible) and target
            #if self.cfg.data.source_val_iterations > 0:
                #self.validate(mode='source')
            #if self.epoch > 0:    
            PA, MPA, MIoU, FWIoU = self.validate()

            # Restore current (non-EMA) params for training
            self.ema.restore()

            # Log val results
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

            # Save checkpoint if new best model
            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.iter
                self.logger.info("=> Saving a new best checkpoint...")
                self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                    self.best_MIou, self.best_iter))
                self.save_checkpoint('best.pth')
            else:
                self.logger.info("=> The MIoU of val did not improve.")
                self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                    self.best_MIou, self.best_iter))
            self.epoch += 1

        # Save final checkpoint
        self.logger.info("=> The best MIou was {:.3f} at iter {}".format(
            self.best_MIou, self.best_iter))
        self.logger.info(
            "=> Saving the final checkpoint to {}".format('final.pth'))
        self.save_checkpoint('final.pth')

    def train_one_epoch(self):

        # Load and reset
        self.model.train()
        self.evaluator.reset()

        # Helper
        def unpack(x):
            return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

        # Training loop
        total = min(len(self.source_dataloader), len(self.target_dataloader))
        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(
            zip(self.source_dataloader, self.target_dataloader),
            total=total, desc=f"Epoch {self.epoch + 1}"
        )):

            # Learning rate
            self.poly_lr_scheduler(optimizer=self.optimizer)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.iter)

            # Losses
            losses = {}

            ##########################
            # Source supervised loss #
            ##########################
            x, y, _ = batch_s

            if True:  # For VS Code collapsing

                # Data
                x = x.to(self.device)

                y = y.squeeze(dim=1).to(device=self.device,
                                        dtype=torch.long, non_blocking=True)

                x_source_aug,x_source_w, y_source_aug_1 = augment(
                    images=x.cpu(), labels=y.detach().cpu(), aug=self.aug)
                y_source_aug_1 = y_source_aug_1.to(device=self.device, non_blocking=True)
                mask_source_list=[]
                label_source_list=[]
                x_s_masked_list=[]
                num_class_select_list=[]

                for batch_id in range(y_source_aug_1.shape[0]):
                    class_pool_s=(y_source_aug_1[batch_id].unique()>4).float()
                    #print(class_pool_s)
                    #print(y_s.shape,y_source_aug_1.shape)
                    #exit()
                    if class_pool_s.sum().long()==1:
                        num_class_select=1
                    elif class_pool_s.sum().long()==0:
                        num_class_select=0
                    #elif class_pool_s.sum().long()<4:
                        #num_class_select=torch.randint(1,(class_pool_s.sum()/2).long(),(1,)).cpu().numpy()[0]
                    else:
                        num_class_select=int((class_pool_s.sum()/2).long().cpu().numpy())#[0]#torch.randint(1,4,(1,)).cpu().numpy()[0]

                    num_class_select_list.append(num_class_select)
                    if num_class_select>0:
                        id_pool=torch.tensor(list(np.array([1 for x_id in range(class_pool_s.sum().long())]))).reshape(1,-1).float()
                        selected_id_source=torch.multinomial(id_pool, num_class_select, replacement=False).long().reshape(-1,1).cuda()+ (len(y_source_aug_1[batch_id].unique())-class_pool_s.sum().long())-1        
                        mask_source=0
                        label_source=0

                        for select_id in selected_id_source:

                            mask_source=mask_source+(y_source_aug_1[batch_id]==y_source_aug_1[batch_id].unique()[select_id]).float()

                            label_source=label_source+y_source_aug_1[batch_id].unique()[select_id]*(y_source_aug_1[batch_id]==y_source_aug_1[batch_id].unique()[select_id]).float()
                    else:
                        mask_source=0
                        label_source=0                    
                        mask_source=mask_source+(y_source_aug_1[batch_id]==(y_source_aug_1[batch_id].unique()[0]+100000)).float()
                        label_source=label_source+0*(y_source_aug_1[batch_id]==(y_source_aug_1[batch_id].unique()[0]+100000)).float() 
                    x_s_masked_list.append(x_source_w[batch_id].cuda()*mask_source)
                    mask_source_list.append( mask_source)
                    label_source_list.append( label_source)

                x=x#[:1,:,:,:]
                y=y#[:1,:,:]
                # Fourier mix: source --> target

                if self.cfg.source_fourier:
                    x = fourier_mix(src_images=x, tgt_images=batch_t[0].to(
                        self.device), L=self.cfg.fourier_beta)

                # Forward
                pred = self.model(x)
                x_source=x.clone()

                pred_1, pred_2,pred_3 =pred[0],pred[1],pred[2]# unpack(pred)
                pred_1_s=pred_1.clone()#[:1]
                y_s=y.clone()#[:1]
                input_size_s=pred_3.size()[2:]
                

                y_s_small = F.interpolate(y.reshape(y.shape[0],-1,y.shape[1],y.shape[2]).float(), size=input_size_s,mode='nearest').long()
                y_s_small=y_s_small[:,0,:,:]
                   
                    

                # Loss (source)
                loss_source_1 = self.loss(pred_1, y)
                if self.iter>1000000:
                    pred_s_t = self.model.prototype(pred[3])   
                    loss_source_1=loss_source_1#+0.005*self.loss(pred_s_t, y_s_small) 
 
                if self.cfg.aux:
                    loss_source_2 = self.loss(pred_2, y) * self.cfg.lam_aux
                    loss_source = loss_source_1 + loss_source_2
                else:
                    loss_source = loss_source_1
                         
                # Backward
                loss_source.backward()
                #print('lll')
                #exit(0)
                # Clean up
                losses['source_main'] = loss_source_1.cpu().item()
                if self.cfg.aux:
                    losses['source_aux'] = loss_source_2.cpu().item()
                del x, y, loss_source, loss_source_1, loss_source_2

            ######################
            # Target Pseudolabel #
            ######################
            x, _, _ = batch_t

            crop_size=512

            
            x = x.to(self.device)#[:,:,idx_ljj:idx_ljj+crop_size,idy_ljj:idy_ljj+crop_size]
            if self.use_NCE:
                with torch.no_grad():
                    pred = self.model(x.to(self.device))
                    pred_1_soft, pred_2_soft,pred_3_soft = pred[0],pred[1],pred[2]
            else:
                with torch.no_grad():
                    pred = self.model(x.to(self.device))
                    #pred_1_soft, pred_2_soft = unpack(pred)                

                # Substep 2: convert soft predictions to hard predictions           
            # First step: run non-augmented image though model to get predictions
            with torch.no_grad():

                # Substep 1: forward pass
                #pred=pred.detach()
                pred_1, pred_2,pred_3 = pred[0],pred[1],pred[2]
                pred_1, pred_2,pred_3 = pred_1.detach(), pred_2.detach(), pred_3.detach()
                # Substep 2: convert soft predictions to hard predictions
                pred_P_1 = F.softmax(pred_1, dim=1)
                label_1 = torch.argmax(pred_P_1.detach(), dim=1)
                maxpred_1, argpred_1 = torch.max(pred_P_1.detach(), dim=1)
                #print((label_1-argpred_1).abs().sum(),self.cfg.pseudolabel_threshold)
                #exit()
                T = 0.0
                mask_1 = (maxpred_1 > T)
                ignore_tensor = torch.ones(1).to(
                    self.device, dtype=torch.long) * self.ignore_index
                label_1 = torch.where(mask_1, label_1, ignore_tensor)
                #print(label_source.shape)
                if num_class_select<0:
                    label_1=label_1*(1-mask_source_crop.cuda())+label_source[:,40:680,:]
                if self.cfg.aux:
                    pred_P_2 = F.softmax(pred_2, dim=1)
                    maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)
                    pred_c = (pred_P_1 + pred_P_2) / 2
                    maxpred_c, argpred_c = torch.max(pred_c, dim=1)
                    mask = (maxpred_1 > T) | (maxpred_2 > T)
                    label_2 = torch.where(mask, argpred_c, ignore_tensor)
                    if num_class_select<0:
                        label_2=label_2*(1-mask_source_crop.cuda())+label_source[:,40:680,:]    
            ############
            # Aug loss #
            ############
            if self.cfg.lam_aug > 0:
                #x=xidy_ljj
                # Second step: augment image and label              
                x_aug,x_w, y_aug_1 = augment(
                    images=x.cpu(), labels=label_1.detach().cpu(), aug=self.aug)
                y_aug_1 = y_aug_1.to(device=self.device, non_blocking=True)
                for batch_t_id in range(y_aug_1.shape[0]):
                    
                    if num_class_select_list[batch_t_id]>0:
                        x_aug[batch_t_id:batch_t_id+1]=x_s_masked_list[batch_t_id].cuda()+x_aug[batch_t_id].cuda()*(1-mask_source_list[batch_t_id].cuda())
                        x_w[batch_t_id:batch_t_id+1]=x_s_masked_list[batch_t_id].cuda()+x_w[batch_t_id].cuda()*(1-mask_source_list[batch_t_id].cuda())

                        y_aug_1[batch_t_id:batch_t_id+1]=(y_aug_1[batch_t_id:batch_t_id+1]*(1-mask_source_list[batch_t_id].cuda())+label_source_list[batch_t_id]).long()

                if self.cfg.aux:
                    _, _,y_aug_2 = augment(
                        images=x.cpu(), labels=label_2.detach().cpu(), aug=self.aug)
                    y_aug_2 = y_aug_2.to(device=self.device, non_blocking=True)

                # Third step: run augmented image through model to get predictions
                pred_aug = self.model(x_aug.to(self.device))
                pred_aug_1, pred_aug_2, pred_aug_3 = pred_aug[0],pred_aug[1],pred_aug[2]
                if not self.use_NCE:
                    with torch.no_grad():
                        #pred_aug = self.model(x_aug.to(self.device))
                        pred_w = self.model(x_w.to(self.device))
                        #pred_w = self.model(x_w.to(self.device))
                        #pred_aug_1, pred_aug_2, pred_aug_3 = pred_aug[0],pred_aug[1],pred_aug[2]
                        pred_w_1, pred_w_2, pred_w_3 = pred_w[0],pred_w[1],pred_w[2]
                        pred_P_1_w = F.softmax(pred_w_1, dim=1)
                        label_1_w = torch.argmax(pred_P_1_w.detach(), dim=1)
                        maxpred_1_w, argpred_1_w = torch.max(pred_P_1_w.detach(), dim=1)
                        #T = self.cfg.pseudolabel_threshold
                        mask_1_w = (maxpred_1_w > T)
                        #print(self.ignore_index,ignore_tensor)
                        #exit()
                        ignore_tensor = torch.ones(1).to(
                            self.device, dtype=torch.long) * self.ignore_index
                        label_1_w = torch.where(mask_1_w, label_1_w, ignore_tensor)
                        
                        pred_P_2_w = F.softmax(pred_w_2.detach(), dim=1)
                        maxpred_2_w, argpred_2_w = torch.max(pred_P_2_w.detach(), dim=1)
                        pred_c = (pred_P_1_w + pred_P_2_w) / 2
                        maxpred_c_w, argpred_c_w = torch.max(pred_c, dim=1)
                        mask = (maxpred_1_w > T) | (maxpred_2_w > T)
                        label_2_w = torch.where(mask, argpred_c_w, ignore_tensor)
                else:  
                    #pred_aug = self.model(x_aug.to(self.device))
                    
                    pred_w = self.model(x_w.to(self.device))


                    pred_w_1, pred_w_2, pred_w_3 = pred_w[0],pred_w[1],pred_w[2]
                    pred_P_1_w = F.softmax(pred_w_1, dim=1)
                    label_1_w = torch.argmax(pred_P_1_w.detach(), dim=1)
                    maxpred_1_w, argpred_1_w = torch.max(pred_P_1_w.detach(), dim=1)
                    #T = self.cfg.pseudolabel_threshold
                    mask_1_w = (maxpred_1_w > T)
                    mask_1_w_f = (maxpred_1_w > self.cfg.Tmax)

                    mask_1_w_b = ((maxpred_1_w < self.cfg.Tmax).float() * (maxpred_1_w > 0.9299).float())>0

                    ignore_tensor = torch.ones(1).to(
                        self.device, dtype=torch.long) * self.ignore_index
                    label_1_w = torch.where(mask_1_w, label_1_w, ignore_tensor)
                    label_1_w_f = torch.where(mask_1_w_f, argpred_1_w, ignore_tensor)
                    use_erode=False
                    if use_erode:
                        kernel = np.ones((60, 60), np.uint8)
                        mask_1_w_f_numpy=mask_1_w_f.cpu().numpy().astype(dtype=np.uint8)
                        erode_mask=[]
                        for i in range(mask_1_w_f_numpy.shape[0]):
                            erode_mask.append(torch.from_numpy((cv2.erode(mask_1_w_f_numpy[i]*255, kernel)>254).astype(dtype=np.uint8)).reshape(1,mask_1_w_f_numpy.shape[1],mask_1_w_f_numpy.shape[2]))
                        erode_mask=torch.cat(erode_mask)

                    if True:

                        mask_1_w_f_4=mask_1_w_f.reshape(mask_1_w_f.shape[0],-1,mask_1_w_f.shape[1],mask_1_w_f.shape[2]).float()
                        #erode_mask_4=erode_mask.reshape(erode_mask.shape[0],-1,erode_mask.shape[1],erode_mask.shape[2]).float()
                        s_t_x_aug=x_source_aug.cuda()*(1-mask_1_w_f_4)+x_aug.cuda()*mask_1_w_f_4
                        #s_t_x_w=x_source_w.cuda()*(1-mask_1_w_f_4)+x_w.cuda()*mask_1_w_f_4
                        s_t_y=y_source_aug_1*(1-mask_1_w_f.float())+label_1_w_f*mask_1_w_f.float()
                        if use_erode:
                            erode_mask_4=erode_mask.reshape(erode_mask.shape[0],-1,erode_mask.shape[1],erode_mask.shape[2]).float()
                            s_t_x_aug=x_source_aug.cuda()*(erode_mask_4.cuda())+s_t_x_aug*(1-erode_mask_4.cuda())
                            #print(s_t_y.unique())
                            s_t_y=s_t_y*(1-erode_mask.cuda())+ignore_tensor*erode_mask.cuda()
                            #print(s_t_y.unique())
                            #exit()
                        pred_s_t_mix=self.model(s_t_x_aug.to(self.device))
                        #pred_s_t_mix_w=self.model(s_t_x_w.to(self.device))
                        pred_s_t_mix_1, pred_s_t_mix_2, pred_s_t_mix_3 = pred_s_t_mix[0],pred_s_t_mix[1],pred_s_t_mix[2]
                        loss_aug_st = self.loss(pred_s_t_mix_1[:1], s_t_y[:1].long())
                        #loss_aug_st_w = self.loss(pred_s_t_mix_w[0][:1], s_t_y[:1].long())
                        #loss_aug_1.backward()
                        #del loss_aug_1
                        #exit()                    
                    for batch_id_label in range(label_1_w.shape[0]):
                        if num_class_select_list[batch_id_label]>0:
                            label_1_w_f[batch_id_label:batch_id_label+1]=label_1_w_f[batch_id_label:batch_id_label+1]*(1-mask_source_list[batch_id_label].cuda())+label_source_list[batch_id_label]
                            #label_1_w_f=label_1_w_f.long()                   
                    label_1_w_b = torch.where(mask_1_w_b, argpred_1_w, ignore_tensor)
                    pred_P_2_w = F.softmax(pred_w_2.detach(), dim=1)
                    maxpred_2_w, argpred_2_w = torch.max(pred_P_2_w.detach(), dim=1)
                    pred_c = (pred_P_1_w + pred_P_2_w) / 2
                    maxpred_c_w, argpred_c_w = torch.max(pred_c, dim=1)
                    mask = (maxpred_1_w > T) | (maxpred_2_w > T)
                    label_2_w = torch.where(mask, argpred_c_w, ignore_tensor)                

                # Fourth step: calculate loss
                loss_aug_1 = self.loss(pred_aug_1, y_aug_1) * \
                    self.cfg.lam_aug+loss_aug_st*self.cfg.mix_weight#+loss_aug_st_w*0.1
                #print(y_aug_1.shape)
                y_aug_1_fix=y_aug_1.clone()
                #exit()
                if self.cfg.aux:
                    loss_aug_2 = self.loss(pred_aug_2, label_2_w) * \
                        self.cfg.lam_aug * self.cfg.lam_aux
                    loss_aug = loss_aug_1 + loss_aug_2
                else:
                    loss_aug = loss_aug_1
                if self.use_NCE:
                    #pred_w = self.model(x_w.to(self.device))
                    
                    pred_w_1=pred_w[2]#[:1]
                    pred_aug_1=pred_aug_3#[:1]


                    batch_target=pred_aug_3.shape[0]
                   
                    for batch_id_NCE in range(pred_aug_3.shape[0]):

                        input_size=pred_aug_1.size()[2:]
                        if self.nofalse:
                            y_small_temp=y_aug_1_fix[batch_id_NCE:batch_id_NCE+1]
                            y_small=F.interpolate(y_small_temp.reshape(y_small_temp.shape[0],-1,y_small_temp.shape[1],y_small_temp.shape[2]).float(), size=input_size,mode='nearest').long()

                            y_small=y_small.reshape(-1)

                            class_small=y_small.unique()
                            n_class_small=len(class_small)
                            if n_class_small>1:
                                pred_list_all=pred_w_1[batch_id_NCE:batch_id_NCE+1].reshape(pred_w_1.shape[1],pred_w_1.shape[2]*pred_w_1.shape[3]).transpose(0,1).contiguous()
                                pred_aug_1_list_all=pred_aug_1[batch_id_NCE:batch_id_NCE+1].reshape(pred_aug_1.shape[1],pred_aug_1.shape[2]*pred_aug_1.shape[3]).transpose(0,1).contiguous()
                                pred_aug_1_softmax_all = F.softmax(pred_aug_1_list_all, dim=1)
                                pred_P_1_softmax_all = F.softmax(pred_list_all, dim=1)                             
                                pred_list_new={}  
                                pred_list_reverse_new={}

                                class2id={}

                            
                                for class_id_temp in class_small:
                                    class_ids=(y_small==class_id_temp).nonzero()[:,0]

                                    class2id[class_id_temp]=class_ids

                                    pred_list_new[class_id_temp]=torch.cat([pred_aug_1_softmax_all[class_ids],pred_P_1_softmax_all[class_ids]])
                                    pred_list_reverse_new[class_id_temp]=torch.cat([pred_P_1_softmax_all[class_ids],pred_aug_1_softmax_all[class_ids]])


                        y_aug_1=label_1_w_f[batch_id_NCE:batch_id_NCE+1]

                        y_aug_1 = F.interpolate(y_aug_1.reshape(y_aug_1.shape[0],-1,y_aug_1.shape[1],y_aug_1.shape[2]).float(), size=input_size,mode='nearest').long()

                        mask_1_w_bjj = (y_aug_1==-1).float().long()
     
                        mask_1_w_bjj=mask_1_w_bjj.reshape(-1)
                        index_b=torch.nonzero(mask_1_w_bjj)

                        y_aug_1=y_aug_1[:,0,:,:]
                        mask_pred_uni=y_aug_1[batch_id_NCE:batch_id_NCE+1].unique()


                        

                        

                        pred_aug_1_list=[]
                        pred_list=[]
                        pred_SO_list=[]
                        pred_sw_list=[]
                        pred_wb_list=[]
                        pred_sb_list=[]
                        num_class_unique=len(mask_pred_uni)

                        if -1 in mask_pred_uni:
                            num_class_unique=num_class_unique-1
                        temp_flag=False  

                        for classid in mask_pred_uni:
                            if classid==-1:
                                continue

                            mask_w_id=(y_aug_1==classid).float().reshape(1,-1,y_aug_1.shape[1],y_aug_1.shape[2])

                            logit_weak=(pred_w_1[batch_id_NCE:batch_id_NCE+1]*mask_w_id).sum(2).sum(2)/mask_w_id.sum()
                            
                            mask_s_id=(y_aug_1==classid).float().reshape(1,-1,y_aug_1.shape[1],y_aug_1.shape[2])
                            logit_s=(pred_aug_1[batch_id_NCE:batch_id_NCE+1]*mask_s_id).sum(2).sum(2)/ mask_s_id.sum()
                           

                            
                            pred_aug_1_list.append(logit_s)
                            pred_list.append(logit_weak)

                        if len(pred_aug_1_list)==0:
                            print('pred_aug_1_list is empty')
                        else:
                            if True:
                                pred_aug_1_list=torch.cat(pred_aug_1_list, 0)
                                pred_list=torch.cat(pred_list, 0) 
                                pred_aug_1_softmax = F.softmax(pred_aug_1_list, dim=1)
                                pred_P_1_softmax = F.softmax(pred_list, dim=1)
                                out1_x_s_c=torch.cat([pred_aug_1_softmax,pred_P_1_softmax],0)
                                NCE_2=torch.mm(out1_x_s_c,out1_x_s_c.transpose(0,1).contiguous())

                                num_class_unique=num_class_unique
                                unit_1=torch.eye(num_class_unique*2).cuda()

                                NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                                a=[idx for idx in range(num_class_unique)]
                                gt_labels_cls=torch.from_numpy(np.array(a)).cuda()

                                gt_labels_cls_cross=torch.cat([gt_labels_cls[:num_class_unique]+num_class_unique,gt_labels_cls[:num_class_unique]])  
                                loss_aug = loss_aug+self.cfg.NCE_weight*self.criterion(7*NCE_2, gt_labels_cls_cross)/batch_target                       


                    

                        

                        if True:
                            #print('ljjj')
                            #exit()
                            pred_list_all=pred_w_1[batch_id_NCE:batch_id_NCE+1].reshape(pred_w_1.shape[1],pred_w_1.shape[2]*pred_w_1.shape[3]).transpose(0,1).contiguous()
                            pred_aug_1_list_all=pred_aug_1[batch_id_NCE:batch_id_NCE+1].reshape(pred_aug_1.shape[1],pred_aug_1.shape[2]*pred_aug_1.shape[3]).transpose(0,1).contiguous()
                            


                            #print(pred_aug_1_list_all.shape)
                            pred_aug_1_softmax_all = F.softmax(pred_aug_1_list_all, dim=1)#[index_b,:][:,0,:]
                            pred_P_1_softmax_all = F.softmax(pred_list_all, dim=1)#[index_b,:][:,0,:]      
                            #print(pred_aug_1_softmax_all.shape)
                            #exit()
                            out1_x_s_c=torch.cat([pred_aug_1_softmax_all,pred_P_1_softmax_all],0)
                            NCE_2_all=torch.mm(out1_x_s_c,out1_x_s_c.transpose(0,1).contiguous())                        
                            num_class_unique=pred_aug_1_softmax_all.shape[0]#.shape[2]*pred_w_1.shape[3]
                            unit_1=torch.eye(num_class_unique*2).cuda()
                            #print(unit_1)
                            #exit()
                            NCE_2_all=NCE_2_all*(1-unit_1)+(-100000)*unit_1
                            a=[idx for idx in range(num_class_unique)]
                            gt_labels_cls=torch.from_numpy(np.array(a)).cuda()
                            #print(NCE_2.shape)
                            #exit()
                            #print(gt_labels_cls)
                            #print(gt_labels_cls.shape)
                            gt_labels_cls_cross_all=torch.cat([gt_labels_cls[:num_class_unique]+num_class_unique,gt_labels_cls[:num_class_unique]])                    
                            #print(NCE_2.shape)
                            #print(gt_labels_cls_cross)
                            #exit()
                            if self.epoch>-1:
                                loss_aug = loss_aug+self.cfg.NCE_weight*self.criterion(20*NCE_2_all, gt_labels_cls_cross_all)/batch_target#+loss_st
                            else:
                                loss_aug = loss_aug+0.05*self.criterion(20*NCE_2, gt_labels_cls_cross) 
                # Backward
                loss_aug.backward()

                # Clean up
                losses['aug_main'] = loss_aug_1.cpu().item()
                if self.cfg.aux:
                    losses['aug_aux'] = loss_aug_2.cpu().item()
                del pred_aug, pred_aug_1, pred_aug_2, loss_aug, loss_aug_1, loss_aug_2

            ################
            # Fourier Loss #
            ################
            if self.cfg.lam_fourier > 0:

                # Second step: fourier mix
                x_fourier = fourier_mix(
                    src_images=x.to(self.device),
                    tgt_images=batch_s[0].to(self.device),
                    L=self.cfg.fourier_beta)

                # Third step: run mixed image through model to get predictions
                pred_fourier = self.model(x_fourier.to(self.device))
                pred_fourier_1, pred_fourier_2 = unpack(pred_fourier)

                # Fourth step: calculate loss
                loss_fourier_1 = self.loss(pred_fourier_1, label_1) * \
                    self.cfg.lam_fourier

                if self.cfg.aux:
                    loss_fourier_2 = self.loss(pred_fourier_2, label_2) * \
                        self.cfg.lam_fourier * self.cfg.lam_aux
                    loss_fourier = loss_fourier_1 + loss_fourier_2
                else:
                    loss_fourier = loss_fourier_1

                # Backward
                loss_fourier.backward()

                # Clean up
                losses['fourier_main'] = loss_fourier_1.cpu().item()
                if self.cfg.aux:
                    losses['fourier_aux'] = loss_fourier_2.cpu().item()
                del pred_fourier, pred_fourier_1, pred_fourier_2, loss_fourier, loss_fourier_1, loss_fourier_2

            ###############
            # CutMix Loss #
            ###############
            if self.cfg.lam_cutmix > 0:

                # Second step: CutMix
                x_cutmix, y_cutmix = cutmix_combine(
                    images_1=x,
                    labels_1=label_1.unsqueeze(dim=1),
                    images_2=batch_s[0].to(self.device),
                    labels_2=batch_s[1].unsqueeze(dim=1).to(self.device, dtype=torch.long))
                y_cutmix = y_cutmix.squeeze(dim=1)

                # Third step: run mixed image through model to get predictions
                pred_cutmix = self.model(x_cutmix)
                pred_cutmix_1, pred_cutmix_2 = unpack(pred_cutmix)

                # Fourth step: calculate loss
                loss_cutmix_1 = self.loss(pred_cutmix_1, y_cutmix) * \
                    self.cfg.lam_cutmix
                if self.cfg.aux:
                    loss_cutmix_2 = self.loss(pred_cutmix_2, y_cutmix) * \
                        self.cfg.lam_cutmix * self.cfg.lam_aux
                    loss_cutmix = loss_cutmix_1 + loss_cutmix_2
                else:
                    loss_cutmix = loss_cutmix_1

                # Backward
                loss_cutmix.backward()

                # Clean up
                losses['cutmix_main'] = loss_cutmix_1.cpu().item()
                if self.cfg.aux:
                    losses['cutmix_aux'] = loss_cutmix_2.cpu().item()
                del pred_cutmix, pred_cutmix_1, pred_cutmix_2, loss_cutmix, loss_cutmix_1, loss_cutmix_2

            ###############
            # CutMix Loss #
            ###############

            # Step optimizer if accumulated enough gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update model EMA parameters each step
            self.ema.update_params()

            # Calculate total loss
            total_loss = sum(losses.values())

            # Log main losses
            for name, loss in losses.items():
                self.writer.add_scalar(f'train/{name}', loss, self.iter)

            # Log
            if batch_idx % 100 == 0:
                log_string = f"[Epoch {self.epoch}]\t"
                log_string += '\t'.join([f'{n}: {l:.3f}' for n, l in losses.items()])
                self.logger.info(log_string)

            # Increment global iteration counter
            self.iter += 1

            # End training after finishing iterations
            if self.iter > self.cfg.opt.iterations:
                self.continue_training = False
                return

        # After each epoch, update model EMA buffers (i.e. batch norm stats)
        self.ema.update_buffer()

    @ torch.no_grad()
    def validate(self, mode='target'):
        """Validate on target"""
        self.logger.info('Validating')
        self.evaluator.reset()
        self.model.eval()

        # Select dataloader
        if mode == 'target':
            val_loader = self.target_val_dataloader
        elif mode == 'source':
            val_loader = self.source_val_dataloader
        else:
            raise NotImplementedError()

        # Loop
        for val_idx, (x, y, id) in enumerate(tqdm(val_loader, desc=f"Val Epoch {self.epoch + 1}")):
            if mode == 'source' and val_idx >= self.cfg.data.source_val_iterations:
                break

            # Forward
            x = x.to(self.device)
            y = y.to(device=self.device, dtype=torch.long)
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]

            # Convert to numpy
            label = y.squeeze(dim=1).cpu().numpy()
            argpred = np.argmax(pred.data.cpu().numpy(), axis=1)

            # Add to evaluator
            self.evaluator.add_batch(label, argpred)

        # Tensorboard images
        vis_imgs = 2
        images_inv = inv_preprocess(x.clone().cpu(), vis_imgs, numpy_transform=True)
        labels_colors = decode_labels(label, vis_imgs)
        preds_colors = decode_labels(argpred, vis_imgs)
        for index, (img, lab, predc) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image(str(index) + '/images', img, self.epoch)
            self.writer.add_image(str(index) + '/labels', lab, self.epoch)
            self.writer.add_image(str(index) + '/preds', predc, self.epoch)

        # Calculate and log
        if self.cfg.data.source.kwargs.class_16:
            PA = self.evaluator.Pixel_Accuracy()
            MPA_16, MPA_13 = self.evaluator.Mean_Pixel_Accuracy()
            MIoU_16, MIoU_13 = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU_13 = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC_16, PC_13 = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                self.epoch, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                self.epoch, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA_16', MPA_16, self.epoch)
            self.writer.add_scalar('MIoU_16', MIoU_16, self.epoch)
            self.writer.add_scalar('FWIoU_16', FWIoU_16, self.epoch)
            self.writer.add_scalar('MPA_13', MPA_13, self.epoch)
            self.writer.add_scalar('MIoU_13', MIoU_13, self.epoch)
            self.writer.add_scalar('FWIoU_13', FWIoU_13, self.epoch)
            PA, MPA, MIoU, FWIoU = PA, MPA_13, MIoU_13, FWIoU_13
        else:
            PA = self.evaluator.Pixel_Accuracy()
            MPA = self.evaluator.Mean_Pixel_Accuracy()
            MIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename='checkpoint.pth'):
        torch.save({
            'epoch': self.epoch + 1,
            'iter': self.iter,
            'state_dict': self.ema.model.state_dict(),
            'shadow': self.ema.shadow,
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        #print('shadow' in checkpoint)
        #exit()
        if 'shadow' in checkpoint:
            shadow_dict_temp=checkpoint['shadow']
            self.model.load_state_dict(shadow_dict_temp,strict=False)
            self.shadow_dict={
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
            }
        # Get model state dict
        if not self.cfg.train and 'shadow' in checkpoint:
            state_dict = checkpoint['shadow']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DP/DDP if it exists
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}
        if False:
            bad_key=[]
            new_fc={}
            for k in state_dict.keys():
                if 'layer5' in k or 'layer6' in k:
                    bad_key.append(k)
                    state_dict[k]=state_dict[k].clone()[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]
                    #print(state_dict[k].shape)
                    #exit()
            #for k in bad_key:
                #state_dict.pop(k)
        #exit()
        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict,strict=False)
        else:
            self.model.load_state_dict(state_dict,strict=False)
        self.logger.info(f"Model loaded successfully from {filename}")

        # Load optimizer and epoch
        if self.cfg.train and self.cfg.model.resume_from_checkpoint and False:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"Optimizer loaded successfully from {filename}")
            if 'epoch' in checkpoint and 'iter' in checkpoint:
                self.epoch = checkpoint['epoch']
                self.iter = checkpoint['iter'] if 'iter' in checkpoint else checkpoint['iteration']
                #print(self.iter)
                #print(self.epoch)
                #exit()
                self.logger.info(f"Resuming training from epoch {self.epoch} iter {self.iter}")
        else:
            self.logger.info(f"Did not resume optimizer")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
        init_lr = self.cfg.opt.lr if init_lr is None else init_lr
        iter = self.iter if iter is None else iter
        max_iter = self.cfg.opt.iterations if max_iter is None else max_iter
        power = self.cfg.opt.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr


@hydra.main(config_path='configs', config_name='gta5')
def main(cfg: DictConfig):

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    #print(cfg.Tmax)
    #exit()
    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if cfg.train:
        base_dir = "/data/lijj/pixmatch_output/"  
        list = os.listdir(base_dir)  
        print(list)
        #exit()
        filelist = base_dir 
        new_time=-1000
        for i in range(0, len(list)):  
            path = os.path.join(base_dir,list[i])  
            #if os.path.isfile(path):  
              
            if os.path.getmtime(path)>new_time:
                filelist=path
                new_time=os.path.getmtime(path)
            #print(timestamp)
        print(filelist)
        #filelist = base_dir
        list = os.listdir(filelist) 
        print(list)
        filelist2=filelist
        #exit()
        new_time=-1000
        for i in range(0, len(list)):  
            path = os.path.join(filelist,list[i])  
            #if os.path.isfile(path):  
            #print(path)
            if os.path.getmtime(path)>new_time:
                filelist2=path
                new_time=os.path.getmtime(path)
        print(filelist2)
        #exit()
        #for i in range(0, len(filelist)):  
            #path = os.path.join(base_dir, filelist[i])  
            #if os.path.isdir(path):  
                #continue  
            #timestamp = os.path.getmtime(path)  
            #print(timestamp)  
            #ts1 = os.stat(path).st_mtime  
            #print(ts1)  
              
            #date = datetime.datetime.fromtimestamp(timestamp)  
            #print(list[i],' 最近修改时间是: ',date.strftime('%Y-%m-%d %H:%M:%S'))     
        #exit()
        record_dir=filelist2
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/configs ' +record_dir)
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/datasets ' +record_dir)
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/models ' +record_dir)
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/perturbations ' +record_dir)
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/pretrained ' +record_dir)
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/scripts ' +record_dir)    
        os.system('cp -r /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/utils ' +record_dir)      
        os.system('cp  /home/lijj/domain_seg/code/pixmatch-master/pixmatch-master/main.py ' +record_dir)
        
    #os.system('cp  /ghome/lijj/DA/pixmatch-master/*.sh ' +record_dir)
    #exit()
    # Monitoring
    if cfg.wandb:
        import wandb
        wandb.init(project='pixmatch', name=cfg.name, config=cfg, sync_tensorboard=True)
    writer = SummaryWriter(cfg.name)

    # Trainer
    trainer = Trainer(cfg=cfg, logger=logger, writer=writer)
    #print(cfg)
    #print(cfg.model.checkpoint)
    #exit()
    # Load pretrained checkpoint
    if cfg.model.checkpoint:
        assert Path(cfg.model.checkpoint).is_file(), f'not a file: {cfg.model.checkpoint}'
        trainer.load_checkpoint(cfg.model.checkpoint)
        if cfg.model.resume_from_checkpoint_finetune:
            #print('enter_resume')
            #exit()
            trainer.load_shadowdict()

    # Print configuration
    logger.info('\n' + OmegaConf.to_yaml(cfg))

    # Train
    if cfg.train:
        trainer.train()

    # Evaluate
    else:
        trainer.validate()
        trainer.evaluator.Print_Every_class_Eval(
            out_16_13=(int(cfg.data.num_classes) in [16, 13]))


if __name__ == '__main__':
    main()


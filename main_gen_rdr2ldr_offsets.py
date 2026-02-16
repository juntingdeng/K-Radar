import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseConv3d, SparseInverseConv3d
import argparse
import random
import pickle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus
from models.skeletons.rdr_base import RadarBase
from models.generatives.unet import *
from models.generatives.generative import *

from dataset_utils.KDataset import *
from torch.amp import GradScaler
from pipelines.pipeline_dect import Validate
# from models.generatives.unet_utlis import *

def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--training', action='store_true')
    args.add_argument('--mdn', action='store_true')
    args.add_argument('--log_sig', type=str, default='251211_145058')
    args.add_argument('--load_epoch', type=int, default='500')
    args.add_argument('--save_res', action='store_true')
    args.add_argument('--nepochs', type=int, default=300)
    args.add_argument('--save_freq', type=int, default=20)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--dect_start_late', action='store_true')
    args.add_argument('--dect_start', type=int, default=100)

    args.add_argument('--gen_stop_early', action='store_true')
    args.add_argument('--gen_stop', type=float, default=200)
    args.add_argument('--gen_enable', action='store_true')
    args.add_argument('--model_cfg', type=str, default='ldr')
    args.add_argument('--ldr_pretrained', action='store_true')
    args.add_argument('--gen_pretrained', action='store_true')
    args.add_argument('--ldr_pretrained_log_sig', type=str, default='251207_223958')
    args.add_argument('--ldr_pretrained_epoch', type=str, default=50)
    args.add_argument('--gen_pretrained_log_sig', type=str, default='260211_104838')
    args.add_argument('--gen_pretrained_epoch', type=str, default=200)
    args.add_argument('--eps', type=float, default=0.5)
    args.add_argument('--gt_topk', default=100, type=int)
    args.add_argument('--set', default='train', type=str)
    return args.parse_args()


if __name__ == '__main__':
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg_path = './configs/cfg_rdr_ldr.yml'
    args = arg_parser()
    rand_eps = args.eps
    training = args.training
    
    if args.model_cfg == 'ldr':
        cfg_path = './configs/cfg_rdr_ldr.yml'
    elif args.model_cfg == 'rdr':
        cfg_path = './configs/cfg_rdr_ldr_sps.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)
    model_cfg = args.model_cfg

    x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
    vsize_xyz = cfg.DATASET.roi.voxel_size
    x_size = int(np.floor((x_max-x_min)/vsize_xyz[0]))
    y_size = int(np.floor((y_max-y_min)/vsize_xyz[1]))
    z_size = int(np.floor((z_max-z_min)/vsize_xyz[2]))
    print(f'zyx-size: {z_size}, {y_size}, {x_size}')
    origin = torch.tensor([x_min, y_min, z_min]).to(d)
    vsize_xyz = torch.tensor(vsize_xyz).to(d)
    
    bs=1
    train_kdataset = KRadarDetection_v2_0(cfg=cfg, split='train')
    train_dataloader = DataLoader(train_kdataset, batch_size=bs, 
                                  collate_fn=train_kdataset.collate_fn, num_workers=0, shuffle=False)

    test_kdataset = KRadarDetection_v2_0(cfg=cfg, split='test')
    test_dataloader = DataLoader(test_kdataset, batch_size=bs, 
                            collate_fn=test_kdataset.collate_fn, num_workers=0, shuffle=False)

    rdr_processor = RadarSparseProcessor(cfg)
    ldr_processor = LdrPreprocessor(cfg)
    simplified_pointnet = nn.Linear(4, 32, bias=False).to(d)

    Nvoxels = cfg.DATASET.max_num_voxels
    if args.gen_enable:
        if not args.mdn:
            gen_net = SparseUNet3D(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL).to(d)  
            gen_loss = SynthLocalLoss(w_occ=0.2, w_off=1.0, w_feat=1.0, gt_topk=args.gt_topk)
        else:
            gen_net = SparseUNet3D_MDN(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL).to(d)
            gen_loss = SynthLocalLoss_MDN(w_occ=0.2, w_mdn=1.0, w_int=1.0, gt_topk=args.gt_topk)
        gen_opt = optim.Adam(gen_net.parameters(), lr=1e-3)
    
        if args.gen_pretrained:
            if not args.mdn:
                gen_net = SparseUNet3D(in_ch=20).to(d)  
            else:
                gen_net = SparseUNet3D_MDN(in_ch=20).to(d)
            model_load_ldr = torch.load(f'./logs/exp_{args.gen_pretrained_log_sig}_RTNH/models/epoch{args.gen_pretrained_epoch}.pth')
            gen_net.load_state_dict(state_dict=model_load_ldr['gen_state_dict'])

    else:
        gen_net = None

    dect_net = Rdr2LdrPvrcnnPP(cfg=cfg) if args.model_cfg == 'ldr' else RadarBase(cfg=cfg)
    dect_net = dect_net.to(d)
    if args.ldr_pretrained:
        model_load_ldr = torch.load(f'./logs/exp_{args.ldr_pretrained_log_sig}_RTNH/models/epoch{args.ldr_pretrained_epoch}.pth')
        dect_net.load_state_dict(state_dict=model_load_ldr['dect_state_dict'])
    
        if args.gen_pretrained:
            gen_net.load_state_dict(state_dict=model_load_ldr['gen_state_dict'])

    dect_opt = optim.AdamW(dect_net.parameters(), lr = args.lr, weight_decay=0)
    scaler = GradScaler()
    ppl = Validate(cfg=cfg, gen_net=gen_net, dect_net=dect_net, spatial_size=[z_size, y_size, x_size], model_cfg=args.model_cfg, mdn=args.mdn)
    ppl.set_validate()
    log_path = ppl.path_log
    save_model_path = os.path.join(log_path, 'models')
    os.makedirs(save_model_path, exist_ok=True)
    scheduler = CosineAnnealingLR(dect_opt, T_max=args.nepochs)

    n_epochs = args.nepochs
    save_freq = args.save_freq
    mseloss = nn.MSELoss(reduction='mean')
    
    loss_gen_curve = []
    loss_dect_curve = []
    log_sig = args.log_sig
    epoch = args.load_epoch
    model_load = torch.load(f'./logs/exp_{log_sig}_RTNH/models/epoch{epoch}.pth')
    if args.gen_enable:
        gen_net.load_state_dict(state_dict=model_load['gen_state_dict'])

    # model_load_ldr = torch.load(f'./logs/exp_251119_133450_RTNH/models/epoch30.pth')
    dect_net.load_state_dict(state_dict=model_load['dect_state_dict'])
    dect_net.eval()
    dect_net.model_cfg.POST_PROCESSING = cfg.MODEL.POST_PROCESSING
    dect_net.roi_head.model_cfg.NMS_CONFIG = cfg.MODEL.ROI_HEAD.NMS_CONFIG
    print(f'dect_net.training: {dect_net.training}')
    print(f"dect_loss: {model_load['loss_dect']}")
    dl = test_dataloader if args.set == 'test' else train_dataloader
    for ss in np.arange(1, 11, 1):
        for dx in np.arange(0, 100, ss):
            for dy in np.arange(0, 100, ss):
                for dz in np.arange(0, 10, ss):
                    print(f'\n################Step size: {ss}; Delta offsets (xyz): ({dx}, {dy}, {dz})################')
                    ppl.validate_kitti_conditional(-1, list_conf_thr=ppl.list_val_conf_thr, 
                                    data_loader=dl, save_res=args.save_res, delta_off_xyz=torch.tensor([dx, dy, dz]).to(d))
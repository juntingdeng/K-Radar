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
from torch.utils.tensorboard import SummaryWriter
from pipelines.pipeline_dect import Validate
# from models.generatives.unet_utlis import *

def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--training', action='store_true')
    args.add_argument('--mdn', action='store_true')
    args.add_argument('--log_sig', type=str, default='')
    args.add_argument('--load_epoch', type=int, default='500')
    args.add_argument('--save_res', action='store_true')
    args.add_argument('--nepochs', type=int, default=300)
    args.add_argument('--save_freq', type=int, default=20)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--lr_gen', type=float, default=2e-4)
    args.add_argument('--weight_decay', type=float, default=0.01)
    args.add_argument('--dect_start_late', action='store_true')
    args.add_argument('--dect_start', type=int, default=100)

    args.add_argument('--gen_stop_early', action='store_true')
    args.add_argument('--gen_stop', type=float, default=200)
    args.add_argument('--gen_enable', action='store_true')
    args.add_argument('--model_cfg', type=str, default='ldr')
    args.add_argument('--ldr_pretrained', action='store_true')
    args.add_argument('--gen_pretrained', action='store_true')
    args.add_argument('--ldr_pretrained_log_sig', type=str, default='')
    args.add_argument('--ldr_pretrained_epoch', type=str, default=50)
    args.add_argument('--gen_pretrained_log_sig', type=str, default='')
    args.add_argument('--gen_pretrained_epoch', type=str, default=200)
    args.add_argument('--eps', type=float, default=0.5)
    args.add_argument('--gt_topk', default=100, type=int)
    args.add_argument('--k_stab', default=1, type=int)
    args.add_argument('--search_radius', default=5.0, type=float)
    args.add_argument('--log_sig_max_start', default=1.0, type=float)
    args.add_argument('--log_sig_max_end', default=-0.5, type=float)
    args.add_argument('--log_sig_anneal_epochs', default=50, type=int)
    args.add_argument('--w_mdn', default=0.3, type=float)
    args.add_argument('--w_stab_start', default=0.0, type=float)
    args.add_argument('--w_stab_end', default=2.0, type=float)
    args.add_argument('--w_stab_anneal_epochs', default=50, type=int)
    args.add_argument('--grad_clip_gen', default=5.0, type=float)
    args.add_argument('--grad_clip_dect', default=5.0, type=float)
    args.add_argument('--set', default='train', type=str)
    args.add_argument('--seqs', nargs='+', default=None,
                       help='K-Radar sequence IDs to train/eval on, e.g. --seqs 1 2 5. '
                            'Overrides cfg.DATASET.path_data.list_seq (default [\'1\']) if given.')
    args.add_argument('--num_workers', default=8, type=int,
                       help='DataLoader workers, so point-cloud file I/O for the next frame '
                            'overlaps with the current frame\'s CPU-bound voxelization/matching '
                            '(was hardcoded to 0, fully serializing data loading with GPU compute).')
    return args.parse_args()


if __name__ == '__main__':
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg_path = './configs/cfg_rdr_ldr.yml'
    args = arg_parser()
    args_dict = vars(args)
    args_written = ''
    for key, val in args_dict.items():
        args_written += f'{key}: {val}\n'
    rand_eps = args.eps
    training = args.training
    
    if args.model_cfg == 'ldr':
        cfg_path = './configs/cfg_rdr_ldr.yml'
    elif args.model_cfg == 'rdr':
        cfg_path = './configs/cfg_rdr_ldr_sps.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)
    if args.seqs is not None:
        cfg.DATASET.path_data.list_seq = [str(s) for s in args.seqs]
    model_cfg = args.model_cfg

    x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
    vsize_xyz = cfg.DATASET.roi.voxel_size
    x_size = int(round((x_max-x_min)/vsize_xyz[0]))
    y_size = int(round((y_max-y_min)/vsize_xyz[1]))
    z_size = int(round((z_max-z_min)/vsize_xyz[2]))
    print(f'zyx-size: {z_size}, {y_size}, {x_size}')
    origin = torch.tensor([x_min, y_min, z_min]).to(d)
    vsize_xyz = torch.tensor(vsize_xyz).to(d)
    
    bs=1
    train_kdataset = KRadarDetection_v2_0(cfg=cfg, split='train')
    train_dataloader = DataLoader(train_kdataset, batch_size=bs,
                                  collate_fn=train_kdataset.collate_fn, num_workers=args.num_workers, shuffle=True)

    test_kdataset = KRadarDetection_v2_0(cfg=cfg, split='test')
    test_dataloader = DataLoader(test_kdataset, batch_size=bs,
                            collate_fn=test_kdataset.collate_fn, num_workers=args.num_workers, shuffle=False)

    rdr_processor = RadarSparseProcessor(cfg)
    ldr_processor = LdrPreprocessor(cfg)
    simplified_pointnet = nn.Linear(4, 32, bias=False).to(d)

    Nvoxels = cfg.DATASET.max_num_voxels
    if args.gen_enable:
        if not args.mdn:
            gen_net = SparseUNet3D(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL).to(d)  
            gen_loss = SynthLocalLoss(w_occ=0.2, w_off=1.0, w_feat=1.0, gt_topk=args.gt_topk)
        else:
            gen_net = SparseUNet3D_MDN(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL, t_max=torch.tensor([5, 5, 2])).to(d)
            gen_loss = SynthLocalLoss_MDN(w_occ=1.0, w_mdn=args.w_mdn, w_int=1.0, w_stab=args.w_stab_start, gt_topk=args.gt_topk, k_stab=args.k_stab, t_max=torch.tensor([5, 5, 2]), voxel_size=vsize_xyz, origin=origin, R=args.search_radius)
        gen_opt = optim.Adam(gen_net.parameters(), lr=args.lr_gen)
    
        if args.gen_pretrained:
            if not args.mdn:
                gen_net = SparseUNet3D(in_ch=20).to(d)  
            else:
                gen_net = SparseUNet3D_MDN(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL).to(d)
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

    dect_opt = optim.AdamW(dect_net.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    ppl = Validate(cfg=cfg, gen_net=gen_net, dect_net=dect_net, spatial_size=[z_size, y_size, x_size], model_cfg=args.model_cfg, mdn=args.mdn)
    ppl.set_validate()
    log_path = ppl.path_log
    save_model_path = os.path.join(log_path, 'models')
    os.makedirs(save_model_path, exist_ok=True)
    with open(os.path.join(log_path, 'args.txt'), 'w') as f:
        f.write(args_written)
    tb_writer = SummaryWriter(log_dir=os.path.join(log_path, 'tensorboard'))

    def log_eval_summary(dict_summary, tag, ei):
        # dict_summary: {conf_thr: {cls_name: {'bev':.., '3d':.., 'recall_bev':.., 'recall_3d':..,
        #   'f1_bev':.., 'f1_3d':.., 'num_frames':.., 'num_gt_obj':.., 'num_dt_obj':..}}}
        # from Validate.validate_kitti_conditional's 'all' condition. Logs a per-epoch curve
        # per class (eval_train/* or eval_test/*) plus a mean-over-classes curve, instead of
        # mAP only ever existing as printed text / complete_results.txt. Recall/F1 ride along
        # AP for the same reason they're in complete_results.txt: AP alone floors out at a
        # fixed 1/11 (~9.09%) whenever only the recall=0 sample point is hit, which reads as
        # "9% performance" regardless of whether the detector found nothing or almost
        # everything at low precision -- Recall/F1 disambiguate that.
        if not dict_summary:
            return
        for conf_thr, dict_cls in dict_summary.items():
            bevs, threeds, recs_bev, recs_3d, f1s_bev, f1s_3d = [], [], [], [], [], []
            for cls_name, metrics in dict_cls.items():
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_bev', metrics['bev'], ei)
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_3d', metrics['3d'], ei)
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_recall_bev', metrics['recall_bev'], ei)
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_recall_3d', metrics['recall_3d'], ei)
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_f1_bev', metrics['f1_bev'], ei)
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_f1_3d', metrics['f1_3d'], ei)
                tb_writer.add_scalar(f'eval_{tag}/{cls_name}_num_gt_obj', metrics['num_gt_obj'], ei)
                bevs.append(metrics['bev'])
                threeds.append(metrics['3d'])
                recs_bev.append(metrics['recall_bev'])
                recs_3d.append(metrics['recall_3d'])
                f1s_bev.append(metrics['f1_bev'])
                f1s_3d.append(metrics['f1_3d'])
            if bevs:
                tb_writer.add_scalar(f'eval_{tag}/mean_bev', sum(bevs) / len(bevs), ei)
                tb_writer.add_scalar(f'eval_{tag}/mean_3d', sum(threeds) / len(threeds), ei)
                tb_writer.add_scalar(f'eval_{tag}/mean_recall_bev', sum(recs_bev) / len(recs_bev), ei)
                tb_writer.add_scalar(f'eval_{tag}/mean_recall_3d', sum(recs_3d) / len(recs_3d), ei)
                tb_writer.add_scalar(f'eval_{tag}/mean_f1_bev', sum(f1s_bev) / len(f1s_bev), ei)
                tb_writer.add_scalar(f'eval_{tag}/mean_f1_3d', sum(f1s_3d) / len(f1s_3d), ei)
                tb_writer.add_scalar(f'eval_{tag}/num_frames', next(iter(dict_cls.values()))['num_frames'], ei)

    scheduler = CosineAnnealingLR(dect_opt, T_max=args.nepochs)
    # decoupled from dect_opt's LR/schedule: the MDN offset head's loss landscape narrows
    # as log_sig_max anneals down (Eq. 8's curvature w.r.t. mu scales like 1/sigma^2), so
    # gen_opt needs its own (lower, independently-decaying) LR rather than sharing dect_opt's,
    # or mu's optimization can overshoot/oscillate once the landscape gets narrow enough.
    scheduler_gen = CosineAnnealingLR(gen_opt, T_max=args.nepochs) if args.gen_enable else None
    n_epochs = args.nepochs
    save_freq = args.save_freq
    mseloss = nn.MSELoss(reduction='mean')
    
    loss_gen_curve = []
    loss_dect_curve = []
    if not training:
        log_sig = args.log_sig
        epoch = args.load_epoch
        model_load = torch.load(f'./logs/exp_{log_sig}_RTNH/models/epoch{epoch}.pth')
        if args.gen_enable:
            gen_net.load_state_dict(state_dict=model_load['gen_state_dict'])
            # was missing: gen_net defaults to train() mode after construction, so
            # without this its BatchNorm1d layers would normalize on each eval frame's
            # own batch statistics (and mutate their running stats as a side effect)
            # instead of using the trained running stats, inconsistent with dect_net
            # correctly being in eval mode below.
            gen_net.eval()

        # model_load_ldr = torch.load(f'./logs/exp_251119_133450_RTNH/models/epoch30.pth')
        dect_net.load_state_dict(state_dict=model_load['dect_state_dict'])
        dect_net.eval()
        dect_net.model_cfg.POST_PROCESSING = cfg.MODEL.POST_PROCESSING
        dect_net.roi_head.model_cfg.NMS_CONFIG = cfg.MODEL.ROI_HEAD.NMS_CONFIG
        print(f'dect_net.training: {dect_net.training}')
        print(f"/////dect_loss: {model_load['loss_dect']}")
        dl = test_dataloader if args.set == 'test' else train_dataloader
        summary = ppl.validate_kitti_conditional(-1, list_conf_thr=ppl.list_val_conf_thr, data_loader=dl, save_res=args.save_res, is_subset=False, split_name=args.set)
        log_eval_summary(summary, args.set, 0)

    else:
        global_step = 0
        for ei in range(n_epochs):
            # ramp the probability of feeding generated (vs. real) radar data to the detector
            # from 0 up to rand_eps (--eps) as training progresses, so the detector relies
            # more on the generator's output only once it has had time to learn.
            rand_eps_ei = rand_eps * (1 - np.exp(-ei / 10))
            if args.gen_enable:
                running_loss_gen = 0
                gen_net.train()

                if args.mdn:
                    # anneal the log_sig_off ceiling from a loose start down to a tight
                    # target over log_sig_anneal_epochs, instead of clamping tight from
                    # step 0 -- gives mu time to improve incrementally as the ceiling
                    # tightens, rather than facing the harder (tightly-clamped) loss
                    # landscape all at once.
                    anneal_frac = max(0.0, 1.0 - ei / args.log_sig_anneal_epochs)
                    gen_net.log_sig_max = args.log_sig_max_end + (args.log_sig_max_start - args.log_sig_max_end) * anneal_frac
                    tb_writer.add_scalar('epoch/log_sig_max', gen_net.log_sig_max, ei)

                    # anneal w_stab up from 0, mirroring log_sig_max's ramp down -- lets
                    # mdn_nll get first priority on improving mu before stab_loss starts
                    # competing for gradient budget against int_loss/occ_loss, instead of
                    # both objectives fighting for priority from step 0 while mu is still bad.
                    anneal_frac_wstab = max(0.0, 1.0 - ei / args.w_stab_anneal_epochs)
                    gen_loss.w_stab = args.w_stab_end + (args.w_stab_start - args.w_stab_end) * anneal_frac_wstab
                    tb_writer.add_scalar('epoch/w_stab', gen_loss.w_stab, ei)

                if args.gen_stop_early and ei >=args.gen_stop:
                    gen_net.eval()

            running_loss_dect = 0
            dect_net.train()
            
            for bi, batch_dict in enumerate(train_dataloader):
                # print(f'ei:{ei}, bi:{bi}')
                if args.gen_enable:
                    gen_opt.zero_grad()
                    batch_dict = rdr_processor.forward(batch_dict)
                
                dect_opt.zero_grad()
                batch_dict = ldr_processor.forward(batch_dict)

                # print('Here::::::::2 ', batch_dict['voxel_num_points'],  {sum(batch_dict['voxel_num_points'])})
                for key, val in batch_dict.items():
                    if key in ['points', 'voxels', 'voxel_coords', 'voxel_num_points', 'gt_boxes', 'sp_features', 'sp_indices']:
                        if isinstance(val, np.ndarray):
                            batch_dict[key] = torch.tensor(val).to(device)
                        elif isinstance(val, torch.Tensor) and val.device != device:
                            batch_dict[key] = batch_dict[key].to(device)

                if args.gen_enable:
                    rdr_data = batch_dict['sp_features']
                    # print(f"sp_features:{batch_dict['sp_features'].shape}")
                    if rdr_data.shape[0] < Nvoxels:
                        n = rdr_data.shape[0]
                        while n < Nvoxels:
                            rdr_data = torch.vstack([rdr_data, rdr_data[ :Nvoxels - n]])
                            batch_dict['sp_indices'] = torch.vstack([batch_dict['sp_indices'], batch_dict['sp_indices'][: Nvoxels- n]])
                            n = rdr_data.shape[0]
                        
                        batch_dict['sp_features'] = rdr_data
                        #bzyx

                ldr_data = batch_dict['voxels']
                lmin, lmax = ldr_data.min(), ldr_data.max()
                if ldr_data.shape[0] < Nvoxels:
                    n = ldr_data.shape[0]
                    while n < Nvoxels:
                        ldr_data = torch.vstack([ldr_data, ldr_data[: Nvoxels - n]])
                        batch_dict['voxels'] = ldr_data
                        #bzyx
                        batch_dict['voxel_coords'] = torch.vstack([batch_dict['voxel_coords'], batch_dict['voxel_coords'][: Nvoxels- n]])
                        batch_dict['voxel_num_points'] = torch.concat([batch_dict['voxel_num_points'], batch_dict['voxel_num_points'][: Nvoxels - n]])
                        n = ldr_data.shape[0]
                    # print('Here::::::::21 ', batch_dict['voxel_num_points'],  {sum(batch_dict['voxel_num_points'])})
                    # print(f"batch_dict['voxels']: {batch_dict['voxels'][:, :, -1]}")
                
                if args.gen_enable:
                    # spconv unet
                    # print('2', ei, bi, batch_dict['sp_features'].shape)
                    radar_st = SparseConvTensor(features=batch_dict['sp_features'].reshape((Nvoxels, -1)), 
                                                indices=batch_dict['sp_indices'].int(), #bzyx
                                                spatial_shape=[z_size, y_size, x_size], 
                                                batch_size=bs)

                    lidar_st = SparseConvTensor(features=batch_dict['voxels'].reshape((Nvoxels, -1)), 
                                                indices=batch_dict['voxel_coords'].int(), #bzyx
                                                spatial_shape=[z_size, y_size, x_size], 
                                                batch_size=bs)


                    # Pseudocode
                    rad_idx = radar_st.indices           # [Nr,4]
                    lid_idx = lidar_st.indices           # [Nl,4]

                    all_idx = torch.cat([rad_idx, lid_idx], dim=0)
                    all_idx = torch.unique(all_idx, dim=0)  # union of occupied voxels
                    union_st = scatter_radar_to_union(radar_st, all_idx, [z_size, y_size, x_size], bs)
                    # print(f'Here1 {union_st.features.shape[0]}, {union_st.indices.shape[0]}')
                    
                    out = gen_net(radar_st)  # SparseConvTensor with logits.features [N_active, K] on same coords as c0
                    # print(f"\nbefore: batch_dict['voxels']: {batch_dict['voxels'][0][0]}; batch_dict['voxel_coords']: {batch_dict['voxel_coords'][0]}")
                    if not args.mdn:
                        pred, occ, attrs = out['st'], out['logits'], out['attrs']
                        loss_gen = gen_loss(occ, attrs, pred, radar_st, lidar_st, R=5, origin=origin, vsize_xyz=vsize_xyz)
                        running_loss_gen += loss_gen.detach().item()
                        offs = attrs[:, :, :3]
                        # print(f'offs: {offs}, ints: {ints}')

                        voxel_center_xyz = origin + (torch.flip(pred.indices[:, 1:4].float(), dims=[1]) + 0.5) * vsize_xyz  # grid center
                        pred_offset_m = offs * vsize_xyz.to(d)  # scale voxel-units → meters
                        voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 5, 1)
                        # print(voxel_center_xyz.shape, pred_offset_m.shape)
                        attrs = torch.cat([voxel_center_xyz + pred_offset_m, attrs[:, :, 3:4]], dim=-1)

                        _pred_indices = pred.indices#.detach()
                        _attrs = attrs#.detach() # xyz
                        if (torch.isnan(_attrs)).any():
                            print(f'_attrs has nan')

                        # select valid slots by probability
                        prob_thresh=0.0
                        probs = torch.sigmoid(occ)                 # [N,K,1]
                        keep = (probs >= prob_thresh)
                        voxel_num_points = keep.sum(dim=1) #[N, ]
                        keep = keep.repeat(1,1,4) 
                        # print(f'keep:{keep.shape}, _attrs:{_attrs.shape}')
                        # print(f"batch_dict['voxel_num_points']: {voxel_num_points}")

                        _attrs = torch.where(keep, _attrs, torch.zeros_like(_attrs))

                        if model_cfg == 'ldr':
                            if random.random() < rand_eps_ei:
                                if _attrs.shape[0] < Nvoxels:
                                    batch_dict['voxels'] = _attrs.contiguous().float().to(d)
                                    batch_dict['voxel_coords'][:, 1] = _pred_indices[:, 1].int().clamp(1, z_size-1)
                                    batch_dict['voxel_coords'][:, 2] = _pred_indices[:, 2].int().clamp(1, y_size-1)
                                    batch_dict['voxel_coords'][:, 3] = _pred_indices[:, 3].int().clamp(1, x_size-1)
                                    batch_dict['voxel_coords'] = batch_dict['voxel_coords'].to(d)
                                    batch_dict['voxel_num_points'] = voxel_num_points
                                else:
                                    _, topN = torch.topk(_attrs[:, :, -1].mean(1), k=Nvoxels)
                                    batch_dict['voxels'] = _attrs.contiguous().float().to(d)[topN]
                                    batch_dict['voxel_coords'][:, 1] = _pred_indices[:, 1].int().clamp(1, z_size-1)[topN]
                                    batch_dict['voxel_coords'][:, 2] = _pred_indices[:, 2].int().clamp(1, y_size-1)[topN]
                                    batch_dict['voxel_coords'][:, 3] = _pred_indices[:, 3].int().clamp(1, x_size-1)[topN]
                                    batch_dict['voxel_coords'] = batch_dict['voxel_coords'].to(d)
                                    batch_dict['voxel_num_points'] = voxel_num_points[topN]
                        
                        else:
                            if _attrs.shape[0] < Nvoxels:
                                batch_dict['sp_features'] = _attrs.contiguous().float().to(d).mean(dim=1, keepdim=False)
                                batch_dict['sp_indices'][:, 1] = _pred_indices[:, 1].int().clamp(1, z_size-1)
                                batch_dict['sp_indices'][:, 2] = _pred_indices[:, 2].int().clamp(1, y_size-1)
                                batch_dict['sp_indices'][:, 3] = _pred_indices[:, 3].int().clamp(1, x_size-1)
                                batch_dict['sp_indices'] = batch_dict['sp_indices'].to(d)
                                # batch_dict['voxel_num_points'] = voxel_num_points
                            else:
                                _, topN = torch.topk(_attrs[:, :, -1].mean(1), k=Nvoxels)
                                batch_dict['sp_features'] = _attrs.contiguous().float().to(d)[topN].mean(dim=1, keepdim=False)
                                batch_dict['sp_indices'][:, 1] = _pred_indices[:, 1].int().clamp(1, z_size-1)[topN]
                                batch_dict['sp_indices'][:, 2] = _pred_indices[:, 2].int().clamp(1, y_size-1)[topN]
                                batch_dict['sp_indices'][:, 3] = _pred_indices[:, 3].int().clamp(1, x_size-1)[topN]
                                batch_dict['sp_indices'] = batch_dict['sp_indices'].to(d)
                                # batch_dict['voxel_num_points'] = voxel_num_points[topN]
                    else:
                        # loss_gen = gen_loss(out, radar_st, lidar_st)
                        occ_loss, mdn_nll, int_loss, tol_loss, stab_loss = gen_loss(out, radar_st, lidar_st)
                        loss_gen = gen_loss.w_occ * occ_loss + gen_loss.w_mdn * mdn_nll + gen_loss.w_int * int_loss + gen_loss.w_stab * stab_loss #+ 0.2*tol_loss
                        running_loss_gen += loss_gen.detach().item()
                        if bi == 0 and ei % 2 == 0:
                            print(f"  [raw, unweighted] occ_loss:{occ_loss.item():.4f}, mdn_nll:{mdn_nll.item():.4f}, int_loss:{int_loss.item():.4f}, tol_loss:{tol_loss.item():.4f}, stab_loss:{stab_loss.item():.4f}")
                        tb_writer.add_scalar('batch/occ_loss', occ_loss.detach().item(), global_step)
                        tb_writer.add_scalar('batch/mdn_nll', mdn_nll.detach().item(), global_step)
                        tb_writer.add_scalar('batch/int_loss', int_loss.detach().item(), global_step)
                        tb_writer.add_scalar('batch/stab_loss', stab_loss.detach().item(), global_step)
                        if bi == 0:
                            tb_writer.add_histogram('dist/log_sig_off', out['log_sig_off'].detach(), ei)
                            if getattr(gen_loss, 'last_p_stab', None) is not None and gen_loss.last_p_stab.numel() > 0:
                                tb_writer.add_histogram('dist/p_stab', gen_loss.last_p_stab, ei)

                        # matched, gt_d, gt_f, gt_coords = local_match_closest(radar_st, lidar_st, gt_topk=args.gt_topk) if not args.mdn else local_match_closest_mdn(radar_st, lidar_st, gt_topk=args.gt_topk)
                        # # gt_d: zyx
                        # out['mu_off'] = torch.flip(gt_d, dims=[-1])
                        # # print(f"out['mu_off']: {out['mu_off']}")
                        # print(f"out['mu_off']-x: {out['mu_off'][:, :, 0].min()} ~ {out['mu_off'][:, :, 0].max()}")
                        # print(f"out['mu_off']-y: {out['mu_off'][:, :, 1].min()} ~ {out['mu_off'][:, :, 1].max()}")
                        # print(f"out['mu_off']-z: {out['mu_off'][:, :, 2].min()} ~ {out['mu_off'][:, :, 2].max()}")
                        attrs_pts, voxel_coords, voxel_num_points, chosen_k, probk, mu = sample_points_from_mdn(
                                                                                        pred_st=out['st'],
                                                                                        mu_off=out["mu_off"],
                                                                                        log_sig_off=out["log_sig_off"],
                                                                                        mix_logit=out["mix_logit"],
                                                                                        mu_int=out["mu_int"],
                                                                                        origin=origin,
                                                                                        vsize_xyz=vsize_xyz,
                                                                                        n_points_per_voxel=cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
                                                                                        prob_thresh=0.05,       # tune: 0.0 ~ 0.2
                                                                                        sample_mode="mixture",  # or "top1" for deterministic
                                                                                        clamp_intensity=(0.0, None),
                                                                                    )
                        if (torch.isnan(attrs_pts)).any():
                            print(f'attrs_pts has nan')

                        voxel_coords[:, 1:4] += torch.flip(mu.int(), dims=[1]) 
                        batch_dict["voxels"] = attrs_pts.float()
                        batch_dict["voxel_coords"] = voxel_coords
                        batch_dict["voxel_num_points"] = voxel_num_points
                        # print(f"after: batch_dict['voxels']: {batch_dict['voxels'][0][0]}; batch_dict['voxel_coords']: {batch_dict['voxel_coords'][0]}")
                        
                
                # print(f"Here--------: {batch_dict['voxels'].shape[0]}, {batch_dict['voxel_num_points'].shape[0]}")
                # print(f"batch_dict['voxels']: {batch_dict['voxels'].shape}, batch_dict['voxel_coords']: {batch_dict['voxel_coords'].shape}")
                if not args.dect_start_late or (args.dect_start_late and ei >= args.dect_start):
                    
                    dect_output = dect_net(batch_dict)
                    loss_dect = dect_net.head.loss(dect_output) if args.model_cfg == 'rdr' else dect_net.loss(dect_output)
                    # loss_dect.backward()
                    # dect_opt.step()
                    running_loss_dect += loss_dect.detach().item()
                    # loss_total = loss_gen + 0.5 * (1.0 - np.cos(np.pi * (ei/n_epochs)))*loss_dect
                    
                    loss_total = loss_dect if not args.gen_enable or (args.gen_stop_early and ei >= args.gen_stop) else loss_gen + loss_dect
                    loss_total.backward()
                    # caps the step size when the loss landscape gets steep (e.g. as
                    # log_sig_max anneals down and narrows the MDN's NLL basin), instead of
                    # relying solely on lr_gen being exactly right to avoid overshoot
                    torch.nn.utils.clip_grad_norm_(dect_net.parameters(), max_norm=args.grad_clip_dect)
                    if args.gen_enable:
                        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), max_norm=args.grad_clip_gen)
                    if not args.gen_enable or (args.gen_stop_early and ei >= args.gen_stop):
                        dect_opt.step()
                    else:
                        dect_opt.step()
                        gen_opt.step()
                else:
                    loss_dect = torch.tensor(0.)
                    loss_total = loss_gen
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(gen_net.parameters(), max_norm=args.grad_clip_gen)
                    gen_opt.step()

                tb_writer.add_scalar('batch/loss_total', loss_total.detach().item(), global_step)
                tb_writer.add_scalar('batch/loss_dect', loss_dect.detach().item(), global_step)
                if args.gen_enable:
                    tb_writer.add_scalar('batch/loss_gen', loss_gen.detach().item(), global_step)
                tb_writer.add_scalar('batch/lr', scheduler.get_last_lr()[0], global_step)
                if args.gen_enable:
                    tb_writer.add_scalar('batch/lr_gen', scheduler_gen.get_last_lr()[0], global_step)
                global_step += 1

                # for key, val in batch_dict.items():
                #     if isinstance(val, torch.Tensor):
                #         batch_dict[key] = batch_dict[key].to('cpu')

                if 'pointer' in batch_dict.keys():
                    for dict_item in batch_dict['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None
                for temp_key in batch_dict.keys():
                    batch_dict[temp_key] = None

            # CosineAnnealingLR's T_max is in epochs (T_max=args.nepochs), so step it once
            # per epoch here, not per batch above (stepping ~100x/epoch made the schedule
            # complete a full anneal every ~3 epochs and then oscillate for the rest of training).
            scheduler.step()
            if args.gen_enable:
                scheduler_gen.step()

            if args.gen_enable:
                loss_gen_curve.append(running_loss_gen/(max(1, len(train_dataloader))))
                loss_dect_curve.append(running_loss_dect/(max(1, len(train_dataloader))))
                tb_writer.add_scalar('epoch/loss_gen', loss_gen_curve[-1], ei)
                tb_writer.add_scalar('epoch/loss_dect', loss_dect_curve[-1], ei)
                tb_writer.add_scalar('epoch/rand_eps', rand_eps_ei, ei)
                tb_writer.add_scalar('epoch/lr_gen', scheduler_gen.get_last_lr()[0], ei)
                if (ei+1) % save_freq == 0:
                    dict_util = {
                        'epoch': ei+1,
                        'gen_state_dict': gen_net.state_dict(),
                        'dect_state_dict': dect_net.state_dict(),
                        'gen_opt_state_dict': gen_opt.state_dict(),
                        'dect_opt_state_dict': dect_opt.state_dict(),
                        'lr': scheduler.get_last_lr(),
                        'lr_gen': scheduler_gen.get_last_lr(),
                        'loss_gen': loss_gen.detach().item(),
                        'loss_dect': loss_dect.detach().item()
                    }

                    torch.save(dict_util, os.path.join(save_model_path, f'epoch{ei+1}.pth'))
            
            else:
                loss_dect_curve.append(running_loss_dect/(max(1, len(train_dataloader))))
                tb_writer.add_scalar('epoch/loss_dect', loss_dect_curve[-1], ei)
                if (ei+1) % args.save_freq == 0:
                    dict_util = {
                        'epoch': ei+1,
                        # 'gen_state_dict': gen_net.state_dict(),
                        'dect_state_dict': dect_net.state_dict(),
                        # 'gen_opt_state_dict': gen_opt.state_dict(),
                        'dect_opt_state_dict': dect_opt.state_dict(),
                        'lr': args.lr,
                        'loss_dect': loss_dect.detach().item()
                    }

                    torch.save(dict_util, os.path.join(save_model_path, f'epoch{ei+1}.pth'))
            
            if ei%2 == 0:

                if args.gen_enable:
                    print(f'epoch:{ei}, rand_eps_ei:{rand_eps_ei}, loss_gen:{loss_gen.detach().item():.4f}, loss_dect:{loss_dect.detach().item():.4f}, loss_total:{loss_total.detach().item():.4f}')
                else:
                    print(f'epoch:{ei}, loss_dect:{loss_dect.detach().item():.4f}')

            # periodic validation, following the same schedule convention as
            # pipelines/pipeline_detection_v1_0.py (cfg.VAL.VAL_PER_EPOCH_SUBSET/FULL)
            ran_final_full_validation = False
            if ppl.is_validate:
                if ppl.is_consider_subset:
                    if (ei + 1) % ppl.val_per_epoch_subset == 0:
                        summary = ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=train_dataloader, is_subset=True, split_name='train')
                        log_eval_summary(summary, 'train', ei)
                        # also track on the held-out test split, on the same cadence, so the
                        # eval curve isn't only ever measuring train-set fit
                        summary = ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=test_dataloader, is_subset=True, split_name='test')
                        log_eval_summary(summary, 'test', ei)
                if (ei + 1) % ppl.val_per_epoch_full == 0:
                    summary = ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=train_dataloader, split_name='train')
                    log_eval_summary(summary, 'train', ei)
                    ran_final_full_validation = (ei == n_epochs - 1)

            tb_writer.flush()


        # skip if the periodic full-validation above already covered this exact epoch
        # (n_epochs a multiple of VAL_PER_EPOCH_FULL) -- avoids re-running the identical
        # full-dataset pass twice back to back
        if not ran_final_full_validation:
            summary = ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=train_dataloader, split_name='train')
            log_eval_summary(summary, 'train', ei)
        # train_dataloader above only measures train-set fit; also report true held-out
        # generalization on the test split at the end of training
        summary = ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=test_dataloader, split_name='test')
        log_eval_summary(summary, 'test', ei)
        if args.gen_enable:
            plt.plot(loss_gen_curve, label='gen-loss')
        # plt.plot(loss_gen_curve, label='dect-loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(ppl.path_log, 'loss.png'))
        tb_writer.close()
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

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus
from models.skeletons.rdr_base import RadarBase
from models.generatives.unet import *

from dataset_utils.KDataset import *
from torch.amp import GradScaler
from pipelines.pipeline_dect import Validate
# from models.generatives.unet_utlis import *

def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--training', action='store_true')
    args.add_argument('--log_sig', type=str, default='251119_142454')
    args.add_argument('--load_epoch', type=int, default='30')
    args.add_argument('--nepochs', type=int, default=200)
    args.add_argument('--save_freq', type=int, default=20)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--gen_stop', type=float, default=200)
    args.add_argument('--gen_enable', action='store_true')
    args.add_argument('--model_cfg', type=str, default='ldr')
    return args.parse_args()


if __name__ == '__main__':
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg_path = './configs/cfg_rdr_ldr.yml'
    args = arg_parser()
    training = args.training
    
    if args.model_cfg == 'ldr':
        cfg_path = './configs/cfg_rdr_ldr.yml'
    elif args.model_cfg == 'rdr':
        cfg_path = './configs/cfg_rdr_ldr_sps.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)
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
                                  collate_fn=train_kdataset.collate_fn, num_workers=4, shuffle=True)

    test_kdataset = KRadarDetection_v2_0(cfg=cfg, split='test')
    test_dataloader = DataLoader(test_kdataset, batch_size=bs, 
                            collate_fn=test_kdataset.collate_fn, num_workers=4, shuffle=False)

    rdr_processor = RadarSparseProcessor(cfg)
    ldr_processor = LdrPreprocessor(cfg)
    simplified_pointnet = nn.Linear(4, 32, bias=False).to(d)

    Nvoxels = cfg.DATASET.max_num_voxels
    if args.gen_enable:
        gen_net = SparseUNet3D(in_ch=20).to(d)
        gen_loss = SynthLocalLoss(w_occ=0.2, w_off=1.0, w_feat=1.0)
        gen_opt = optim.SGD(gen_net.parameters(), lr=args.lr)
    else:
        gen_net = None

    dect_net = Rdr2LdrPvrcnnPP(cfg=cfg) if args.model_cfg == 'ldr' else RadarBase(cfg=cfg)
    dect_net = dect_net.to(d)
    dect_opt = optim.AdamW(dect_net.parameters(), lr = args.lr, weight_decay=0.)
    scaler = GradScaler()
    ppl = Validate(cfg=cfg, gen_net=gen_net, dect_net=dect_net, spatial_size=[z_size, y_size, x_size], model_cfg=args.model_cfg)
    ppl.set_validate()
    log_path = ppl.path_log
    save_model_path = os.path.join(log_path, 'models')
    os.makedirs(save_model_path, exist_ok=True)

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

        # model_load_ldr = torch.load(f'./logs/exp_251119_133450_RTNH/models/epoch30.pth')
        dect_net.load_state_dict(state_dict=model_load['dect_state_dict'])
        ppl.validate_kitti_conditional(-1, list_conf_thr=ppl.list_val_conf_thr, data_loader=train_dataloader)

    else:
        for ei in range(n_epochs):
            if args.gen_enable:
                running_loss_gen = 0
                gen_net.train()

                if ei >=args.gen_stop:
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
                    
                    out = gen_net(union_st)  # SparseConvTensor with logits.features [N_active, K] on same coords as c0

                    pred, occ, attrs = out['st'], out['logits'], out['attrs']
                    loss_gen = gen_loss(occ, attrs, pred, union_st, lidar_st, R=5, origin=origin, vsize_xyz=vsize_xyz)
                    offs = attrs[:, :, :3]
                    # print(f'offs: {offs}, ints: {ints}')

                    voxel_center_xyz = origin + (torch.flip(union_st.indices[:, 1:4].float(), dims=[1]) + 0.5) * vsize_xyz  # grid center
                    pred_offset_m = offs * vsize_xyz.to(d)  # scale voxel-units → meters
                    voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 5, 1)
                    # print(voxel_center_xyz.shape, pred_offset_m.shape)
                    attrs = torch.cat([voxel_center_xyz + pred_offset_m, attrs[:, :, 3:4]], dim=-1)

                    _pred_indices = pred.indices.detach()
                    _attrs = attrs.detach() # xyz
                    if (torch.isnan(_attrs)).any():
                        print(f'_attrs has nan')

                    # select valid slots by probability
                    prob_thresh=0.9
                    probs = torch.sigmoid(occ)                 # [N,K,1]
                    keep = (probs >= prob_thresh)
                    voxel_num_points = keep.sum(dim=1) #[N, ]
                    keep = keep.repeat(1,1,4) 
                    # print(f'keep:{keep.shape}, _attrs:{_attrs.shape}')
                    # print(f"batch_dict['voxel_num_points']: {voxel_num_points}")

                    _attrs = torch.where(keep, _attrs, torch.zeros_like(_attrs))

                    if model_cfg == 'ldr':
                        if _attrs.shape[0] < Nvoxels:
                            batch_dict['voxels'] = _attrs.contiguous().float().to(d)
                            batch_dict['voxel_coords'][:, 1] = _pred_indices[:, 0].int().clamp(1, z_size-1)
                            batch_dict['voxel_coords'][:, 2] = _pred_indices[:, 1].int().clamp(1, y_size-1)
                            batch_dict['voxel_coords'][:, 3] = _pred_indices[:, 2].int().clamp(1, x_size-1)
                            batch_dict['voxel_coords'] = batch_dict['voxel_coords'].to(d)
                            batch_dict['voxel_num_points'] = voxel_num_points
                        else:
                            _, topN = torch.topk(_attrs[:, :, -1].mean(1), k=Nvoxels)
                            batch_dict['voxels'] = _attrs.contiguous().float().to(d)[topN]
                            batch_dict['voxel_coords'][:, 1] = _pred_indices[:, 0].int().clamp(1, z_size-1)[topN]
                            batch_dict['voxel_coords'][:, 2] = _pred_indices[:, 1].int().clamp(1, y_size-1)[topN]
                            batch_dict['voxel_coords'][:, 3] = _pred_indices[:, 2].int().clamp(1, x_size-1)[topN]
                            batch_dict['voxel_coords'] = batch_dict['voxel_coords'].to(d)
                            batch_dict['voxel_num_points'] = voxel_num_points[topN]
                    
                    else:
                        if _attrs.shape[0] < Nvoxels:
                            batch_dict['sp_features'] = _attrs.contiguous().float().to(d).mean(dim=1, keepdim=False)
                            batch_dict['sp_indices'][:, 1] = _pred_indices[:, 0].int().clamp(1, z_size-1)
                            batch_dict['sp_indices'][:, 2] = _pred_indices[:, 1].int().clamp(1, y_size-1)
                            batch_dict['sp_indices'][:, 3] = _pred_indices[:, 2].int().clamp(1, x_size-1)
                            batch_dict['sp_indices'] = batch_dict['sp_indices'].to(d)
                            # batch_dict['voxel_num_points'] = voxel_num_points
                        else:
                            _, topN = torch.topk(_attrs[:, :, -1].mean(1), k=Nvoxels)
                            batch_dict['sp_features'] = _attrs.contiguous().float().to(d)[topN].mean(dim=1, keepdim=False)
                            batch_dict['sp_indices'][:, 1] = _pred_indices[:, 0].int().clamp(1, z_size-1)[topN]
                            batch_dict['sp_indices'][:, 2] = _pred_indices[:, 1].int().clamp(1, y_size-1)[topN]
                            batch_dict['sp_indices'][:, 3] = _pred_indices[:, 2].int().clamp(1, x_size-1)[topN]
                            batch_dict['sp_indices'] = batch_dict['sp_indices'].to(d)
                            # batch_dict['voxel_num_points'] = voxel_num_points[topN]
                        
                        # voxel_features = simplified_pointnet(batch_dict['sp_features'])
                        # voxel_features = torch.max(voxel_features, dim=1, keepdim=False)[0]
                        # batch_dict['_sp_features'] = voxel_features
                        
                
                # print(f"Here--------: {batch_dict['voxels'].shape[0]}, {batch_dict['voxel_num_points'].shape[0]}")
                # print(f"batch_dict['voxels']: {batch_dict['voxels'].shape}, batch_dict['voxel_coords']: {batch_dict['voxel_coords'].shape}")
                dect_output = dect_net(batch_dict)
                
                loss_dect = dect_net.head.loss(dect_output) if args.model_cfg == 'rdr' else dect_net.loss(dect_output)
                
                # loss = loss_dect + loss_gen
                if args.gen_enable:
                    if ei < args.gen_stop:
                        loss_gen.backward()
                        gen_opt.step()

                        # for name, param in gen_net.named_parameters():
                        #     if param.grad is not None:
                        #         print('gen_net: ', name, param.grad.mean(), param.grad.abs().max())
                        #     else:
                        #         print(name, "has no grad!")
                        
                        # with torch.no_grad():
                        #     for name, p in gen_net.named_parameters():
                        #         if p.grad is None:
                        #             continue

                        #         grad_norm = p.grad.norm().item()
                        #         weight_norm = p.data.norm().item()
                        #         # assume Adam/AdamW or SGD; use your lr here
                        #         lr = gen_opt.param_groups[0]['lr']

                        #         rel_update = lr * grad_norm / (weight_norm + 1e-12)
                        #         print(f"gen_net: {name:30s} grad_norm={grad_norm:.3e}  "
                        #             f"w_norm={weight_norm:.3e}  "
                        #             f"rel_update={rel_update:.3e}")

                    running_loss_gen += loss_gen.detach().item()

                loss_dect.backward()
                dect_opt.step()
                # for name, param in gen_net.named_parameters():
                #     if param.grad is not None:
                #         print('gen_net: ', name, param.grad.mean(), param.grad.abs().max())
                #     else:
                #         print(name, "has no grad!")
                # scaler.update()
                running_loss_dect += loss_dect.detach().item()
                

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

            if args.gen_enable:
                loss_gen_curve.append(running_loss_gen/(max(1, len(train_dataloader))))
                loss_dect_curve.append(running_loss_dect/(max(1, len(train_dataloader))))
                if (ei < args.gen_stop and (ei+1) % save_freq == 0) or (ei >=args.gen_stop and (ei+1) % save_freq == 0):
                    dict_util = {
                        'epoch': ei+1,
                        'gen_state_dict': gen_net.state_dict(),
                        'dect_state_dict': dect_net.state_dict(),
                        'gen_opt_state_dict': gen_opt.state_dict(),
                        'dect_opt_state_dict': dect_opt.state_dict(),
                        'lr': args.lr
                    }

                    torch.save(dict_util, os.path.join(save_model_path, f'epoch{ei+1}.pth'))
            
            else:
                loss_dect_curve.append(running_loss_dect/(max(1, len(train_dataloader))))
                if (ei+1) % args.save_freq == 0:
                    dict_util = {
                        'epoch': ei+1,
                        # 'gen_state_dict': gen_net.state_dict(),
                        'dect_state_dict': dect_net.state_dict(),
                        # 'gen_opt_state_dict': gen_opt.state_dict(),
                        'dect_opt_state_dict': dect_opt.state_dict(),
                        'lr': args.lr
                    }

                    torch.save(dict_util, os.path.join(save_model_path, f'epoch{ei+1}.pth'))
            
            if ei%2 == 0:
                if args.gen_enable:
                    print(f'epoch:{ei}, loss_gen:{loss_gen_curve[-1]}, loss_dect:{loss_dect_curve[-1]}')
                else:
                    print(f'epoch:{ei}, loss_dect:{loss_dect_curve[-1]}')


        ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=train_dataloader)
        if args.gen_enable:
            plt.plot(loss_gen_curve, label='gen-loss')
        plt.plot(loss_dect_curve, label='dect-loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(ppl.path_log, 'loss.png'))
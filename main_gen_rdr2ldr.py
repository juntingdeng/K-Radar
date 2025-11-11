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

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus
from models.generatives.unet import *

from depthEst.KDataset import *
from torch.amp import GradScaler
from pipelines.pipeline_dect import Validate

if __name__ == '__main__':
    # kdataset = Kdataset(root='data')
    # cfg_path = './K-Radar/configs/cfg_PVRCNNPP.yml'
    cfg_path = './configs/cfg_rdr_ldr.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)

    x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
    vsize_xyz = cfg.DATASET.roi.voxel_size
    x_size = int(round((x_max-x_min)/vsize_xyz[0]))
    y_size = int(round((y_max-y_min)/vsize_xyz[1]))
    z_size = int(round((z_max-z_min)/vsize_xyz[2]))
    print(f'zyx-size: {z_size}, {y_size}, {x_size}')
    
    bs=1
    train_kdataset = KRadarDetection_v2_0(cfg=cfg, split='train')
    train_dataloader = DataLoader(train_kdataset, batch_size=bs, 
                                  collate_fn=train_kdataset.collate_fn, num_workers=4, shuffle=True)

    test_kdataset = KRadarDetection_v2_0(cfg=cfg, split='test')
    test_dataloader = DataLoader(test_kdataset, batch_size=bs, 
                            collate_fn=test_kdataset.collate_fn, num_workers=4, shuffle=False)

    rdr_processor = RadarSparseProcessor(cfg)
    ldr_processor = LdrPreprocessor(cfg)

    Nvoxels = cfg.DATASET.max_num_voxels
    gen_net = nn.Sequential(
        nn.Conv2d(Nvoxels*bs, 1024, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(1024, Nvoxels*bs, kernel_size=1),
        nn.Sigmoid()
    )

    gen_resnet = torchvision.models.resnet50(weights = "DEFAULT")

    dect_net = Rdr2LdrPvrcnnPP(cfg=cfg)

    gen_opt = optim.SGD(gen_net.parameters(), lr=1e-2)
    dect_opt = optim.SGD(dect_net.parameters(), lr = 1e-3)
    scaler = GradScaler()
    ppl = Validate(cfg=cfg, gen_net=gen_net, dect_net=dect_net, vsize=[z_size, y_size, x_size])
    ppl.set_validate()

    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 50
    mseloss = nn.MSELoss(reduction='mean')
    gen_net = gen_net.to(d)
    loss_gen_curve = []
    loss_dect_curve = []
    for ei in range(n_epochs):
        running_loss_gen = 0
        running_loss_dect = 0
        gen_opt.zero_grad()
        dect_opt.zero_grad()
        for bi, batch_dict in enumerate(train_dataloader):
            batch_dict = rdr_processor.forward(batch_dict)
            batch_dict = ldr_processor.forward(batch_dict)

            for key, val in batch_dict.items():
                if key in ['points', 'voxels', 'voxel_coords', 'voxel_num_points', 'gt_boxes', 'sp_features', 'sp_indices']:
                    if isinstance(val, np.ndarray):
                        batch_dict[key] = torch.tensor(val).to(device)
                    elif isinstance(val, torch.Tensor) and val.device != device:
                        batch_dict[key] = batch_dict[key].to(device)

            rdr_data = batch_dict['sp_features']
            ldr_data = batch_dict['voxels']
            lmin, lmax = ldr_data.min(), ldr_data.max()
            
            # if rdr_data.shape[0] < Nvoxels:
            #     rdr_data = torch.vstack([rdr_data, torch.zeros((Nvoxels - rdr_data.shape[0], rdr_data.shape[1], rdr_data.shape[2])).to(d)])
            #     batch_dict['sp_indices'] = torch.vstack([batch_dict['sp_indices'], torch.zeros((Nvoxels - batch_dict['sp_indices'].shape[0], batch_dict['sp_indices'].shape[1])).to(d)])

            # if ldr_data.shape[0] < Nvoxels:
            #     ldr_data = torch.vstack([ldr_data, torch.zeros((Nvoxels - ldr_data.shape[0], ldr_data.shape[1], ldr_data.shape[2])).to(d)])
            #     batch_dict['voxel_coords'] = torch.vstack([batch_dict['voxel_coords'], torch.zeros((Nvoxels - batch_dict['voxel_coords'].shape[0], batch_dict['voxel_coords'].shape[1])).to(d)])

            # input = torch.concatenate([rdr_data, batch_dict['sp_indices'][:, 1:].unsqueeze(1).repeat(1, 5, 1)], dim=-1) #(N, 4+3)
            # gt = torch.concatenate([ldr_data, batch_dict['voxel_coords'][:, 1:].unsqueeze(1).repeat(1, 5, 1)], dim=-1)

            # spconv unet
            x = SparseConvTensor(batch_dict['sp_features'], batch_dict['sp_indices'], [z_size, y_size, x_size], bs)
            model = SparseUNet3D(in_ch=batch_dict['sp_features'].shape[1], base_ch=32, num_classes=K)
            logits = model(x)  # SparseConvTensor with logits.features [N_active, K] on same coords as c0

            output = gen_net(input)
            output = lmin + output*(lmax - lmin)
            if (torch.isnan(output)).any():
                print(f'output has nan')

            loss_gen = mseloss(output, gt)/(Nvoxels)
            scaler.scale(loss_gen).backward()

            # print(f"batch_dict['voxels'] shape, before: {batch_dict['voxels'].shape}, after: {output.shape}")
            _output = output.detach()
            batch_dict['voxels'] = _output[:, :, :4].contiguous().float().to(d)
            batch_dict['voxel_coords'][:, 1] = _output[:, 0, 4].int().clamp(1, z_size-1)
            batch_dict['voxel_coords'][:, 2] = _output[:, 0, 5].int().clamp(1, y_size-1)
            batch_dict['voxel_coords'][:, 3] = _output[:, 0, 6].int().clamp(1, x_size-1)
            batch_dict['voxel_coords'] = batch_dict['voxel_coords'].to(d)
            # print(f"batch_dict['voxels']:{batch_dict['voxels'].shape}, batch_dict['voxel_coords']:{batch_dict['voxel_coords'].shape}")
            
            batch_dict['voxel_num_points'] = torch.tensor(Nvoxels).to(d)
            
            dect_output = dect_net(batch_dict)
            
            loss_dect = dect_net.loss(dect_output)
            
            # loss = loss_dect + loss_gen
            scaler.scale(loss_dect).backward()

            scaler.step(gen_opt)       # safe even if gen_net has no grads
            scaler.step(dect_opt)
            scaler.update()

            running_loss_dect += loss_dect.detach().item()
            running_loss_gen += loss_gen.detach().item()

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

        loss_gen_curve.append(running_loss_gen/(max(1, len(train_dataloader))))
        loss_dect_curve.append(running_loss_dect/(max(1, len(train_dataloader))))

        if ei%2 == 0:
            print(f'epoch:{ei}, loss_gen:{loss_gen_curve[-1]}, loss_dect:{loss_dect_curve[-1]}')

    ppl.validate_kitti_conditional(ei, list_conf_thr=ppl.list_val_conf_thr, data_loader=test_dataloader)
    plt.plot(loss_gen_curve, label='gen-loss')
    plt.plot(loss_dect_curve, label='dect-loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(ppl.path_log, 'loss.png'))
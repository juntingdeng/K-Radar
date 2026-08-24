import numpy as np
import os
import sys
import collections
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
# sys.path.append('./K-Radar')
# sys.path.append('./K-Radar/models')

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import cumm.tensorview as tv

class VoxelWrapper:
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        
        self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )
    
    def generate(self, points):
        voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class LdrPreprocessor:
    def __init__(self, cfg):
        self.dataset_cfg = cfg.DATASET
        self.model_cfg = cfg.MODEL
        self.training = True
        self.split = 'train' if self.training else 'test'
        # Shuffle points before voxelization so the max_num_voxels cap (spconv keeps
        # voxels in point-scan order) doesn't always drop the same *spatial region* of
        # the point cloud -- see diagnose_seq_matches.py's per-seq empty-match rates.
        self.shuffle_points = True

        self.vsize_xyz=self.dataset_cfg.roi.voxel_size
        self.coors_range_xyz=np.array(self.dataset_cfg.roi.xyz)
        self.num_point_features= 4 #self.dataset_cfg.ldr64.n_used

        self.voxel_generator = VoxelWrapper(
            vsize_xyz=self.vsize_xyz,
            coors_range_xyz=self.coors_range_xyz,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel= self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=cfg.DATASET.max_num_voxels #self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS[self.split]
            )
        
    def forward(self, batch_dict):
        try:
            device = batch_dict.device
        except:
            device = 'cpu'
        batched_ldr64 = batch_dict['ldr64']
        batched_indices_ldr64 = batch_dict['batch_indices_ldr64']
        list_points = []
        list_voxels = []
        list_voxel_coords = []
        list_voxel_num_points = []
        for batch_idx in range(batch_dict['batch_size']):
            temp_points = batched_ldr64[torch.where(batched_indices_ldr64 == batch_idx)[0],:self.num_point_features]

            if self.shuffle_points and self.training:
                shuffle_idx = np.random.permutation(temp_points.shape[0])
                temp_points = temp_points[shuffle_idx,:]
            list_points.append(temp_points)
            
            
            voxels, coordinates, num_points = self.voxel_generator.generate(temp_points.numpy())
            voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
            coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1) # bzyx

            list_voxels.append(voxels)
            list_voxel_coords.append(coordinates)
            list_voxel_num_points.append(num_points)
        
        batched_points = torch.cat(list_points, dim=0)
        batch_dict['points'] = torch.cat((batched_indices_ldr64.reshape(-1,1), batched_points), dim=1).to(device)# b, x, y, z, intensity
        batch_dict['voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).to(device)
        batch_dict['voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).to(device)
        batch_dict['voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).to(device)
        batch_dict['gt_boxes'] = batch_dict['gt_boxes'].to(device)
        # batch_dict['gt_ldr'] = torch.concatenate([batch_dict['voxels'], batch_dict['voxels_coords'][:, 1:]], dim=-1)

        return batch_dict

import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
class RadarSparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarSparseProcessor, self).__init__()
        self.cfg = cfg
        # self.training = cfg.isTraining
        # Same rationale as LdrPreprocessor.shuffle_points: the sparse radar points
        # are stored in a fixed (likely power/SNR-descending) order, so without
        # shuffling, the max_num_voxels cap always keeps the same spatial subset.
        self.shuffle_points = True

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)

        cfg_ds = self.cfg.DATASET
        roi = cfg_ds.roi
        x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
        self.min_roi = [x_min, y_min, z_min]
        self.grid_size = roi.grid_size
        self.input_dim = 4 #cfg.MODEL.PRE_PROCESSOR.INPUT_DIM
        self.origin = torch.tensor([x_min, y_min, z_min])
        self.vsize_xyz = roi.voxel_size
        self.vsize_xyz = torch.tensor(self.vsize_xyz)

        # self.is_with_simplified_pointnet = cfg.MODEL.PRE_PROCESSOR.SIMPLIFIED_POINTNET.IS_WITH_SIMPLIFIED_POINTNET
        # if self.is_with_simplified_pointnet:
        #     out_channel = cfg.MODEL.PRE_PROCESSOR.SIMPLIFIED_POINTNET.OUT_CHANNEL
        #     cfg.MODEL.PRE_PROCESSOR.INPUT_DIM = out_channel
        #     self.simplified_pointnet = nn.Linear(self.input_dim, out_channel, bias=False)
        #     self.pooling_method = cfg.MODEL.PRE_PROCESSOR.SIMPLIFIED_POINTNET.POOLING

        max_vox_percentage = 0.25
        x_size = int(round((x_max-x_min)/self.grid_size))
        y_size = int(round((y_max-y_min)/self.grid_size))
        z_size = int(round((z_max-z_min)/self.grid_size))

        max_num_vox = self.cfg.DATASET.max_num_voxels #int(x_size*y_size*z_size*max_vox_percentage)

        # NOTE: self.grid_size is a plain float (roi.grid_size from the yaml), not a
        # tensor, so `self.grid_size.device` always raised AttributeError here and
        # silently fell back to 'cpu' -- forcing radar voxelization onto CPU even when
        # a GPU is available. Mirror the `'cuda' if torch.cuda.is_available() else
        # 'cpu'` convention used everywhere else in the training scripts instead.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.gen_voxels = PointToVoxel(
            # vsize_xyz = [self.grid_size, self.grid_size, self.grid_size],
            vsize_xyz= cfg_ds.roi.voxel_size,
            coors_range_xyz = roi.xyz,
            num_point_features = self.input_dim,
            max_num_voxels = max_num_vox,
            max_num_points_per_voxel = self.cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            device= torch.device(self.device)
        )

    def forward(self, dict_item):
        self.vsize_xyz = self.vsize_xyz .to(self.device)
        rdr_sparse = dict_item['rdr_sparse'].to(self.device)
        batch_indices = dict_item['batch_indices_rdr_sparse'].to(self.device)

        batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []

        for batch_idx in range(dict_item['batch_size']):
            corr_ind = torch.where(batch_indices == batch_idx)
            vox_in = rdr_sparse[corr_ind[0],:]

            if self.shuffle_points:
                shuffle_idx = torch.randperm(vox_in.shape[0], device=vox_in.device)
                vox_in = vox_in[shuffle_idx,:]

            voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(vox_in)
            voxel_batch_idx = torch.full((voxel_coords.shape[0], 1), batch_idx, device=rdr_sparse.device, dtype=torch.int64)
            voxel_coords = torch.cat((voxel_batch_idx, voxel_coords), dim=-1) # bzyx

            batch_voxel_features.append(voxel_features)
            batch_voxel_coords.append(voxel_coords)
            batch_num_pts_in_voxels.append(voxel_num_points)

        voxel_features, voxel_coords, voxel_num_points = torch.cat(batch_voxel_features), torch.cat(batch_voxel_coords), torch.cat(batch_num_pts_in_voxels)
        
        # voxel_features = voxel_features.sum(dim=1, keepdim=False)
        # normalizer = torch.clamp_min(voxel_num_points.view(-1,1), min=1.0).type_as(voxel_features)
        # voxel_features = voxel_features/normalizer

        dict_item['sp_features'] = voxel_features.contiguous()
        dict_item['sp_indices'] = voxel_coords.int()
        # dict_item['sp_rdr'] = torch.cat([dict_item['sp_features'], dict_item['sp_indices'][:, 1:]], dim=-1)
        return dict_item

class Rdr2LdrPvrcnnPP(PVRCNNPlusPlus):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.backbone_3d = self.backbone_3d.to(device)
        self.backbone_2d = self.backbone_2d.to(device)
        self.dense_head = self.dense_head.to(device)
        self.roi_head = self.roi_head.to(device)
        self.point_head = self.point_head.to(device)
        self.pfe = self.pfe.to(device)

        # print(f'PP self.training: {self.training}')
    
    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        
        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)
        
        if self.training:
            return batch_dict
        else:
            # print(f"Here post_processing, {self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']}")
            batch_dict = self.post_processing(batch_dict)
            
            return batch_dict
        


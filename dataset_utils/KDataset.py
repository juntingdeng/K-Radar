import numpy as np
import os
import sys
import collections
from pypcd4 import PointCloud
from PIL import Image
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
# sys.path.append('./K-Radar')
# sys.path.append('./K-Radar/models')

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_time(path):
    digit = int(path.split('.')[-2].split('_')[-1])
    return digit

class Kdataset:
    def __init__(self, root, seqs=[1], ldr_res=64, cam_chan='front', items=['rdr', 'ldr', 'cam']):
        self.items = items
        rdr_path_root = 'rdr_sparse_data/sparse_radar_tensor_wide_range/rtnh_wider_1p_1'
        ldr_path_root = 'sequences'
        cam_path_root = 'sequences'
        ldr_res = 'os2-64' if ldr_res == 64 else 'os1-128'
        cam_chan = 'cam-'+cam_chan

        self.data_dicts = collections.defaultdict(dict)
        for seq in seqs:
            rdr_path_seq = os.path.join(root, rdr_path_root, str(seq))
            ldr_path_seq = os.path.join(root, ldr_path_root, str(seq), ldr_res)
            cam_path_seq = os.path.join(root, cam_path_root, str(seq), cam_chan)

            rdr_paths = os.listdir(rdr_path_seq) 
            ldr_paths = os.listdir(ldr_path_seq) 
            cam_paths = os.listdir(cam_path_seq)

            for rdr_path in rdr_paths:
                t = get_time(rdr_path)
                self.data_dicts[(seq, t)]['rdr_path'] = os.path.join(rdr_path_seq, rdr_path)

            for ldr_path in ldr_paths:
                t = get_time(ldr_path)
                self.data_dicts[(seq, t)]['ldr_path'] = os.path.join(ldr_path_seq, ldr_path)

            for cam_path in cam_paths:
                t = get_time(cam_path)
                self.data_dicts[(seq, t)]['cam_path'] = os.path.join(cam_path_seq, cam_path)
        
        incomplete = []
        for frame, data in self.data_dicts.items():
            data_keys = list(data.keys()) # {'rdr_path':xxxx, 'ldr_path:xxxx, 'cam_path':xxxx}
            for item in self.items:
                if item+'_path' not in data_keys:
                    incomplete.append(frame)
                    break
        
        for frame in incomplete:
            del self.data_dicts[frame]
        
        self.frames = list(self.data_dicts.keys())
        self.data_dicts = list(self.data_dicts.values())
        print(f'{len(self.data_dicts)} complete frames are loaded, {len(incomplete)} incomplete frames have been removed.')

    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, index):
        frame, data_dict = self.frames[index], self.data_dicts[index]
        rdr_pts = np.load(data_dict['rdr_path'])
        ldr_pts = PointCloud.from_path(data_dict['ldr_path']).numpy()[:, :4] #(x, y, z, intensity)
        img = np.array(Image.open(data_dict['cam_path']))

        return frame, rdr_pts, ldr_pts, img

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
        # self.model_cfg = cfg.MODEL
        self.training = True
        self.split = 'train' if self.training else 'test'

        self.vsize_xyz=self.dataset_cfg.roi.voxel_size
        self.coors_range_xyz=np.array(self.dataset_cfg.roi.xyz)
        self.num_point_features= 4 #self.dataset_cfg.ldr64.n_used

        self.voxel_generator = VoxelWrapper(
            vsize_xyz=self.vsize_xyz,
            coors_range_xyz=self.coors_range_xyz,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel= 5 ,#self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
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
            
            # if (self.shuffle_points) and (self.training):
            #     shuffle_idx = np.random.permutation(temp_points.shape[0])
            #     temp_points = temp_points[shuffle_idx,:]
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

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)

        cfg_ds = self.cfg.DATASET
        roi = cfg_ds.roi
        x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
        self.min_roi = [x_min, y_min, z_min]
        self.grid_size = roi.grid_size
        self.input_dim = 4 #cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

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

        try:
            self.device = self.grid_size.device
        except:
            self.device = 'cpu'
        
        self.gen_voxels = PointToVoxel(
            # vsize_xyz = [self.grid_size, self.grid_size, self.grid_size],
            vsize_xyz= cfg_ds.roi.voxel_size,
            coors_range_xyz = roi.xyz,
            num_point_features = self.input_dim,
            max_num_voxels = max_num_vox,
            max_num_points_per_voxel = 5,
            device= torch.device(self.device)
        )

    def forward(self, dict_item):

        rdr_sparse = dict_item['rdr_sparse'].to(self.device)
        batch_indices = dict_item['batch_indices_rdr_sparse'].to(self.device)

        batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []

        for batch_idx in range(dict_item['batch_size']):
            corr_ind = torch.where(batch_indices == batch_idx)
            vox_in = rdr_sparse[corr_ind[0],:]
                
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
        


'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import shutil
import time
from torch.utils.data import Subset

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

from torch.utils.tensorboard import SummaryWriter
from spconv.pytorch import SparseConvTensor
from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.kitti_eval.eval_revised import get_official_eval_result_revised

from utils.util_optim import clip_grad_norm_
from depthEst.KDataset import *
from models.generatives.unet import *

d = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_local_time_str():
    now = time.localtime()
    tm_year = f'{now.tm_year}'[2:4]
    tm_mon = f'{now.tm_mon}'.zfill(2)
    tm_mday = f'{now.tm_mday}'.zfill(2)
    tm_mday = f'{now.tm_mday}'.zfill(2)
    tm_hour = f'{now.tm_hour}'.zfill(2)
    tm_min = f'{now.tm_min}'.zfill(2)
    tm_sec = f'{now.tm_sec}'.zfill(2)
    return f'{tm_year}{tm_mon}{tm_mday}_{tm_hour}{tm_min}{tm_sec}'

class Validate:
    def __init__(self, cfg, gen_net, dect_net, spatial_size=[]):
        self.is_validate = True
        self.gen_net = gen_net
        self.dect_net = dect_net
        self.cfg = cfg
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL
        self.Nvoxels = self.cfg.DATASET.max_num_voxels
        self.voxel_size = self.cfg.DATASET.roi.voxel_size #xyz
        self.spatial_size = spatial_size #zyx
        x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
        self.origin = torch.tensor([x_min, y_min, z_min]).to(d)

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []
        str_local_time = get_local_time_str()
        str_exp = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)
        self.log_test = os.path.join(self.path_log, 'test')

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

        self.rdr_processor = RadarSparseProcessor(cfg)
        self.ldr_processor = LdrPreprocessor(cfg)

        self.voxel_size = torch.tensor(self.voxel_size).to(d)

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, data_loader=None):
        # self.network.eval()
        if self.gen_net:
            self.gen_net.eval()
        self.dect_net.training=False
        self.dect_net.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            # dict_label = self.dataset_test.label.copy()
            dict_label = self.cfg.DATASET.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        
        road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        time_cond_list = ['day', 'night']
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        # Check is_validate with small dataset
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='Test (Subset): ')
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(data_loader), desc='Test (Total): ')

        # data_loader = torch.utils.data.DataLoader(self.dataset_test, \
        #         batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn, \
        #         num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

            os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                f.write('')

            for road_cond in road_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                    f.write('')

            for time_cond in time_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                    f.write('')

            for weather_cond in weather_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                    f.write('')

            pred_dir_list = []
            label_dir_list = []
            desc_dir_list = []
            split_path_list = []

            ### For All Conditions ###
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dir_list.append(preds_dir)
            label_dir_list.append(labels_dir)
            desc_dir_list.append(desc_dir)
            split_path_list.append(split_path)
                            
            ### For Specific Conditions ###
            for road_cond in road_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)
                
                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for time_cond in time_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for weather_cond in weather_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

        # Creating gts and preds txt files for evaluation
        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset:
                if (idx_datum >= self.val_num_subset):
                    break

            try:
                # dict_out = self.network(dict_datum) # inference
                # is_feature_inferenced = True

                is_feature_inferenced = True

                dict_datum = self.rdr_processor.forward(dict_datum)
                dict_datum = self.ldr_processor.forward(dict_datum)

                for key, val in dict_datum.items():
                    if key in ['points', 'voxels', 'voxel_coords', 'voxel_num_points', 'gt_boxes', 'sp_features', 'sp_indices']:
                        if isinstance(val, np.ndarray):
                            dict_datum[key] = torch.tensor(val).to(device)
                        elif isinstance(val, torch.Tensor) and val.device != device:
                            dict_datum[key] = dict_datum[key].to(device)

                
                
                if self.gen_net:
                    rdr_data = dict_datum['sp_features']
                    if rdr_data.shape[0] < self.Nvoxels:
                        rdr_data = torch.vstack([rdr_data, torch.zeros((self.Nvoxels - rdr_data.shape[0], rdr_data.shape[1], rdr_data.shape[2])).to(d)])
                        dict_datum['sp_indices'] = torch.vstack([dict_datum['sp_indices'], torch.zeros((self.Nvoxels - dict_datum['sp_indices'].shape[0], dict_datum['sp_indices'].shape[1])).to(d)])

                    
                ldr_data = dict_datum['voxels']
                lmin, lmax = ldr_data.min(), ldr_data.max()
                if ldr_data.shape[0] < self.Nvoxels:
                        dict_datum['voxels'] = torch.vstack([ldr_data, torch.zeros((self.Nvoxels - ldr_data.shape[0], ldr_data.shape[1], ldr_data.shape[2])).to(d)])
                        dict_datum['voxel_coords'] = torch.vstack([dict_datum['voxel_coords'], torch.zeros((self.Nvoxels - dict_datum['voxel_coords'].shape[0], dict_datum['voxel_coords'].shape[1])).to(d)])
                dict_datum['voxel_num_points'] = torch.tensor(self.Nvoxels).to(d)
                
                if self.gen_net:
                    #vsize: zyx
                    radar_st = SparseConvTensor(features=rdr_data.reshape((self.Nvoxels, -1)), 
                                            indices=dict_datum['sp_indices'].int(), 
                                            spatial_shape=self.spatial_size, 
                                            batch_size=1)
                    
                    lidar_st = SparseConvTensor(features=ldr_data.reshape((self.Nvoxels, -1)), 
                                            indices=dict_datum['voxel_coords'].int(), 
                                            spatial_shape=self.spatial_size, 
                                            batch_size=1)
            
                    # Pseudocode
                    rad_idx = radar_st.indices           # [Nr,4]
                    lid_idx = lidar_st.indices           # [Nl,4]

                    all_idx = torch.cat([rad_idx, lid_idx], dim=0)
                    all_idx = torch.unique(all_idx, dim=0)  # union of occupied voxels
                    union_st = scatter_radar_to_union(radar_st, all_idx, self.spatial_size, 1)

                    out = self.gen_net(union_st) 

                    pred, occ, attrs = out['st'], out['logits'], out['attrs']
                    offs = attrs[:, :, :3]

                    voxel_center_xyz = self.origin + (torch.flip(union_st.indices[:, 1:4].float(), dims=[1]) + 0.5) * torch.tensor(self.voxel_size).to(d)  # grid center
                    pred_offset_m = offs * self.voxel_size #scale voxel-units → meters
                    voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 5, 1)
                    # print(voxel_center_xyz.shape, pred_offset_m.shape)
                    attrs = torch.cat([voxel_center_xyz + pred_offset_m, attrs[:, :, -1][..., None]], dim=-1)
                    _pred_indices = pred.indices.detach()
                    _attrs = attrs.detach()
                    if (torch.isnan(_attrs)).any():
                        print(f'_attrs has nan')
                    _, topN = torch.topk(_attrs[:, :, -1].mean(1), k=self.Nvoxels)
                    dict_datum['voxels'] = _attrs.contiguous().float().to(d)[topN]
                    dict_datum['voxel_coords'][:, 1] = _pred_indices[:, 0].int().clamp(1, self.spatial_size[0]-1)[topN]
                    dict_datum['voxel_coords'][:, 2] = _pred_indices[:, 1].int().clamp(1, self.spatial_size[1]-1)[topN]
                    dict_datum['voxel_coords'][:, 3] = _pred_indices[:, 2].int().clamp(1, self.spatial_size[2]-1)[topN]
                    dict_datum['voxel_coords'] = dict_datum['voxel_coords'].to(d)
                    dict_datum['voxel_num_points'] = torch.tensor(self.Nvoxels).to(d)
                
                dict_out = self.dect_net(dict_datum)

            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            if is_print_memory:
                print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                
            idx_name = str(idx_datum).zfill(6)
            
            road_cond_tag, time_cond_tag, weather_cond_tag = \
                dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
            # print(dict_out['desc'][0])

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                os.makedirs(labels_dir_road, exist_ok=True)
                os.makedirs(labels_dir_time, exist_ok=True)
                os.makedirs(labels_dir_weather, exist_ok=True)
                os.makedirs(desc_dir_road, exist_ok=True)
                os.makedirs(desc_dir_time, exist_ok=True)
                os.makedirs(desc_dir_weather, exist_ok=True)
                os.makedirs(preds_dir_road, exist_ok=True)
                os.makedirs(preds_dir_time, exist_ok=True)
                os.makedirs(preds_dir_weather, exist_ok=True)
                
                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]
                            
                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                if dict_out_current is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                    pass
                else:
                    ### Gt ###
                    # print(f"len(dict_out_current['kitti_gt']): {len(dict_out_current['kitti_gt'])}")
                    for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                        if idx_label == 0:
                            mode = 'w'
                        else:
                            mode = 'a'

                        with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])

                    ### Process description ###
                    if len(dict_out_current['kitti_pred']) == 0:
                        with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                    else:
                        for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                            if idx_pred == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                    
                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
                    with open(split_path_road, 'a') as f:
                        f.write(str_log)
                    with open(split_path_time, 'a') as f:
                        f.write(str_log)
                    with open(split_path_weather, 'a') as f:
                        f.write(str_log)
                        
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
        for conf_thr in list_conf_thr:
            for condition in all_condition_list:
                try:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                    split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                    dt_annos = kitti.get_label_annos(preds_dir)
                    val_ids = read_imageset_file(split_path)
                    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                    list_metrics = []
                    list_results = []
                    for idx_cls_val in self.list_val_care_idx:
                        # if self.is_validation_updated:
                        #     # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
                        #     # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
                        #     dict_metrics, result = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        # else:
                        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        list_metrics.append(dict_metrics)
                        list_results.append(result)
                    print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                    with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                        for dic_metric in list_metrics:
                            print('='*50)
                            print('Cls: ', dic_metric['cls'])
                            print('IoU:', dic_metric['iou'])
                            print('BEV: ', dic_metric['bev'])
                            print('3D: ', dic_metric['3d'])
                            print('-'*50)
                            
                            f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                            f.write('cls: ' + dic_metric['cls'] + '\n')
                            f.write('iou: ')
                            for iou in dic_metric['iou']:
                                f.write(str(iou) + ' ')
                            f.write('\n')
                            f.write('bev: ')
                            for bev in dic_metric['bev']:
                                f.write(str(bev) + ' ')
                            f.write('\n')
                            f.write('3d  :')
                            for det3d in dic_metric['3d']:
                                f.write(str(det3d) + ' ')
                            f.write('\n\n')
                    print('\n')
                except:
                    print('* Exception error (Pipeline): Samples for the codition are not found')

        path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
        print(f'* Check {path_check}')
        ### Validate per conf ###

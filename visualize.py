import open3d as o3d
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus
from models.skeletons.rdr_base import RadarBase
from models.generatives.unet import *

from dataset_utils.KDataset import *
from torch.amp import GradScaler
# from pipelines.pipeline_dect import Validate
from visualize_unet_points import *
from models.generatives.unet_utlis import *
from models.generatives.generative import *

def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--log_sig', type=str, default='251218_214707')
    args.add_argument('--load_epoch', type=str, default='150')
    args.add_argument('--set', type=str, default='train')
    args.add_argument('--thresh', type=float, default=0.9)
    args.add_argument('--model_cfg', type=str, default='ldr')
    args.add_argument('--gt_topk', default=100, type=int)
    args.add_argument('--mdn', action='store_true')
    args.add_argument('--plot_nframes', default=1, type=int)
    args.add_argument('--plot_all', action='store_true')
    args.add_argument('--newtest', default=None, type=str)
    return args.parse_args()

def save_open3d_render(points_xyz, intensities=None, boxes_kitti=None,
                       filename="render.png", point_size=1.0, w=1600, h=900):
    """
    Render the synthesized LiDAR point cloud + boxes to an image file.
    Works headlessly (no window).
    """
    # print(f'Here: {type(points_xyz)}')
    scene = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene.scene.set_background([1, 1, 1, 1])  # white background

    # ---- Point cloud ----
    _3dvec = o3d.utility.Vector3dVector(points_xyz)
    pcd = o3d.geometry.PointCloud(_3dvec)
    if intensities is None:
        colors = np.zeros_like(points_xyz); colors[:] = [0.2, 0.7, 1.0]
    else:
        v = (intensities - np.min(intensities)) / (np.ptp(intensities) + 1e-6)
        colors = np.stack([v, np.minimum(1.0, 0.5 + 0.5*v), 1 - v], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    scene.scene.add_geometry("points", pcd, o3d.visualization.rendering.MaterialRecord())
    scene.scene.show_axes(False)

    # ---- Boxes ----
    if boxes_kitti is not None:
        for i, b in enumerate(boxes_kitti):
            x, y, z, h, w_b, l, yaw = b.tolist()
            X = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
            Y = np.array([ w_b/2, -w_b/2, -w_b/2,  w_b/2,  w_b/2, -w_b/2, -w_b/2,  w_b/2])
            Z = np.array([ 0, 0, 0, 0, -h, -h, -h, -h])
            C = np.vstack([X,Y,Z])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            C = R @ C + np.array([[x],[y],[z]])
            lines = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(C.T),
                lines=o3d.utility.Vector2iVector(lines))
            line_mat = o3d.visualization.rendering.MaterialRecord()
            line_mat.shader = "unlitLine"
            line_mat.line_width = 2.0
            line_mat.base_color = (1.0, 0.6, 0.0, 1.0)
            scene.scene.add_geometry(f"box{i}", line_set, line_mat)

    # ---- Camera setup ----
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent().max()
    cam = scene.scene.camera
    cam.look_at(center, center + [extent, 0, 0], [0, 0, 1])  # simple top-view
    # cam.set_projection(60.0, w/h, 0.1, 1000.0)
    cam.set_projection(
        60.0,                                   # field_of_view (degrees)
        w/h,                                 # width / height
        0.1,                                    # near
        1000.0,                                 # far
        o3d.visualization.rendering.Camera.FovType.Vertical
    )

    # ---- Render & save ----
    img = scene.render_to_image()
    o3d.io.write_image(filename, img)
    print(f"Saved: {filename}")

if __name__ == '__main__':
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = arg_parser()
    log_sig = args.log_sig
    load_epoch = args.load_epoch
    set = args.set

    cfg_path = './configs/cfg_rdr_ldr.yml' if args.model_cfg == 'ldr' else './configs/cfg_rdr_ldr_sps.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)

    x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
    vsize_xyz = cfg.DATASET.roi.voxel_size
    x_size = int(round((x_max-x_min)/vsize_xyz[0]))
    y_size = int(round((y_max-y_min)/vsize_xyz[1]))
    z_size = int(round((z_max-z_min)/vsize_xyz[2]))
    print(f'zyx-size: {z_size}, {y_size}, {x_size}')
    centors = voxel_center(cfg=cfg).to(d)
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

    Nvoxels = cfg.DATASET.max_num_voxels
    gen_net = SparseUNet3D(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL) if not args.mdn else SparseUNet3D_MDN(in_ch=4*cfg.MODEL.PRE_PROCESSING.MAX_POINTS_PER_VOXEL)
    dect_net = Rdr2LdrPvrcnnPP(cfg=cfg) if args.model_cfg == 'ldr' else RadarBase(cfg=cfg)
    model_load = torch.load(f'./logs/exp_{log_sig}_RTNH/models/epoch{load_epoch}.pth')
    gen_net.load_state_dict(state_dict=model_load['gen_state_dict'])
    dect_net.load_state_dict(state_dict=model_load['dect_state_dict'])

    # ppl = Validate(cfg=cfg, gen_net=gen_net, dect_net=dect_net, spatial_size=[z_size, y_size, x_size])
    # ppl.set_validate()

    gen_net = gen_net.to(d)
    # gen_net.eval()
    dl = test_dataloader if args.set == 'test' else train_dataloader
    x_min_all, x_max_all = float('inf'), 0
    radar_error_all, lidar_err_all = [], []
    for bi, batch_dict in enumerate(dl):
        if bi >0:
            # print(f'finished {bi}')
            break

        # print(f'idx:{batch_dict}, dict_datum:{batch_dict}')
        if not args.plot_all and bi >= args.plot_nframes: break
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
        print(f'rdr_data.shape[0]: {rdr_data.shape[0]}, rdr_data: {rdr_data}')
        print(f'ldr_data.shape[0]: {ldr_data.shape[0]}, ldr_data: {ldr_data}')
        if rdr_data.shape[0] < Nvoxels:
            n = rdr_data.shape[0]
            while n < Nvoxels:
                rdr_data = torch.vstack([rdr_data, rdr_data[ :Nvoxels - n]])
                batch_dict['sp_indices'] = torch.vstack([batch_dict['sp_indices'], batch_dict['sp_indices'][: Nvoxels- n]])
                n = rdr_data.shape[0]
            
            batch_dict['sp_features'] = rdr_data

        if ldr_data.shape[0] < Nvoxels:
            n = ldr_data.shape[0]
            while n < Nvoxels:
                ldr_data = torch.vstack([ldr_data, ldr_data[: Nvoxels - n]])
                batch_dict['voxels'] = ldr_data
                batch_dict['voxel_coords'] = torch.vstack([batch_dict['voxel_coords'], batch_dict['voxel_coords'][: Nvoxels- n]])
                batch_dict['voxel_num_points'] = torch.concat([batch_dict['voxel_num_points'], batch_dict['voxel_num_points'][: Nvoxels - n]])
                n = ldr_data.shape[0]

        # spconv unet
        radar_st = SparseConvTensor(features=rdr_data.reshape((Nvoxels, -1)), 
                                    indices=batch_dict['sp_indices'].int(), 
                                    spatial_shape=[z_size, y_size, x_size], 
                                    batch_size=bs)

        lidar_st = SparseConvTensor(features=ldr_data.reshape((Nvoxels, -1)), 
                                    indices=batch_dict['voxel_coords'].int(), 
                                    spatial_shape=[z_size, y_size, x_size], 
                                    batch_size=bs)
        
        # matched, gt_offsets, gt_features = local_match_new(radar_st, lidar_st)
        
        rdr_is_all_zerocount = (rdr_data[:, 1:, :] == 0).all(dim=(1, 2)).sum()
        ldr_is_all_zerocount = (ldr_data[:, 1:, :] == 0).all(dim=(1, 2)).sum()
        print(f'rdr_is_all_zerocount: {rdr_is_all_zerocount}, ldr_is_all_zerocount:{ldr_is_all_zerocount}')
        rdr_cnt = (rdr_data.abs().sum(dim=2) == 0).sum(dim=1)
        ldr_cnt = (ldr_data.abs().sum(dim=2) == 0).sum(dim=1)
        print(f'rdr_cnt: {rdr_cnt.sum()/rdr_cnt.shape[0]}, ldr_cnt: {ldr_cnt.sum()/ldr_cnt.shape[0]}')

        rad_idx = radar_st.indices           # [Nr,4]
        lid_idx = lidar_st.indices           # [Nl,4]

        all_idx = torch.cat([rad_idx, lid_idx], dim=0)
        all_idx = torch.unique(all_idx, dim=0)  # union of occupied voxels
        union_st = scatter_radar_to_union(radar_st, all_idx, [z_size, y_size, x_size], bs)

        #gt_d: zyx
        matched, gt_d, gt_f, gt_coords = local_match_closest(radar_st, lidar_st, gt_topk=args.gt_topk) if not args.mdn else local_match_closest_mdn(radar_st, lidar_st, gt_topk=args.gt_topk)
        gt_d_xyz = torch.flip(gt_d, dims=[-1]) #gt_d: zyx -> xyz
        print(f'gt_d: {gt_d.shape}, gt_f:{gt_f.shape}, {gt_d.abs().mean().item()}, {gt_d.abs().median().item()}, {gt_d.abs().min().item()}, {gt_d.abs().max().item()}')

        rdr_features = rdr_data[:, 0, :3]
        ids = (rdr_features != 0).any(dim=1)      # boolean mask
        rdr_features_nonzeros = rdr_features[ids]
        # print(f'rdr_features_nonzeros: {rdr_features_nonzeros.shape}')
        voxel_min_xyz = origin + (torch.flip(batch_dict['sp_indices'][ids, 1:4].float(), dims=[1]) + -1.) * vsize_xyz 
        voxel_max_xyz = origin + (torch.flip(batch_dict['sp_indices'][ids, 1:4].float(), dims=[1]) + 1.) * vsize_xyz 
        within = (rdr_features_nonzeros <= voxel_max_xyz) & (rdr_features >= voxel_min_xyz)
        within_ = within.all(axis=1)
        # print(f"Radar within: {within}, sum:{sum(within_)}, N points:{ids.shape[0]}")

        # print(f'lidar data:{ldr_data}')
        ldr_features = ldr_data[:, 0, :3]
        ids = (ldr_features != 0).any(dim=1)      # boolean mask
        ldr_features_nonzeros = ldr_features[ids]
        # print(f'ldr_features_nonzeros: {ldr_features_nonzeros.shape}')
        voxel_min_xyz = origin + (torch.flip(batch_dict['voxel_coords'][ids, 1:4].float(), dims=[1]) + 0.) * vsize_xyz 
        voxel_max_xyz = origin + (torch.flip(batch_dict['voxel_coords'][ids, 1:4].float(), dims=[1]) + 1.) * vsize_xyz 
        within = (ldr_features_nonzeros <= voxel_max_xyz) & (ldr_features_nonzeros >= voxel_min_xyz)
        within_ = within.all(axis=1)
        # print(f"Lidar within: {within}, sum:{sum(within_)}, N points:{ids.shape[0]}")

        
        out = gen_net(radar_st)
        # print(f'out:{out}, radar_st.features:{radar_st.features}, radar_st.indices:{radar_st.indices}')
        prob_thresh=0.9
        if not args.mdn:
            pred, occ, attrs = out['st'], out['logits'], out['attrs']
            offs = attrs[:, :, :3]
            # print(f'offs: {offs.abs().mean().item()}, {offs.abs().median().item()}, {offs.abs().min().item()}, {offs.abs().max().item()}')

            voxel_center_xyz = origin + (torch.flip(pred.indices[:, 1:4].float(), dims=[1]) + 0.5) * vsize_xyz  # grid center
            pred_offset_m = offs * vsize_xyz.to(d)  # scale voxel-units → meters
            voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 5, 1)
            # print(voxel_center_xyz.shape, pred_offset_m.shape)
            # print(f'pred_offset_m:{pred_offset_m.shape}, {type(pred_offset_m)}, voxel_center_xyz:{voxel_center_xyz}')
            attrs = torch.cat([voxel_center_xyz+pred_offset_m, attrs[:, :, 3:4]], dim=-1)

            _pred_indices = pred.indices.detach()
            _attrs = attrs.detach() # xyz
            if (torch.isnan(_attrs)).any():
                print(f'_attrs has nan')
        
            # select valid slots by probability
            probs = torch.sigmoid(occ)                 # [N,K,1]
            keep = (probs > prob_thresh)
            voxel_num_points = keep.sum(dim=1) #[N, ]
            keep = keep.repeat(1,1,4) 
            # print(f'keep:{keep.shape}, _attrs:{_attrs.shape}')
            # print(f"batch_dict['voxel_num_points']: {voxel_num_points}")

            _attrs = torch.where(keep, _attrs, torch.zeros_like(_attrs)).detach().cpu().numpy()

            # _attrs = torch.where(keep.unsqueeze(-1), attrs, torch.zeros_like(attrs)).detach().cpu().numpy()
            points_xyz = _attrs[:,:, :3].reshape(-1, 3)
            intensity = _attrs[:,:, -1].reshape(-1)

            points_xyz = np.ascontiguousarray(points_xyz)
            intensity = np.ascontiguousarray(intensity)
            # print(f'points:{points_xyz.shape}, intensity:{intensity.shape}')
        
        else:
            pred_st, offs, occ = out['st'], out['mu_off'], out['occ_logit']
            # out['mu_off'] = gt_d_xyz
            print(f"out['mu_off']: {out['mu_off']}, gt_d_xyz: {gt_d_xyz}")
            print(f"out['mu_off']- gt_d: {(out['mu_off'].mean(dim=1) - gt_d_xyz.mean(dim=1))}")
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
            
            attrs_pts_gt, voxel_coords_gt, voxel_num_points_gt, _, _, _ = sample_points_from_mdn(
                                                                pred_st=out['st'],
                                                                mu_off=gt_d_xyz,
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
            print(f'rdr indices: {radar_st.indices}, pred indices: {pred_st.indices}')
            ids = (attrs_pts[:, 0, :3] != 0).any(dim=1)      # boolean mask
            features_nonzeros = attrs_pts[ids, 0, :3]
            print(f'features_nonzeros: {features_nonzeros.shape}')

            # new_coords = mu[ids].int() + torch.flip(pred_st.indices[ids, 1:4], dims=[1]) #xyz
            voxel_coords[:, 1:4] += torch.flip(mu.int(), dims=[1]) 
            voxel_min_xyz = origin + (torch.flip(voxel_coords[ids, 1:4], dims=[1]) + 0.) * vsize_xyz 
            voxel_max_xyz = origin + (torch.flip(voxel_coords[ids, 1:4], dims=[1]) + 1.) * vsize_xyz 

            within = (features_nonzeros <= voxel_max_xyz) & (features_nonzeros >= voxel_min_xyz)
            within_any = within.any(axis=1)
            within_all = within.all(axis=1)
            print(f"Prediced: within: {within}, sum-any:{sum(within_any)}, sum-all:{sum(within_all)}, N points:{features_nonzeros.shape[0]}")

            points_xyz = attrs_pts[:,:, :3].reshape(-1, 3).detach().cpu().numpy()
            intensity = attrs_pts[:,:, -1].reshape(-1).detach().cpu().numpy()

            points_xyz = np.ascontiguousarray(points_xyz)
            intensity = np.ascontiguousarray(intensity)

        # plot gt
        gt_f = gt_f.view(gt_f.shape[0], -1, 4)
        # print(voxel_center_xyz_gt.shape, offset_m_gt.shape, gt_f.shape)
        attrs_gt = attrs_pts_gt.detach().cpu().numpy() #torch.cat([voxel_center_xyz_gt + offset_m_gt, gt_f[:, :, 3:4]], dim=-1).detach().cpu().numpy()

        points_xyz_gt = attrs_gt[:,:, :3].reshape(-1, 3)
        intensity_gt = attrs_gt[:,:, -1].reshape(-1)

        # points_xyz_gt = batch_dict['voxels'][:, :, :3].reshape(-1, 3).detach().cpu().numpy()
        # intensity_gt = batch_dict['voxels'][:, :, -1].reshape(-1).detach().cpu().numpy()
        # print(f"batch_dict['voxels']: {batch_dict['voxels'][0]}, attrs_gt:{attrs_gt[0]}")
        # print(f"batch_dict['voxel_coords']: {batch_dict['voxel_coords'][0]}, voxel_coords:{gt_coords[0]}")
        # print(f"batch_dict['voxel_num_points']: {batch_dict['voxel_num_points'][0]}, voxel_num_points:{voxel_num_points[0]}")

        points_xyz_gt = np.ascontiguousarray(points_xyz_gt)
        intensity_gt = np.ascontiguousarray(intensity_gt)
        # print(f'points:{points_xyz_gt.shape}, intensity:{intensity_gt.shape}')
        
        pred_xyz = unet_slots_to_xyz_attrs(
            out,                       # your dict
            offs,
            occ,
            voxel_size=cfg.DATASET.roi.voxel_size,   # <-- set to your grid
            origin=origin,       # <-- set to your grid origin
            prob_thresh=prob_thresh,
            clamp_offsets=False
        )
        # print(f'points_xyz:{points_xyz}, intensity:{intensity}')
        rdr_points_xyz=np.ascontiguousarray(batch_dict['sp_features'][:, :, :3].detach().cpu().numpy().reshape(-1, 3)) 
        rdr_intensities=np.ascontiguousarray(batch_dict['sp_features'][:, :, -1].detach().cpu().numpy().reshape(-1)) 
        ldr_points_xyz=np.ascontiguousarray(batch_dict['voxels'][:, :, :3].detach().cpu().numpy().reshape(-1, 3))
        ldr_intensities=np.ascontiguousarray(batch_dict['voxels'][:, :, -1].detach().cpu().numpy().reshape(-1))

        # Pick a shared camera pose (e.g., from LiDAR cloud)
        pose = compute_reference_pose(ldr_points_xyz, view="bev")

        # Save all with the SAME pose
        fig_path = os.path.join('visualize', log_sig, load_epoch)
        os.makedirs(fig_path, exist_ok=True)
        # print(f'image save path: {fig_path}')
        list_tuple_objs = batch_dict['meta'][0]['label']
        dx, dy, dz = batch_dict['meta'][0]['calib']
        gt_boxes = []
        for obj in list_tuple_objs:
            cls_name, (x, y, z, th, l, w, h), trk, avail = obj
            x = x + dx
            y = y + dy
            z = z + dz
            # print(f'dx, dy, dz: {dx}, {dy}, {dz}')
            gt_boxes.append([cls_name, (x, y, z, th, l, w, h), trk, avail])

        # print(f'points_xyz: {points_xyz.shape[0]}, pred_xyz: {pred_xyz.shape[0]}, \
        #        rdr_points_xyz: {rdr_points_xyz.shape[0]}, ldr_points_xyz: {ldr_points_xyz.shape[0]}')

        # print(f'points_xyz: {intensity.mean()},\
        #        rdr_points_xyz: {rdr_intensities.mean()}, ldr_points_xyz: {ldr_intensities.mean()}')
        
        # randinx = random.sample(range(0, points_xyz.shape[0]), k=10000)
        # _, randinx = torch.topk(_attrs[:, :, -1].mean(1), k=10000)
        # plot_quiver(pts_pred=rdr_points_xyz, off_pred=pred_offset_m.detach().cpu().numpy().reshape(-1, 3), name=os.path.join(fig_path, f"{set}_{bi}_offset_quiver.png"))
        if not args.plot_all and bi < args.plot_nframes:
            prefix = f"{set}_{bi}" if not args.newtest else f"{set}_new{args.newtest}_{bi}"
            print(f'/////// prefix:{prefix}')
            save_open3d_render_offsets(points_xyz=points_xyz, 
                                    points_gt=attrs_gt[:,:,:3].reshape(-1, 3), 
                                    points_rdr=rdr_points_xyz,
                                        intensities=intensity,
                                        intensities_gt=intensity_gt, 
                                        intensities_rdr=rdr_intensities,
                                        boxes=gt_boxes,
                                        filename=os.path.join(fig_path, f"{prefix}_offset.png"), 
                                        pose=pose)
            if args.mdn:
                save_open3d_render_fixed_pose(points_xyz=points_xyz, 
                                            intensities=intensity, 
                                            boxes=gt_boxes,
                                            filename=os.path.join(fig_path, f"{prefix}_pred1.png"), 
                                            pose=pose)
            save_open3d_render_fixed_pose(points_xyz=pred_xyz, 
                                        intensities=intensity, 
                                        boxes=gt_boxes,
                                        filename=os.path.join(fig_path, f"{prefix}_pred2.png"), 
                                        pose=pose)
            save_open3d_render_fixed_pose(rdr_points_xyz, 
                                        intensities=rdr_intensities, 
                                        boxes=gt_boxes,
                                        filename=os.path.join(fig_path, f"{prefix}_rdr.png"),   
                                        pose=pose)
            save_open3d_render_fixed_pose(ldr_points_xyz, 
                                        intensities=ldr_intensities, 
                                        boxes=gt_boxes,
                                        filename=os.path.join(fig_path, f"{prefix}_ldr.png"),  
                                        pose=pose)
            
            save_open3d_render_fixed_pose(points_xyz=points_xyz_gt, 
                                        intensities=intensity_gt, 
                                        boxes=gt_boxes,
                                        filename=os.path.join(fig_path, f"{prefix}_gt.png"), 
                                        pose=pose)


        # radar_pts: (Nr, 3), lidar_pts: (Nl, 3), pred_pts_union: (Np, 3)
        bin_x, radar_err, lidar_err, radar_cnt, lidar_cnt, _x_min, _x_max = modality_error_vs_range_numpy_with_zero(
            rdr_points_xyz, ldr_points_xyz, points_xyz, num_bins=700
        )
        radar_error_all.append(radar_err)
        lidar_err_all.append(lidar_err)
        x_min_all, x_max_all = min(x_min_all, _x_min), max(x_max_all, x_max)

        if not args.plot_all and bi< args.plot_nframes:
            fig, ax, (radar_x, radar_y), (lidar_x, lidar_y) = plot_mapping_error_cdf(radar_dists=radar_err, lidar_dists=lidar_err, unit='m', save_path=os.path.join(fig_path, f"{prefix}_error_cdf.png"))
            np.save(os.path.join(fig_path, f'radar_x_{prefix}.npy'), radar_x)
            np.save(os.path.join(fig_path, f'radar_y_{prefix}.npy'), radar_y)
            np.save(os.path.join(fig_path, f'lidar_x_{prefix}.npy'), lidar_x)
            np.save(os.path.join(fig_path, f'lidar_y_{prefix}.npy'), lidar_y)

    plot_mapping_error_cdf(radar_dists=np.stack(radar_error_all).reshape(-1), lidar_dists=np.stack(lidar_err_all).reshape(-1), unit='m', save_path=os.path.join(fig_path, f"{set}_all_error_cdf.png"))
    np.save(os.path.join(fig_path, f'{set}_radar_error_all.npy'), radar_error_all)
    np.save(os.path.join(fig_path, f'{set}_lidar_error_all.npy'), lidar_err_all)

        # # Plot
        # plt.figure()
        # plt.plot(bin_x, radar_err, label="Radar error")
        # plt.xlabel("x distance (m)")
        # plt.ylabel("Avg NN error to prediction (m)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(fig_path, f"{set}_{bi}_error_rdr.png"))

        # plt.figure()
        # plt.plot(bin_x, lidar_err, label="LiDAR error")
        # plt.xlabel("x distance (m)")
        # plt.ylabel("Avg NN error to prediction (m)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(fig_path, f"{set}_{bi}_error_ldr.png"))

        
import open3d as o3d
import numpy as np
import argparse

from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from utils.util_config import *
from models.skeletons import PVRCNNPlusPlus
from models.generatives.unet import *

from depthEst.KDataset import *
from torch.amp import GradScaler
from pipelines.pipeline_dect import Validate
from visualize_unet_points import *

def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--log_sig', type=str, default='251119_142454')
    args.add_argument('--load_epoch', type=str, default='30')
    args.add_argument('--set', type=str, default='test')
    args.add_argument('--thresh', type=float, default=0.9)
    return args.parse_args()

def save_open3d_render(points_xyz, intensities=None, boxes_kitti=None,
                       filename="render.png", point_size=1.0, w=1600, h=900):
    """
    Render the synthesized LiDAR point cloud + boxes to an image file.
    Works headlessly (no window).
    """
    print(f'Here: {type(points_xyz)}')
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
    cfg_path = './configs/cfg_rdr_ldr.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)
    args = arg_parser()
    log_sig = args.log_sig
    load_epoch = args.load_epoch
    set = args.set

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
                                  collate_fn=train_kdataset.collate_fn, num_workers=4, shuffle=False)

    test_kdataset = KRadarDetection_v2_0(cfg=cfg, split='test')
    test_dataloader = DataLoader(test_kdataset, batch_size=bs, 
                            collate_fn=test_kdataset.collate_fn, num_workers=4, shuffle=False)

    rdr_processor = RadarSparseProcessor(cfg)
    ldr_processor = LdrPreprocessor(cfg)

    Nvoxels = cfg.DATASET.max_num_voxels
    gen_net = SparseUNet3D(in_ch=20)
    dect_net = Rdr2LdrPvrcnnPP(cfg=cfg)
    model_load = torch.load(f'./logs/exp_{log_sig}_RTNH/models/epoch{load_epoch}.pth')
    gen_net.load_state_dict(state_dict=model_load['gen_state_dict'])
    dect_net.load_state_dict(state_dict=model_load['dect_state_dict'])

    ppl = Validate(cfg=cfg, gen_net=gen_net, dect_net=dect_net, spatial_size=[z_size, y_size, x_size])
    ppl.set_validate()

    gen_net = gen_net.to(d)
    for bi, batch_dict in enumerate(test_dataloader):
        if bi > 3 : break

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
        
        if rdr_data.shape[0] < Nvoxels:
            rdr_data = torch.vstack([rdr_data, torch.zeros((Nvoxels - rdr_data.shape[0], rdr_data.shape[1], rdr_data.shape[2])).to(d)])
            batch_dict['sp_indices'] = torch.vstack([batch_dict['sp_indices'], torch.zeros((Nvoxels - batch_dict['sp_indices'].shape[0], batch_dict['sp_indices'].shape[1])).to(d)])

        if ldr_data.shape[0] < Nvoxels:
            ldr_data = torch.vstack([ldr_data, torch.zeros((Nvoxels - ldr_data.shape[0], ldr_data.shape[1], ldr_data.shape[2])).to(d)])
            batch_dict['voxel_coords'] = torch.vstack([batch_dict['voxel_coords'], torch.zeros((Nvoxels - batch_dict['voxel_coords'].shape[0], batch_dict['voxel_coords'].shape[1])).to(d)])

        # spconv unet
        radar_st = SparseConvTensor(features=rdr_data.reshape((Nvoxels, -1)), 
                                    indices=batch_dict['sp_indices'].int(), 
                                    spatial_shape=[z_size, y_size, x_size], 
                                    batch_size=bs)

        lidar_st = SparseConvTensor(features=ldr_data.reshape((Nvoxels, -1)), 
                                    indices=batch_dict['voxel_coords'].int(), 
                                    spatial_shape=[z_size, y_size, x_size], 
                                    batch_size=bs)
        
        rad_idx = radar_st.indices           # [Nr,4]
        lid_idx = lidar_st.indices           # [Nl,4]

        all_idx = torch.cat([rad_idx, lid_idx], dim=0)
        all_idx = torch.unique(all_idx, dim=0)  # union of occupied voxels
        union_st = scatter_radar_to_union(radar_st, all_idx, [z_size, y_size, x_size], bs)
        
        out = gen_net(union_st)  # SparseConvTensor with logits.features [N_active, K] on same coords as c0
        pred, occ, attrs = out['st'], out['logits'], out['attrs']
        offs = attrs[:, :, :3] 

        voxel_center_xyz = origin + (torch.flip(out['st'].indices[:, 1:4].float(), dims=[1]) + 0.5) * vsize_xyz  # grid center
        pred_offset_m = offs * vsize_xyz.to(d)  # scale voxel-units → meters
        voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 5, 1)
        # print(voxel_center_xyz.shape, pred_offset_m.shape)
        attrs = torch.cat([voxel_center_xyz + pred_offset_m, attrs[:, :, -1][..., None]], dim=-1)
        
        # select valid slots by probability
        prob_thresh=args.thresh
        probs = torch.sigmoid(occ)                 # [N,K]
        keep  = probs >= prob_thresh 
        points_xyz = attrs[keep][:, :3].detach().cpu().numpy().reshape(-1, 3)
        intensity = attrs[keep][:, -1].detach().cpu().numpy().reshape(-1)
        points_xyz = np.ascontiguousarray(points_xyz)
        intensity = np.ascontiguousarray(intensity)
        print(f'points:{points_xyz.shape}, intensity:{intensity.shape}')
        

        # out_tmp = out
        # out_tmp['st'] = union_st
        # Build pred points from your UNet output
        pred_xyz, pred_attr = unet_slots_to_xyz_attrs(
            out,                       # your dict
            voxel_size=[0.05, 0.05, 0.1],   # <-- set to your grid
            origin=origin,       # <-- set to your grid origin
            prob_thresh=prob_thresh,
            clamp_offsets=False
        )

        rdr_points_xyz=np.ascontiguousarray(batch_dict['sp_features'][:, :, :3].detach().cpu().numpy().reshape(-1, 3)) 
        rdr_intensities=np.ascontiguousarray(batch_dict['sp_features'][:, :, -1].detach().cpu().numpy().reshape(-1)) 
        ldr_points_xyz=np.ascontiguousarray(batch_dict['voxels'][:, :, :3].detach().cpu().numpy().reshape(-1, 3))
        ldr_intensities=np.ascontiguousarray(batch_dict['voxels'][:, :, -1].detach().cpu().numpy().reshape(-1))

        # Pick a shared camera pose (e.g., from LiDAR cloud)
        pose = compute_reference_pose(ldr_points_xyz, view="bev")

        # Save all with the SAME pose
        fig_path = os.path.join('visualize', log_sig, load_epoch)
        os.makedirs(fig_path, exist_ok=True)
        list_tuple_objs = batch_dict['meta'][0]['label']
        dx, dy, dz = batch_dict['meta'][0]['calib']
        gt_boxes = []
        for obj in list_tuple_objs:
            cls_name, (x, y, z, th, l, w, h), trk, avail = obj
            x = x + dx
            y = y + dy
            z = z + dz
            print(f'dx, dy, dz: {dx}, {dy}, {dz}')
            gt_boxes.append([cls_name, (x, y, z, th, l, w, h), trk, avail])

        save_open3d_render_fixed_pose(points_xyz=points_xyz, 
                                      intensities=intensity, 
                                      boxes=gt_boxes,
                                      filename=os.path.join(fig_path, f"pred1_{set}_{bi}.png"), 
                                      pose=pose)
        save_open3d_render_fixed_pose(points_xyz=pred_xyz, 
                                      intensities=pred_attr[:,-1], 
                                      boxes=gt_boxes,
                                      filename=os.path.join(fig_path, f"pred2_{set}_{bi}.png"), 
                                      pose=pose)
        save_open3d_render_fixed_pose(rdr_points_xyz, 
                                      intensities=rdr_intensities, 
                                      boxes=gt_boxes,
                                      filename=os.path.join(fig_path, f"rdr_{set}_{bi}.png"),   
                                      pose=pose)
        save_open3d_render_fixed_pose(ldr_points_xyz, 
                                      intensities=ldr_intensities, 
                                      boxes=gt_boxes,
                                      filename=os.path.join(fig_path, f"ldr_{set}_{bi}.png"),  
                                      pose=pose)



        # train_kdataset.vis_in_open3d(batch_dict)
from json import encoder
import os
from random import sample
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.voxel_encoder import SparseVoxelEncoder
from plyfile import PlyData, PlyElement
from datetime import datetime
from tqdm import tqdm

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        if mode != 'train':
            self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], mode)
        else:
            self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = -1

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        
        # voxel parameters
        self.param = argparse.Namespace
        self.param.init_voxel_size = self.conf.get_float('voxel.init_voxel_size')
        self.param.raymarching_step_ratio = self.conf.get_float('voxel.raymarching_step_ratio')
        self.param.max_voxel_hits = self.conf.get_int('voxel.max_voxel_hits')
        self.param.voxel_embedding_dim = self.conf.get_int('voxel.voxel_embedding_dim')
        self.param.debug = False
        
        vox_max = np.ones(3) + 0.2
        vox_min = -np.ones(3) - 0.2
        
        sample_size = self.param.init_voxel_size
        steps = ((vox_max - vox_min) / sample_size).round().astype(np.int32) + 1
        x, y, z = np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]))
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        x, y, z = x * sample_size + vox_min[0], y * sample_size + vox_min[1], z * sample_size + vox_min[2]
        vox_points = np.stack([x, y, z], axis=-1).astype(np.float32) # voxel corner coords
        # vox_points = vox_points[(np.abs(vox_points) < 1.0).sum(-1) == 3]
        
        case_id = -1
        if is_continue or mode != 'train':
             case_id = int(case[8:])
        self.voxel_encoder = SparseVoxelEncoder(self.param, vox_points).to(self.device)
        
        ############################################
        
        # Networks
        self.nerf_outside = None #NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(d_in=self.param.voxel_embedding_dim, 
                                      **self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(d_in=self.param.voxel_embedding_dim, 
                                              d_feature=self.conf.get_int('model.sdf_network.d_hidden'),
                                              **self.conf['model.rendering_network']).to(self.device)
        
        params_to_train = []
        # params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.voxel_encoder.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                self.sdf_network,
                                self.deviation_network,
                                self.color_network,
                                self.voxel_encoder,
                                **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        # if is_continue or mode != 'train':
            # ckpt_dict = {
            #     122: '/home/lihai/project/NeuS/exp/dtu_scan122/wmask_vox/2022_08_02_20_14_38/checkpoints/ckpt_400000.pth',
            #     118: '/home/lihai/project/NeuS/exp/dtu_scan118/wmask_vox/2022_07_03_11_24_43/checkpoints/ckpt_400000.pth',
            #     114: '/home/lihai/project/NeuS/exp/dtu_scan114/wmask_vox/2022_08_03_20_28_46/checkpoints/ckpt_400000.pth',
            #     110: '/home/lihai/project/NeuS/exp/dtu_scan110/wmask_vox/2022_08_10_12_48_31/checkpoints/ckpt_400000.pth',
            #     106: '/home/lihai/project/NeuS/exp/dtu_scan106/wmask_vox/2022_08_09_10_29_25/checkpoints/ckpt_400000.pth',
            #     105: '/home/lihai/project/NeuS/exp/dtu_scan105/wmask_vox/2022_07_31_14_57_54/checkpoints/ckpt_400000.pth',
            #     97: '/home/lihai/project/NeuS/exp/dtu_scan97/wmask_vox/2022_08_04_21_27_35/checkpoints/ckpt_400000.pth',
            #     83: '/home/lihai/project/NeuS/exp/dtu_scan83/wmask_vox/2022_08_10_01_42_29/checkpoints/ckpt_400000.pth',
            #     69: '/home/lihai/project/NeuS/exp/dtu_scan69/wmask_vox/2022_08_08_18_29_44/checkpoints/ckpt_400000.pth',
            #     65: '/home/lihai/project/NeuS/exp/dtu_scan65/wmask_vox/2022_07_31_01_53_13/checkpoints/ckpt_400000.pth',   
            #     63: '/home/lihai/project/NeuS/exp/dtu_scan63/wmask_vox/2022_08_07_13_09_35/checkpoints/ckpt_400000.pth',
            #     55: '/home/lihai/project/NeuS/exp/dtu_scan55/wmask_vox/2022_08_06_23_55_43/checkpoints/ckpt_400000.pth',   
            #     40: '/home/lihai/project/NeuS/exp/dtu_scan40/wmask_vox/2022_07_02_17_01_59/checkpoints/ckpt_400000.pth',
            #     37: '/home/lihai/project/NeuS/exp/dtu_scan37/wmask_vox/2022_08_11_15_55_38/checkpoints/ckpt_400000.pth',
            #     24: '/home/lihai/project/NeuS/exp/dtu_scan24/wmask_vox/2022_08_05_18_28_11/checkpoints/ckpt_400000.pth',     
            # }
            # latest_model_name = ckpt_dict[case_id]

        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
            
    def reset_optimizer(self):
        params_to_train = []
        # params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.voxel_encoder.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        
        encoder_state = self.voxel_encoder.precompute()

        for iter_i in tqdm(range(res_step)):
            ## division
            if self.iter_step in [2e4, 5e4, 10e4, 20e4, 30e4]:
                print('Voxel spliting at {} step.'.format(self.iter_step))
                plydata = self.voxel_encoder.export_voxels(return_mesh=True)
                plydata.write(open(f'before_{self.iter_step}.ply', 'wb'))
                
                self.voxel_encoder.pruning(self.sdf_network.sdf, th=0.02)
                self.voxel_encoder.splitting(split_step=self.iter_step>2e4)
                self.voxel_encoder.pruning(self.sdf_network.sdf, th=0.02)
                encoder_state = self.voxel_encoder.precompute()
                
                plydata = self.voxel_encoder.export_voxels(return_mesh=True)
                plydata.write(open(f'after_{self.iter_step}.ply', 'wb'))
                
                self.reset_optimizer()
                print("************ prune & split ***********")
                
                # self.igr_weight *= 0.9
            
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far, encoder_state,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              iter_step=self.iter_step)

            color_fine = render_out['color_fine']
            depth_fine = render_out['depth_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            
            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            if self.iter_step % 10 == 0:
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(is_train=True)

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        # self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        
        vox_size = checkpoint['voxel_encoder_fine']['points'].shape[0]
        emd_size = checkpoint['voxel_encoder_fine']['values.weight'].shape[0]
        self.voxel_encoder.modify_voxel_params(vox_size, emd_size)

        self.voxel_encoder.load_state_dict(checkpoint['voxel_encoder_fine'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']


        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            # 'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'voxel_encoder_fine': self.voxel_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            # idx = np.random.randint(self.dataset.n_images)
            idx = 7

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size // 8)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size // 8)
        encoder_state = self.voxel_encoder.precompute()

        out_rgb_fine = []
        out_depth_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              encoder_state,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              iter_step=self.iter_step)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('depth_fine'):
                out_depth_fine.append(render_out['depth_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'][:, :n_samples] * render_out['weights'][:, :n_samples, None]
                
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        
        depth_fine = None
        if len(out_depth_fine) > 0:
            depth_fine = (np.concatenate(out_depth_fine, axis=0).reshape([H, W, 1, -1]) * 10).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'depths'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_depth_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                    'depths',
                                    '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            depth_fine[..., i])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, is_train=False, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        encoder_state = self.voxel_encoder.precompute()

        # from plyfile import PlyData, PlyElement
        # plydata = self.voxel_encoder.export_voxels(return_mesh=True)
        # plydata.write(open(f'{self.iter_step}.ply', 'wb'))
        
        # from unisurf.extracting import Extractor3D            
        # extractor = Extractor3D(self.voxel_encoder, self.sdf_network.sdf, device=self.device, padding=self.param.init_voxel_size * 0.5, 
        #                         resolution0= (16 / (self.param.init_voxel_size / self.voxel_encoder.voxel_size)).int() )
        # mesh, stats_dict = extractor.generate_mesh(encoder_state)
        # mesh.export(f'surface_{self.iter_step}.obj')

        # import numpy as np
        # from pyvox.models import Vox
        # from pyvox.writer import VoxWriter

        # vox_center = encoder_state['voxel_center_xyz'].cpu().numpy()
        # voxel_size = self.voxel_encoder.voxel_size.cpu().numpy()
        # min_coord = vox_center.min(axis=0)
        # max_coord = vox_center.max(axis=0)
        
        # voxel_shape = ((max_coord - min_coord) / voxel_size).round().astype(np.int) + 1
        # print(voxel_shape.dtype)
        # vox_grid = np.mgrid[0:voxel_shape[0], 0:voxel_shape[1], 0:voxel_shape[2]]
        # vox_grid = vox_grid.reshape(3, -1)
        # vox_center_scaled = ((vox_center - min_coord) / voxel_size).round()
        # vox_center_id = vox_center_scaled[:, 0] * voxel_shape[1] * voxel_shape[2] + vox_center_scaled[:, 1] * voxel_shape[2] + vox_center_scaled[:, 2]
        # vox_center_id = vox_center_id.astype(np.int32)

        # a = np.zeros_like(vox_grid[0])
        # a[vox_center_id] = 1
        # a = a.reshape(voxel_shape[0], voxel_shape[1], voxel_shape[2])
        # a = a == 1
        
        # vox = Vox.from_dense(a)
        # VoxWriter('test.vox', vox).write()
        # exit()
        
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, encoder_state, resolution=resolution, threshold=threshold)
        if is_train: 
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if is_train:
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        else:
            mesh.export(os.path.join(self.base_exp_dir, 'final.ply'))

        logging.info('End')
        
    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def validate_psnr(self, resolution_level=1, psnr_file=None):
        os.makedirs(os.path.join(self.base_exp_dir, 'psnr'), exist_ok=True)
        if psnr_file is None:
            psnr_file = os.path.join(self.base_exp_dir, 'psnr', '{}_psnr_{:0>8d}.txt'.format(resolution_level, self.iter_step))

        psnrs = []
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        encoder_state = self.voxel_encoder.precompute()
        encoder_state['voxel_visible_count'] = torch.zeros_like(encoder_state['voxel_center_xyz'][:, 0]).long()

        for idx in tqdm(range(self.dataset.n_images)):
            if not os.path.exists(os.path.join(self.base_exp_dir, 'psnr', f'{idx}_cache.npy')):
                
                print('====> eval psnr of {}-th file.'.format(idx))
                rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
                H, W, _ = rays_o.shape

                rays_o = rays_o.reshape(-1, 3).split(self.batch_size // 4)
                rays_d = rays_d.reshape(-1, 3).split(self.batch_size // 4)

                out_rgb_fine = []

                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                    background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                    render_out = self.renderer.render(rays_o_batch,
                                                    rays_d_batch,
                                                    near,
                                                    far,
                                                    encoder_state,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                    background_rgb=background_rgb, 
                                                    iter_step=self.iter_step,
                                                    is_train=False
                                                    )

                    def feasible(key): return (key in render_out) and (render_out[key] is not None)

                    if feasible('color_fine'):
                        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                    del render_out

                img_fine = None
                if len(out_rgb_fine) > 0:
                    img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])).clip(0, 1)
                    np.save(os.path.join(self.base_exp_dir, 'psnr', f'{idx}.npy'), img_fine)
                    img_fine = torch.from_numpy(img_fine) 
            else:
                img_fine = np.load(os.path.join(self.base_exp_dir, 'psnr', f'{idx}.npy'))
                img_fine = torch.from_numpy(img_fine)

            true_rgb = self.dataset.images[idx]
            mask = self.dataset.masks[idx]
            mask_sum = mask.sum() + 1e-7

            # import cv2
            # cv2.imwrite('img_fine.png', img_fine.numpy()*255)
            # cv2.imwrite('true_rgb.png', true_rgb.numpy()*255)
            # exit()

            psnr = 20.0 * torch.log10(1.0 / (((img_fine - true_rgb)**2 * mask).sum() / (mask_sum)).sqrt())
            print('{} psnr:'.format(idx), psnr)
            psnrs.append(psnr)

        keep_ids = torch.arange(self.voxel_encoder.keep.shape[0])
        invalid_ids = keep_ids[self.voxel_encoder.keep.bool()][encoder_state['voxel_visible_count'] < 2]
        self.voxel_encoder.keep[invalid_ids] = False

        # plydata = self.voxel_encoder.export_voxels(return_mesh=True, visible_mask=encoder_state['voxel_visible_count']>0)
        # plydata.write(open(f'after_{self.iter_step}.ply', 'wb'))

        # print
        print('PSNRs: ', psnrs)
        print('Means: ', np.mean(psnrs))
        psnr_file_obj = open(psnr_file, 'a')
        for i in range(len(psnrs)):
            psnr_file_obj.write(str(psnrs[i].numpy()) + ' ')
        psnr_file_obj.write('\nMean: ' + str(np.mean(psnrs)) + '\n')
        psnr_file_obj.close()
        return psnrs

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'psnr':
        runner.validate_psnr()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)

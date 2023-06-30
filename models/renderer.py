from json import encoder
from re import L
from cv2 import _INPUT_ARRAY_STD_VECTOR_CUDA_GPU_MAT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
import open3d
import time

MAX_DEPTH = 100.0

def write_obj(obj_file, verts, faces=None):
    assert len(verts.shape) == 2
    assert verts.shape[1] == 3

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(verts)
    if faces is not None:
        assert len(faces.shape) == 2
        assert faces.shape[1] == 3
        mesh.triangles = open3d.utility.Vector3iVector(faces)
    open3d.io.write_triangle_mesh(obj_file, mesh)

def masked_scatter(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)


def extract_fields(bound_min, bound_max, resolution, encoder_states, input_fn, field_fn):
    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    emb, valid_mask = input_fn(pts, encoder_states)
                    val = field_fn(emb)
                    # if valid_mask is not None:
                    #     val[~valid_mask] = 1
                    val = val.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = -val
    return u


def extract_geometry(bound_min, bound_max, encoder_states, resolution, threshold, input_fn, field_fn, voxel_size):
    print('threshold: {}'.format(threshold))

    # print(bound_min, bound_max)
    # for k,v in encoder_states.items():
    #     print(k, v.shape)
    # print(resolution, threshold)
    
    voxel_centers = encoder_states['voxel_center_xyz']
    n_v = voxel_centers.shape[0]

    vertices_list = []
    triangles_list = []
    triangles_counts = 0
    resolution = 10
    for v in range(n_v):
        center = voxel_centers[v]
        bound_min = center - 0.5 * voxel_size
        bound_max = center + 0.5 * voxel_size
        u = extract_fields(bound_min, bound_max, resolution, encoder_states, input_fn, field_fn)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        
        vertices_list.append(vertices)
        triangles_list.append(triangles + triangles_counts)
        triangles_counts += vertices.shape[0]

    vertices = np.concatenate(vertices_list)
    triangles = np.concatenate(triangles_list)

    print(vertices.shape, triangles.shape)

    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 voxel_encoder,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.voxel_encoder = voxel_encoder
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)
        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    @torch.no_grad()
    def find_surface(self, sdf, ray_start, ray_dir, samples, intersection_outputs, encoder_states, first_only=False):
        # sdf: R x S  
        R, S, _ = sdf.shape
        sdf = sdf.view(R, S)
        
        depth = samples['sampled_point_depth']
        voxel_id = samples['sampled_point_voxel_idx']

        # Create mask for valid points where the first point outside the surface
        mask_not_occupied = sdf[:, 0] > 0
        # Calculate if sign change occurred and concat 1 (no sign change) in last dimension
        sign_matrix = torch.cat([torch.sign(sdf[:, :-1] * sdf[:, 1:]), 
                                torch.ones(R, 1).to(sdf.device)],
                                dim=-1)

        if first_only:
            cost_matrix = sign_matrix * torch.arange(S, 0, -1).float().to(sign_matrix.device)
            # Get first sign change and mask for values where a.) a sign changed
            # occurred and b.) no a neg to pos sign change occurred (meaning from
            # inside surface to outside)
            values, indices = torch.min(cost_matrix, -1)
            mask_sign_change = values < 0
            mask_neg_to_pos = sdf[torch.arange(R), indices] > 0
            # Define mask where a valid depth value is found
            mask = mask_sign_change & mask_neg_to_pos & mask_not_occupied

            # temerate for render 
            mask = mask_sign_change

            out_voxel = voxel_id[torch.arange(R), indices]
            next_indices = torch.clamp(indices + 1, max=S-1)
            in_voxel = voxel_id[torch.arange(R), next_indices]
            
            voxels = torch.stack([out_voxel, in_voxel], dim=-1)
            voxels = torch.where(mask.unsqueeze(-1), voxels, torch.full_like(voxels, -2))

            if 'voxel_visible_count' in encoder_states.keys():
                indices = voxels[voxels != -2].long()
                encoder_states['voxel_visible_count'].index_add_(0, indices, torch.ones_like(indices))
        else:
            voxel_start_mask = torch.logical_and(sign_matrix < 0, sdf > 0)
            voxel_end_mask = torch.cat([torch.zeros_like(voxel_start_mask)[:, :1],  voxel_start_mask[:, :-1]], dim=-1)
            voxel_mask = torch.logical_or(voxel_start_mask, voxel_end_mask)

            voxels = torch.where(voxel_mask, voxel_id, torch.full_like(voxel_id, -2))
            mask = torch.logical_and(mask_not_occupied, voxel_mask.any(-1))
            voxels = torch.where(mask.unsqueeze(-1), voxels, torch.full_like(voxels, -2))

        return voxels, mask
        
    @torch.no_grad()
    def selct_interval(self, surface_voxels, intersection_outputs, step_size, multiple=10.0, fix_step=64):
        t_pts_idx = intersection_outputs['intersected_voxel_idx']
        t_probs = intersection_outputs['probs']
        t_steps = intersection_outputs['steps']

        interval_intersection = {}
        interval_intersection['min_depth'] = intersection_outputs['min_depth']
        interval_intersection['max_depth'] = intersection_outputs['max_depth']
        interval_intersection['intersected_voxel_idx'] = intersection_outputs['intersected_voxel_idx']
        
        # important_mask = torch.logical_or(t_pts_idx == surface_voxels[:, [0]], t_pts_idx == surface_voxels[:, [1]])
        important_mask = torch.zeros_like(t_pts_idx) == 1
        for i in range(surface_voxels.shape[-1]):
            important_mask = torch.logical_or(important_mask, t_pts_idx == surface_voxels[:, [i]])

        t_probs[important_mask] *= multiple
        interval_intersection['steps'] = t_steps 

        if multiple < 0.0:
            t_probs[~important_mask] *= 0.0
            # print(surface_voxels[0])
            interval_intersection['steps'] = torch.full_like(t_steps, fix_step) * important_mask.sum(-1) # steps for each ray
            # t_pts_idx[~important_mask] = -1
            # interval_intersection['intersected_voxel_idx'] = t_pts_idx
        interval_intersection['probs'] = t_probs / t_probs.sum(dim=-1, keepdim=True) # uniform distribution

        return interval_intersection

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    encoder_state, 
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    iter_step=0,
                    is_train=True):
        
        ###########################################
        
        ray_start = rays_o.unsqueeze(0)
        ray_dir = rays_d.unsqueeze(0)   
        
        start = time.time()
        ray_start, ray_dir, intersection_outputs, hits = self.voxel_encoder.ray_intersect(ray_start, 
                                                                                        ray_dir, 
                                                                                        encoder_state)
        # print('ray_intersect: ', time.time() - start)     


        dists = (intersection_outputs['max_depth'] - intersection_outputs['min_depth']
                ).masked_fill(intersection_outputs['intersected_voxel_idx'].eq(-1), 0)
        intersection_outputs['probs'] = dists / dists.sum(dim=-1, keepdim=True)
        intersection_outputs['steps'] = dists.sum(-1) / self.voxel_encoder.step_size

        if hits.sum() != 0 and intersection_outputs['steps'].sum() != 0:
            samples = self.voxel_encoder.ray_sample(intersection_outputs)

            sampled_depth = samples['sampled_point_depth']
            sampled_idx = samples['sampled_point_voxel_idx']
            sample_mask = sampled_idx.ne(-1)
            
            batch_size, n_samples = sampled_idx.shape
    
            sampled_xyz = ray_start.unsqueeze(1) + ray_dir.unsqueeze(1) * sampled_depth.unsqueeze(2)
            sampled_dir = ray_dir.unsqueeze(1).expand(*sampled_depth.shape, ray_dir.shape[-1])
            
            samples['sampled_point_xyz'] = sampled_xyz
            samples['sampled_point_ray_direction'] = sampled_dir
            
            dists = samples['sampled_point_distance'].unsqueeze(-1)
            masked_samples = {name: s[sample_mask] for name, s in samples.items()}
            
            start = time.time() 
            field_inputs = self.voxel_encoder(masked_samples, encoder_state)
            # print('voxel_encoder: ', time.time() - start)   
            pts = field_inputs['pos']
            emb = field_inputs['emb']
            dirs = field_inputs['ray']

            start = time.time() 
            sdf_nn_output = sdf_network(emb)
            # print('sdf_network: ', time.time() - start) 

            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]

            if iter_step > 10e4:
                sdf = masked_scatter(sample_mask, sdf)
                start = time.time() 
                surface_voxels, surafce_mask = self.find_surface(sdf, ray_start, ray_dir, samples, intersection_outputs, encoder_state, first_only=iter_step > 29e4)
                # print('find_surface: ', time.time() - start)    
                if surafce_mask.sum() > 0:
                    start = time.time() 
                    interval_intersection = self.selct_interval(surface_voxels, intersection_outputs, self.voxel_encoder.step_size)
                    # print('selct_interval: ', time.time() - start)    
                    
                    start = time.time() 
                    interval_samples = self.voxel_encoder.ray_sample(interval_intersection)
                    # print('ray_sample: ', time.time() - start)   
                    
                    # display_points = samples['sampled_point_xyz'][surafce_mask][:10].reshape(-1, 3)
                    # display_points = display_points[(torch.abs(display_points) < 4.0).all(-1)]
                    # write_obj('sampled_init.obj', display_points.detach().cpu().numpy())
                    
                    samples = interval_samples
                        
                    sampled_depth = samples['sampled_point_depth']
                    sampled_idx = samples['sampled_point_voxel_idx']
                    sample_mask = sampled_idx.ne(-1)
                    
                    batch_size, n_samples = sampled_idx.shape
            
                    sampled_xyz = ray_start.unsqueeze(1) + ray_dir.unsqueeze(1) * sampled_depth.unsqueeze(2)
                    sampled_dir = ray_dir.unsqueeze(1).expand(*sampled_depth.shape, ray_dir.shape[-1])
                    samples['sampled_point_xyz'] = sampled_xyz

                    # print(interval_intersection['intersected_voxel_idx'][surafce_mask][0])
                    # print(interval_intersection['probs'][surafce_mask][0])
                    # print(sampled_depth[surafce_mask][0])
                    # print(sampled_idx[surafce_mask][0])
                    
                    # display_points = samples['sampled_point_xyz'][surafce_mask][:10].reshape(-1, 3)
                    # display_points = display_points[(torch.abs(display_points) < 4.0).all(-1)]
                    # write_obj('sampled_re.obj', display_points.detach().cpu().numpy())

                    samples['sampled_point_ray_direction'] = sampled_dir
                    
                    dists = samples['sampled_point_distance'].unsqueeze(-1)
                    masked_samples = {name: s[sample_mask] for name, s in samples.items()}
                    
                    start = time.time() 
                    field_inputs = self.voxel_encoder(masked_samples, encoder_state)
                    # print('voxel_encoder: ', time.time() - start)    

                    pts = field_inputs['pos']
                    emb = field_inputs['emb']
                    dirs = field_inputs['ray']
            
                    sdf_nn_output = sdf_network(emb)
                    sdf = sdf_nn_output[:, :1]
                    feature_vector = sdf_nn_output[:, 1:]
            
            start = time.time() 
            gradients = sdf_network.gradient(emb, pts).squeeze(1)
            # gradients = dirs.clone()
            # print('gradient: ', time.time() - start)  
            if not is_train:
                gradients = gradients.detach()
            start = time.time() 
            sampled_color = color_network(emb, gradients, dirs, feature_vector)
            # print('color_network: ', time.time() - start)  
            
            ######################################
            sampled_color = masked_scatter(sample_mask, sampled_color)
            sdf = masked_scatter(sample_mask, sdf)
            gradients = masked_scatter(sample_mask, gradients)
            dirs = sampled_dir
            pts = sampled_xyz
            
            sampled_color.reshape(batch_size, n_samples, 3)
            
            ######################################

            start = time.time() 

            inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1)
            
            true_cos = (dirs * dirs).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf.reshape(-1, 1) + iter_cos.reshape(-1, 1) * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf.reshape(-1, 1) - iter_cos.reshape(-1, 1) * dists.reshape(-1, 1) * 0.5

            inv_s = inv_s
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-7) / (c + 1e-7)).reshape(batch_size, n_samples).clip(0.0, 1.0)

            pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
            inside_sphere = (pts_norm < 1.0).float().detach()
            relax_inside_sphere = (pts_norm < 1.2).float().detach()

            # Render with background
            if background_alpha is not None:
                alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
                alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
                sampled_color = sampled_color * inside_sphere[:, :, None] +\
                                background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
                sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            weights_sum = weights.sum(dim=-1, keepdim=True)

            # print(weights_sum.squeeze())
            # weight_mask = weights_sum.squeeze() < 9e-1
            # weights[weight_mask] = 0
    
            color = (sampled_color * weights[:, :, None]).sum(dim=1)

            # print('volume render: ', time.time() - start)  

            depth = (sampled_depth * weights).sum(dim=1)

            depth = sampled_depth.min(dim=-1)[0]
            # depth[~surafce_mask] = 100
            # print(depth.shape, surafce_mask.shape, (~surafce_mask).sum())

            if background_rgb is not None:    # Fixed background, usually black
                color = color + background_rgb * (1.0 - weights_sum)

            # Eikonal loss
            gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
            
            gradients = gradients.reshape(batch_size, n_samples, 3)
            s_val = 1.0 / inv_s.reshape(batch_size, n_samples)
            cdf = c.reshape(batch_size, n_samples)
        
        else:
            batch_size = ray_dir.shape[0]
            n_samples = 1
            color = torch.zeros(batch_size, 3).to(ray_start.device)
            depth = torch.zeros(batch_size).to(ray_start.device)
            sdf = torch.zeros(batch_size, n_samples, 1).to(ray_start.device)
            dists = torch.zeros(batch_size, n_samples, 1).to(ray_start.device)
            gradients = torch.ones(batch_size, n_samples, 3).to(ray_start.device)
            s_val = torch.ones(batch_size, n_samples).to(ray_start.device)
            weights = torch.ones(batch_size, n_samples).to(ray_start.device)
            cdf = torch.zeros(batch_size, n_samples).to(ray_start.device)
            gradient_error = torch.tensor([0.0])
            inside_sphere = torch.zeros(batch_size, n_samples).to(ray_start.device)


        return {
            'color': color,
            'depth': depth,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients,
            's_val': s_val,
            # 'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': cdf,
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, encoder_state, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, iter_step=0, is_train=True):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Render core
        start = time.time()
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    encoder_state,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    iter_step=iter_step,
                                    is_train=is_train)
        # print('render_core: ', time.time() - start)

        # for k,v in ret_fine.items():
        #     print(k, v.shape)
        
        color_fine = ret_fine['color']
        depth_fine = ret_fine['depth']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].mean(dim=-1, keepdim=True)
        
        return {
            'color_fine': color_fine,
            'depth_fine': depth_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }


    def extract_geometry(self, bound_min, bound_max, encoder_states, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                encoder_states,
                                resolution=resolution,
                                threshold=threshold,
                                input_fn=self.voxel_encoder.get_point_embedding,
                                field_fn=self.sdf_network.sdf,
                                voxel_size=self.voxel_encoder.voxel_size
                                )

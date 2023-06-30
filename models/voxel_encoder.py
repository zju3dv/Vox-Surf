from wave import Wave_write
import torch
import torch.nn as nn
import torch.nn.functional as F
from nsvf.ext import build_octree
from nsvf_ext import svo_ray_intersect, inverse_cdf_sampling
from plyfile import PlyData, PlyElement
import numpy as np
import open3d

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

def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c])
    offset = (torch.cat([
                    ox.reshape(-1, 1), 
                    oy.reshape(-1, 1), 
                    oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel
    return offset.type_as(point_xyz) * quarter_voxel

def trilinear_interp(p, q, point_feats):
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)
    point_feats = (weights * point_feats).sum(1)

    return point_feats

def splitting_points(point_xyz, point_feats, values, half_voxel):
    
    # generate new centers
    quarter_voxel = half_voxel * .5
    new_points = offset_points(point_xyz, quarter_voxel).reshape(-1, 3)
    min_vox_point = point_xyz.min(dim=0)[0]
    old_coords = ((point_xyz - min_vox_point) / quarter_voxel).round_().int() 
    new_coords = offset_points(old_coords).reshape(-1, 3)
    new_keys0  = offset_points(new_coords).reshape(-1, 3) 
    
    # get unique keys and inverse indices (for original key0, where it maps to in keys)
    new_keys, new_feats = torch.unique(new_keys0, dim=0, sorted=True, return_inverse=True)
    new_keys_idx = new_feats.new_zeros(new_keys.size(0)).scatter_(
        0, new_feats, torch.arange(new_keys0.size(0), device=new_feats.device) // 64)
    
    # recompute key vectors using trilinear interpolation 
    new_feats = new_feats.reshape(-1, 8)
    
    if values is not None:
        p = (new_keys - old_coords[new_keys_idx]).type_as(point_xyz).unsqueeze(1) * .25 + 0.5 # (1/4 voxel size)
        q = offset_points(p, .5, offset_only=True).unsqueeze(0) + 0.5   # BUG?
        point_feats = point_feats[new_keys_idx]
        point_feats = F.embedding(point_feats, values).view(point_feats.size(0), -1)
        new_values = trilinear_interp(p, q, point_feats)
    else:
        new_values = None
    return new_points, new_feats, new_values, new_keys

class SparseVoxelEncoder(nn.Module):
    def __init__(self, param, vox_points):
        super(SparseVoxelEncoder, self).__init__()
        self.param = param

        vox_points = torch.from_numpy(vox_points) # raw voxel coords (float)
        init_voxel_size = self.param.init_voxel_size
        half_voxel_size = init_voxel_size * 0.5

        min_vox_point = vox_points.min(dim=0)[0]
        vox_coords = ((vox_points - min_vox_point) / half_voxel_size).round_().int() # resized voxel coords (int, distance 2)
        residual = (vox_points - vox_coords.type_as(vox_points) * half_voxel_size).mean(0, keepdim=True) # average offset

        bits = 2 # split n directions on each dimension
        c = torch.arange(1, 2 * bits, 2)
        ox, oy, oz = torch.meshgrid([c, c, c])
        offset = (torch.cat([ox.reshape(-1, 1), 
                            oy.reshape(-1, 1), 
                            oz.reshape(-1, 1)], dim=1).type_as(vox_coords) - bits) / float(bits - 1) # bits^3 directions in [-1, 1]

        octree_coords = vox_coords.unsqueeze(1) + offset.unsqueeze(0).type_as(vox_coords) # 8 nearby voxel center for each resized voxel coords (integer)
        octree_coords = octree_coords.reshape(-1, 3) # O x 3
        octree_coords, octree_coord_ids = torch.unique(octree_coords, dim=0, sorted=True, return_inverse=True) # voxel centers (integer)
        octree_coord_ids = octree_coord_ids.reshape(-1, 8) # O x 8 

        num_octree_coords = torch.scalar_tensor(octree_coords.shape[0]).long()
        raymarching_step_size = self.param.raymarching_step_ratio * init_voxel_size

        # register parameters (will be saved to checkpoints)
        self.register_buffer("points", vox_points)          # voxel centers
        self.register_buffer("keys", octree_coords.long())       # id used to find voxel corners/embeddings
        self.register_buffer("feats", octree_coord_ids.long())     # for each voxel, 8 voxel corner ids
        self.register_buffer("num_keys", num_octree_coords)
        self.register_buffer("keep", octree_coord_ids.new_ones(octree_coord_ids.shape[0]).long())  # whether the voxel will be pruned

        self.register_buffer("voxel_size", torch.scalar_tensor(init_voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(raymarching_step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(self.param.max_voxel_hits))

        print("num voxels: ", self.points.shape)
        print("num octree nodes:", self.keys.shape)

        write_obj('vox_points.obj', self.points.detach().cpu().numpy())
        write_obj('octree_coords.obj', (self.keys  * half_voxel_size + residual).detach().cpu().numpy())
        
        # set-up other hyperparameters and initialize running time caches
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
        }

        # sparse voxel embeddings
        self.embed_dim = self.param.voxel_embedding_dim
        self.values = nn.Embedding(self.num_keys, self.embed_dim)
        nn.init.normal_(self.values.weight, mean=0, std=self.embed_dim ** -0.5)
        
    # temporal for checkpoint
    def modify_voxel_params(self, vox_size, emb_size):
        self.register_buffer("points", torch.ones(vox_size, 3).float())
        self.register_buffer("feats", torch.ones(vox_size, 8).long())
        self.register_buffer("keep", torch.ones(vox_size).long())
        self.values = nn.Embedding(emb_size, self.embed_dim)

        self.step_size *= 0.2
        
    def reset_runtime_caches(self):
        points = self.points[self.keep.bool()]

        if points.shape[0] == 1:
            centers = points
            children = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, 1]).to(points.device)
        else:
            half_voxel_size = self.voxel_size * 0.5
            min_vox_point = points.min(dim=0)[0]
            coords = ((points - min_vox_point) / half_voxel_size).round_().int() # resized voxel coords (int, distance 2)
            residual = (points - coords.type_as(points) * half_voxel_size).mean(0, keepdim=True) # average offset

            ranges = coords.max(0)[0] - coords.min(0)[0]
            depths = torch.log2(ranges.max().float()).ceil_().long() - 1 # depth of octree

            center = (coords.max(0)[0] + coords.min(0)[0]) / 2

            centers, children = build_octree(center, coords, depths)
            centers = centers.float() * half_voxel_size + residual

        self._runtime_caches['flatten_centers'] = centers
        self._runtime_caches['flatten_children'] = children

    # @torch.no_grad()
    def precompute(self, build_octree=True):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        values = self.values.weight[:self.num_keys]

        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if build_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children

        return encoder_states
    
    @torch.no_grad()
    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']

        B, P, _ = ray_dir.shape
        H, D = point_feats.shape

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(B * P, 3).contiguous()
        ray_dir = ray_dir.reshape(B * P, 3).contiguous()

        # ray-voxel intersection with SVO
        flatten_centers = encoder_states['voxel_octree_center_xyz']
        flatten_children = encoder_states['voxel_octree_children_idx']

        pts_idx, min_depth, max_depth = svo_ray_intersect(
            self.voxel_size, self.max_hits, flatten_centers, flatten_children, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)  
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits
    
    @torch.no_grad()
    def ray_sample(self, intersection_outputs, depth_clip=-1):
        # sample points and use middle point approximation
        
        sampled_idx, sampled_min_depth, sampled_max_depth, sampled_depth, sampled_dists = inverse_cdf_sampling(
            intersection_outputs['intersected_voxel_idx'],
            intersection_outputs['min_depth'], 
            intersection_outputs['max_depth'], 
            intersection_outputs['probs'],
            intersection_outputs['steps'], 
            -1)
        
        sampled_dists = sampled_dists.clamp(min=0.0)
            
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)
        
        samples = {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,
        }
        return samples

    def get_point_embedding(self, sampled_xyz, encoder_states, point_voxle_idx=None):
        # encoder states
        point_feats = encoder_states['voxel_vertex_idx']
        point_xyz = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']

        valid_mask = None
        if point_voxle_idx is None:
            dists = sampled_xyz.unsqueeze(1) - point_xyz.unsqueeze(0)
            norm_dists = dists.norm(dim=-1)
            _, point_voxle_idx = torch.min(norm_dists, dim=1)
            # valid_mask = point_voxle_dist <= 0.5 * self.voxel_size
            
            min_dists = dists[torch.arange(dists.size(0)), point_voxle_idx]
            valid_mask = (min_dists.abs() <= 0.5 * self.voxel_size).all(-1)

        # if self.param.debug and point_voxle_idx is not None:
        #     dists = (sampled_xyz.unsqueeze(1) - point_xyz.unsqueeze(0)).norm(dim=-1)
        #     index = torch.min(dists, dim=1)[1]
        #     index_mask = point_voxle_idx != index
        #     if index_mask.sum() > 0:
        #         print(point_voxle_idx[index_mask])
        #         print(index[index_mask])
        #         print(sampled_xyz[index_mask])
        #         print(point_xyz[index[index_mask].long()])
        #         print(point_xyz[point_voxle_idx[index_mask].long()])

        # resample point features
        point_xyz = F.embedding(point_voxle_idx, point_xyz)
        point_feats = F.embedding(F.embedding(point_voxle_idx, point_feats), values).view(point_xyz.size(0), -1)

        # tri-linear interpolation
        p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
        q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5
        embedding = trilinear_interp(p, q, point_feats)

        return embedding, valid_mask

    def forward(self, samples, encoder_states):
        # encoder states
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']

        # print(encoder_states['voxel_vertex_emb'].requires_grad)
        # print(values.sum())
        
        # ray point samples
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        sampled_xyz = samples['sampled_point_xyz'].requires_grad_(True)
        sampled_dir = samples['sampled_point_ray_direction']
        sampled_dis = samples['sampled_point_distance']

        # prepare inputs for implicit field
        #  / self.scene_scale
        inputs = {
            'pos': sampled_xyz, 
            'ray': sampled_dir, 
            'dists': sampled_dis}

        # resample point features
        point_xyz = F.embedding(sampled_idx, point_xyz)
        point_feats = F.embedding(F.embedding(sampled_idx, point_feats), values).view(point_xyz.size(0), -1)

        # tri-linear interpolation
        p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
        q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5
        inputs['emb'] = trilinear_interp(p, q, point_feats)

        return inputs

    @torch.no_grad()
    def pruning(self, field_fn, th=0.0):
        keep = self.get_scores(field_fn, th, bits=20)
        # keep = torch.logical_and((scores < th).sum(-1) > 0, (scores > th).sum(-1) > 0)
        # keep = (scores.abs() < th).sum(-1) > 0
        self.keep.masked_scatter_(self.keep.bool(), keep.long())
        
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
        }
    
    @torch.no_grad()
    def get_scores(self, field_fn, th, bits=16):
        encoder_states = self.precompute()
        
        feats = encoder_states['voxel_vertex_idx'] 
        points = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']
        chunk_size = 1

        def get_scores_once(feats, points, values):
            # sample points inside voxels
            sampled_xyz = offset_points(points, self.voxel_size / 2.0, bits=bits)
            sampled_xyz = sampled_xyz.reshape(-1, 3)
            field_inputs, _ = self.get_point_embedding(sampled_xyz, encoder_states)
            sdf = field_fn(field_inputs)   
            
            sdf[torch.sqrt((sampled_xyz ** 2).sum(-1)) > 1.0] = 1.0
            sdf = sdf.reshape(-1, bits ** 3)  

            keep = (sdf.abs() < th).sum(-1) > 0
            
            return keep

        return torch.cat([get_scores_once(feats[i: i + chunk_size], points[i: i + chunk_size], values) 
            for i in range(0, points.size(0), chunk_size)], 0)

    @torch.no_grad()
    def splitting(self, split_step=False):
        encoder_states = self.precompute()

        feats, points, values = encoder_states['voxel_vertex_idx'], encoder_states['voxel_center_xyz'], encoder_states['voxel_vertex_emb']
        new_points, new_feats, new_values, new_keys = splitting_points(points, feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)
        
        # set new voxel embeddings
        if new_values is not None:
            self.values.weight = nn.Parameter(new_values)
            self.values.num_embeddings = self.values.weight.size(0)
        
        self.total_size = new_num_keys
        self.num_keys = self.num_keys * 0 + self.total_size

        self.points = new_points
        self.feats = new_feats
        self.keep = self.keep.new_ones(new_point_length)

        self.voxel_size *= 0.5
        self.max_hits *= 1.2
        if split_step:
            self.step_size *= 0.5

        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
        }

    @property
    def flatten_centers(self):
        if self._runtime_caches['flatten_centers'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_centers']
    
    @property
    def flatten_children(self):
        if self._runtime_caches['flatten_children'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_children']

    @torch.no_grad()
    def export_voxels(self, return_mesh=False, visible_mask=None):
        voxel_idx = torch.arange(self.keep.size(0), device=self.keep.device)
        voxel_idx = voxel_idx[self.keep.bool()]
        voxel_pts = self.points[self.keep.bool()]

        if visible_mask is not None:
            voxel_idx = voxel_idx[visible_mask]
            voxel_pts = voxel_pts[visible_mask]

        if not return_mesh:
            # HACK: we export the original voxel indices as "quality" in case for editing
            points = [
                (voxel_pts[k, 0], voxel_pts[k, 1], voxel_pts[k, 2], voxel_idx[k])
                for k in range(voxel_idx.size(0))
            ]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])
            return PlyData([PlyElement.describe(vertex, 'vertex')])
        else:
            # generate polygon for voxels            
            voxel_size = self.voxel_size / 2
            minimal_voxel_point = voxel_pts.min(dim=0, keepdim=True)[0]
            center_coords = ((voxel_pts - minimal_voxel_point) / voxel_size).round_().long()  # float
            residual = (voxel_pts - center_coords.type_as(voxel_pts) * voxel_size).mean(0, keepdim=True)

            offsets = torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1]], device=center_coords.device)
            vertex_coords = center_coords[:, None, :] + offsets[None, :, :]
            vertex_points = vertex_coords.type_as(residual) * self.voxel_size / 2 + residual
            
            faceidxs = [[1,6,7,5],[7,6,2,4],[5,7,4,3],[1,0,2,6],[1,5,3,0],[0,3,4,2]]
            all_vertex_keys, all_vertex_idxs  = {}, []
            for i in range(vertex_coords.shape[0]):
                for j in range(8):
                    key = " ".join(["{}".format(int(p)) for p in vertex_coords[i,j]])
                    if key not in all_vertex_keys:
                        all_vertex_keys[key] = vertex_points[i,j]
                        all_vertex_idxs += [key]
            all_vertex_dicts = {key: u for u, key in enumerate(all_vertex_idxs)}
            all_faces = torch.stack([torch.stack([vertex_coords[:, k] for k in f]) for f in faceidxs]).permute(2,0,1,3).reshape(-1,4,3)
    
            all_faces_keys = {}
            for l in range(all_faces.size(0)):
                key = " ".join(["{}".format(int(p)) for p in all_faces[l].sum(0) // 4])
                if key not in all_faces_keys:
                    all_faces_keys[key] = all_faces[l]

            vertex = np.array([tuple(all_vertex_keys[key].cpu().tolist()) for key in all_vertex_idxs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            face = np.array([([all_vertex_dicts["{} {} {}".format(*b)] for b in a.cpu().tolist()],) for a in all_faces_keys.values()],
                dtype=[('vertex_indices', 'i4', (4,))])
            return PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])

    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        encoder_states = self.precompute()
        points = encoder_states['voxel_center_xyz']

        scores = self.get_scores(field_fn, th=th, bits=bits, encoder_states=encoder_states)
        voxel_size = self.voxel_size
        minimal_voxel_point = points.min(dim=0, keepdim=True)[0]
        coords = ((points - minimal_voxel_point) / voxel_size).round_().long()  # float
        residual = (points - coords.type_as(points) * voxel_size).mean(0, keepdim=True)

        A, B, C = [s + 1 for s in coords.max(0).values.cpu().tolist()]
        # prepare grids
        full_grids = points.new_ones(A * B * C, bits ** 3)
        full_grids[coords[:, 0] * B * C + coords[:, 1] * C + coords[:, 2]] = scores
        full_grids = full_grids.reshape(A, B, C, bits, bits, bits)
        full_grids = full_grids.permute(0, 3, 1, 4, 2, 5).reshape(A * bits, B * bits, C * bits)
        full_grids = 1 - full_grids

        # marching cube
        from skimage import measure
        space_step = self.voxel_size.item() / bits
        verts, faces, normals, _ = measure.marching_cubes_lewiner(
            volume=full_grids.cpu().numpy(), level=0.5,
            spacing=(space_step, space_step, space_step)
        )
        verts += (residual - (self.voxel_size / 2)).cpu().numpy()
        verts = np.array([tuple(a) for a in verts.tolist()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        faces = np.array([(a, ) for a in faces.tolist()], dtype=[('vertex_indices', 'i4', (3,))])
        return PlyData([PlyElement.describe(verts, 'vertex'), PlyElement.describe(faces, 'face')])
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _raymarching as _backend
except ImportError:
    from .backend import _backend


# ----------------------------------------
# utils
# ----------------------------------------

class _near_far_from_aabb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)

        _backend.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)

        return nears, fars

near_far_from_aabb = _near_far_from_aabb.apply


class _sph_from_ray(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, radius):
        ''' sph_from_ray, CUDA implementation
        get spherical coordinate on the background sphere from rays.
        Assume rays_o are inside the Sphere(radius).
        Args:
            rays_o: [N, 3]
            rays_d: [N, 3]
            radius: scalar, float
        Return:
            coords: [N, 2], in [-1, 1], theta and phi on a sphere. (further-surface)
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        coords = torch.empty(N, 2, dtype=rays_o.dtype, device=rays_o.device)

        _backend.sph_from_ray(rays_o, rays_d, radius, N, coords)

        return coords

sph_from_ray = _sph_from_ray.apply


class _morton3D(Function):
    @staticmethod
    def forward(ctx, coords):
        ''' morton3D, CUDA implementation
        Args:
            coords: [N, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...) 
            TODO: check if the coord range is valid! (current 128 is safe)
        Returns:
            indices: [N], int32, in [0, 128^3)
            
        '''
        if not coords.is_cuda: coords = coords.cuda()
        
        N = coords.shape[0]

        indices = torch.empty(N, dtype=torch.int32, device=coords.device)
        
        _backend.morton3D(coords.int(), N, indices)

        return indices

morton3D = _morton3D.apply

class _morton3D_invert(Function):
    @staticmethod
    def forward(ctx, indices):
        ''' morton3D_invert, CUDA implementation
        Args:
            indices: [N], int32, in [0, 128^3)
        Returns:
            coords: [N, 3], int32, in [0, 128)
            
        '''
        if not indices.is_cuda: indices = indices.cuda()
        
        N = indices.shape[0]

        coords = torch.empty(N, 3, dtype=torch.int32, device=indices.device)
        
        _backend.morton3D_invert(indices.int(), N, coords)

        return coords

morton3D_invert = _morton3D_invert.apply


class _packbits(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, thresh, bitfield=None):
        ''' packbits, CUDA implementation
        Pack up the density grid into a bit field to accelerate ray marching.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [C, H * H * H / 8]
        '''
        if not grid.is_cuda: grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        N = C * H3 // 8

        if bitfield is None:
            bitfield = torch.empty(N, dtype=torch.uint8, device=grid.device)

        _backend.packbits(grid, N, thresh, bitfield)

        return bitfield

packbits = _packbits.apply

# ----------------------------------------
# train functions
# ----------------------------------------

class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only)
        Args:
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            step_counter: int32, (2), used to count the actual number of generated points.
            mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
            perturb: bool
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs.
            deltas: float, [M, 2], all generated points' deltas. (first for RGB, second for Depth)
            rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 2]] --> points belonging to rays[i, 0]
        '''

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        if not density_bitfield.is_cuda: density_bitfield = density_bitfield.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()

        N = rays_o.shape[0] # num rays
        M = N * max_steps # init max points number in total

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random ignored rays if underestimated.
        if not force_all_rays and mean_count > 0:
            if align > 0:
                mean_count += align - mean_count % align
            M = mean_count
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps

        if step_counter is None:
            step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)
        
        _backend.march_rays_train(rays_o, rays_d, density_bitfield, bound, dt_gamma, max_steps, N, C, H, M, nears, fars, xyzs, dirs, deltas, rays, step_counter, noises) # m is the actually used points number

        #print(step_counter, M)

        # only used at the first (few) epochs.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item() # D2H copy
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]

            torch.cuda.empty_cache()

        return xyzs, dirs, deltas, rays

march_rays_train = _march_rays_train.apply


class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, normals, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        normals = rgbs.contiguous()
        

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)
        normal = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)
        
        _backend.composite_rays_train_forward(sigmas, rgbs,normals, deltas, rays, M, N, T_thresh, weights_sum, depth, image, normal)

        ctx.save_for_backward(sigmas, rgbs, normals, deltas, rays, weights_sum, depth, image, normal)
        ctx.dims = [M, N, T_thresh]

        return weights_sum, depth, image, normal
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_depth, grad_image, grad_normal):

        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, normals, deltas, rays, weights_sum, depth, image, normal = ctx.saved_tensors
        M, N, T_thresh = ctx.dims
   
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        grad_normals = torch.zeros_like(normals)

        _backend.composite_rays_train_backward(grad_weights_sum, grad_image, grad_normal,\
            sigmas, rgbs, normals, deltas, rays, weights_sum, image, normal, M, N, T_thresh, grad_sigmas, grad_rgbs, grad_normals)

        return grad_sigmas, grad_rgbs, grad_normals, None, None, None


composite_rays_train = _composite_rays_train.apply

# ----------------------------------------
# infer functions
# ----------------------------------------

class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_bitfield, C, H, near, far, align=-1, perturb=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            perturb: bool/int, int > 0 is used as the random seed.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        '''
        
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device) # 2 vals, one for rgb, one for depth

        if perturb:
            # torch.manual_seed(perturb) # test_gui uses spp index as seed
            noises = torch.rand(n_alive, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(n_alive, dtype=rays_o.dtype, device=rays_o.device)

        _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, dt_gamma, max_steps, C, H, density_bitfield, near, far, xyzs, dirs, deltas, noises)

        return xyzs, dirs, deltas

march_rays = _march_rays.apply


class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, normals, deltas, weights_sum, depth, image, normal, T_thresh=1e-2):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        _backend.composite_rays(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs,normals, deltas, weights_sum, depth, image, normal)
        return tuple()

composite_rays = _composite_rays.apply

class _update_tsdf(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, depths, grid_size, trunc, narrow, tsdf_grid, weight_grid, weight_type=0):
        ''' Integrate TSDF & weight Grid using depth (forward only)
        Args:
            rays_o/d: float, [B,N_pixels, 3], 
            depths: float, [B,N_pixels,1], normalized distance as bound
            bound: float, scalar
            tsdf/weight grid: float, [grid_size^3]
            grid_size: int
            trunc: float, truncated distance
            narrow: nearest distance for computing weight
        '''

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        if not tsdf_grid.is_cuda: tsdf_grid = tsdf_grid.cuda()
        if not weight_grid.is_cuda: weight_grid = weight_grid.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        depths = depths.contiguous().view(-1,1)
        tsdf_grid = tsdf_grid.contiguous()
        weight_grid = weight_grid.contiguous()

        N = rays_o.shape[0] # num rays

        _backend.integrate_tsdf(rays_o, rays_d, depths, grid_size, trunc, narrow, N, weight_type, tsdf_grid, weight_grid)
        
        return
    
update_tsdf = _update_tsdf.apply

class _march_rays_tsdf(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, tsdf_grid, dist_surf, min_dt, 
                max_dt, nb_margin, max_walk_step, grid_size,
                near=None, far= None,
                max_steps = 512, perturb = False):
        
        '''
        ray marching using tsdf
            
        Progress        
            1. apply AABB Intersection
            2. Find First Surface (sdf value < dist_surf)
            3. march rays regarding current voxel's sdf. sdf value is bigger, step size is longer
            4. check stop conditions
               1) number of step is maximum.
               2)ray is inside on surface(neighbor's sdfs is all negative) 
        
        Args:
            rays_o/d: float, [N,3], camera origin and direction
            bound: float, scalar, bounded sample space (-bound,bound)^3
            dist_surf: float, scalar, if sdf is larger than dist_surf, skip marching
            min/max_dt: float, scalar, mapping: [0,dist] -> [min_dt,max_dt] inversely
            nb_margin: float, scalar, search space (-nb_margin,nb_margin)^3 to determine the current voxel is inside on surface
            max_walk_step: int, scalar, even if all neighbor's sdf is negative, keep marching until such cases occur max_walk_step times. 
            grid_size: int, scalar, num of voxels = grid_size^3
            max_steps: int, scalar, the maximum number of steps per ray 
        
        Returns:
            rays: int, [N,3], [ray index, start index, num of steps]
            xyzs: float, [M,3], samples points [ray i, ray j, ...]
            dirs: float, [M,3], samples points's directions
            deltas: float, [M,2], samples points's directions
        '''
        
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        if not tsdf_grid.is_cuda: tsdf_grid = tsdf_grid.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        tsdf_grid = tsdf_grid.contiguous()

        N = rays_o.shape[0] # num rays
        M = N * max_steps # init max points number in total
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps
    
        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)
    
        step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        if near is None and far is None:
            near, far = near_far_from_aabb(rays_o, rays_d, torch.tensor([-bound, -bound, -bound, bound, bound, bound], dtype=torch.float32, device=rays_o.device))
        
        _backend.march_rays_tsdf(rays_o,rays_d, tsdf_grid, dist_surf, min_dt, max_dt, nb_margin, max_walk_step, noises, grid_size, max_steps, N, near,far,
        xyzs, dirs, deltas, rays, step_counter)
    
        m = step_counter[0].item() # D2H copy
        
        xyzs = xyzs[:m]
        dirs = dirs[:m]
        deltas = deltas[:m]
    
        return rays, xyzs, dirs, deltas

march_rays_tsdf = _march_rays_tsdf.apply 

class _march_rays_tsdf_uniform(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, tsdf_grid, dist_surf, dt, nb_margin, max_walk_step, grid_size,
                near=None, far= None,
                max_steps = 512, perturb = False):
        
        '''
        ray marching using tsdf uniformly
            
        Progress        
            1. apply AABB Intersection => [t1,t2]
            2. Find First Surface (sdf value < dist_surf) => t1
            3. check stop conditions and determine t2
               1) number of step is maximum.
               2)ray is inside on surface(neighbor's sdfs is all negative) 
            4. sample uniformly in reduced intervals
        Args:
            rays_o: float, [N,3], camera origin
            rays_d: float, [N,3], camera direction
            bound: float, scalar, bounded sample space (-bound,bound)^3
            dist_surf: float, scalar, if sdf is larger than dist_surf, skip marching
            dt: float, scalar, step_size
            nb_margin: float, scalar, search space (-nb_margin,nb_margin)^3 to determine the current voxel is inside on surface
            max_walk_step: int, scalar, even if all neighbor's sdf is negative, keep marching until such cases occur max_walk_step times. 
            grid_size: int, scalar, num of voxels = grid_size^3
            max_steps: int, scalar, the maximum number of steps per ray 
        
        Returns:
            rays: int, [N,3], [ray index, start index, num of steps]
            xyzs: float, [M,3], samples points [ray i, ray j, ...]
            dirs: float, [M,3], samples points's directions
            deltas: float, [M,3], samples points's directions
            intervals: float, [N,2], reduced search intervals per rays
        '''
        
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        if not tsdf_grid.is_cuda: tsdf_grid = tsdf_grid.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        tsdf_grid = tsdf_grid.contiguous()

        N = rays_o.shape[0] # num rays
        M = N * max_steps # init max points number in total
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
        zs = torch.zeros(M, 1, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps
        intervals = torch.zeros(N,2, dtype=rays_o.dtype,device=rays_o.device)
    
        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)
    
        step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        if near is None and far is None:
            near, far = near_far_from_aabb(rays_o, rays_d, torch.tensor([-bound, -bound, -bound, bound, bound, bound], dtype=torch.float32, device=rays_o.device))
        
        _backend.march_rays_tsdf_uniform(rays_o,rays_d, tsdf_grid, dist_surf, dt, nb_margin, max_walk_step, noises, grid_size, max_steps, N, near, far, xyzs, dirs, deltas, zs, rays, step_counter, intervals)
    
        m = step_counter[0].item() # D2H copy
        
        xyzs = xyzs[:m]
        dirs = dirs[:m]
        deltas = deltas[:m]
        zs = zs[:m]
    
        return rays, xyzs, dirs, deltas,zs, intervals

march_rays_tsdf_uniform = _march_rays_tsdf_uniform.apply

class _evaluate(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, rays, alphas, deltas, nears, n_fine_samples):
        '''
        evaluate and resample
        Progress        
            1. compute weights from alphas
            2. resample n_fine_sample using the weights
        Args:
           rays: int, [N,3], (index, offset, count)
           alphas: float, [M,] alphas = 1-e^(sigma*dt)
           dletas: float, [M,2] (dt, real dt)
           nears: float, [M,] (near)
           N: float, scalar N = batch_size * N_samples
           n_fine_samples, int, scalar, num of fine samples 
        Returns:
            z_vals: float, [M,3], samples points's directions
        '''
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        alphas = alphas.contiguous()
        nears = nears.contiguous()
        
        N = rays.shape[0] # batch size
        zs_fine = torch.zeros(N, n_fine_samples, dtype=torch.float32, device=rays.device).view(-1,1).contiguous()
        weights = torch.zeros_like(alphas)
        weights_sum = torch.zeros(N,1,dtype=torch.float32, device=rays.device)
        
        M = alphas.shape[0] + n_fine_samples * N 
        xyzs_all = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs_all = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas_all = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
        zs_all = torch.zeros(M, 1, dtype=rays_o.dtype, device=rays_o.device)
        rays_all = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps
        step_counter_all = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        _backend.evaluate(rays_o,rays_d, alphas, deltas, nears, rays, n_fine_samples, N, weights, weights_sum, zs_fine,xyzs_all,dirs_all, deltas_all, zs_all,rays_all, step_counter_all)
        
        zs_fine = zs_fine.reshape((N,n_fine_samples))
        
        return rays_all, xyzs_all, dirs_all, deltas_all,zs_all, zs_fine, weights_sum 

evaluate = _evaluate.apply 


class _volume_render_adaptive(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, zs, deltas, rays, T_thresh=1e-5):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            deltas: float, [M, 2]
            zs: float, [M, 1]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights = torch.zeros_like(sigmas)
        alphas = torch.zeros_like(sigmas)
        weights_sum = torch.empty(N,1,dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N,1,dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)
        
        _backend.volume_render(sigmas, rgbs,deltas, zs, rays, M, N, T_thresh,alphas, weights, weights_sum, depth, image)

        return weights, weights_sum, depth, image 


volume_render_adaptive = _volume_render_adaptive.apply
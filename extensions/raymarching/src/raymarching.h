#pragma once

#include <stdint.h>
#include <torch/torch.h>


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars);
void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords);
void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises);
void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs,const at::Tensor normals,
 const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image, at::Tensor normal);
void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor grad_normal, const at::Tensor sigmas, const at::Tensor rgbs,const at::Tensor normals, const at::Tensor deltas, 
const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image,const at::Tensor normal, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor grad_normals);

void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor normals, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor image, at::Tensor normal);

// TSDF grid
void integrate_tsdf(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor depths, const uint32_t grid_size, const float trunc, const float narrow, const uint32_t N, const uint32_t weight_type,
    at::Tensor tsdf_grid, at::Tensor weight_grid);

void march_rays_tsdf(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor tsdf_grid, const float dist_surf, const float min_dt, const float max_dt, const uint32_t nb_margin, const int max_walk_step, const at::Tensor noise, const uint32_t grid_size, const uint32_t max_steps, const uint32_t N, const at::Tensor nears, const at::Tensor fars, 
    //outputs
    at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter);

void march_rays_tsdf_uniform(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor tsdf_grid, const float dist_surf, const float dt, const uint32_t nb_margin,
    const uint32_t max_walk_step, const at::Tensor noise, const uint32_t grid_size, const uint32_t max_steps, const uint32_t N, at::Tensor nears, at::Tensor fars, 
    //outputs
    at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas,at::Tensor zs,
    at::Tensor rays, at::Tensor counter, at::Tensor intervals);

void evaluate(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor alphas, const at::Tensor deltas, const at::Tensor nears, const at::Tensor rays, const uint32_t n_fine_samples, const uint32_t N, 
    // outputs
    at::Tensor weights,at::Tensor weights_sum, at::Tensor zs_fine,
    at::Tensor xyzs_all, at::Tensor dirs_all, at::Tensor deltas_all,at::Tensor zs_all, at::Tensor rays_all,at::Tensor step_counter_all);

void volume_render(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor zs, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor alphas,at::Tensor weights, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);

// void compute_important_samples(const at::Tensor weights, const at::Tensor rays, const uint32_t M, const uint32_t N, at::Tensor important_samples);
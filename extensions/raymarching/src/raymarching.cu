#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

// #include <./include/helper_math.h>
#include "helper_math.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

// inline __host__ __device__ float signf(const float x) {
//     return copysignf(1.0, x);
// }

// inline __host__ __device__ float clamp(const float x, const float min, const float max) {
//     return fminf(max, fmaxf(min, x));
// }

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = __expand_bits(x);
	uint32_t yy = __expand_bits(y);
	uint32_t zz = __expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

// atomic update
__device__ void update_distance_weight_safe(float* distance_addr, float* weight_addr, float dist, float weight) {
    float old_dist;
    float old_weight;
    float new_dist;
    
    unsigned int assumed_dist;

    do {
        old_dist = *distance_addr;
        old_weight = *weight_addr;
        new_dist = (old_weight * old_dist + weight * dist) / (old_weight + weight);
        assumed_dist = atomicCAS((unsigned int*)distance_addr,__float_as_uint(old_dist),__float_as_uint(new_dist));
    } while (__float_as_uint(old_dist) != assumed_dist);
    atomicAdd(weight_addr, weight);
}

// weight function
#define CONSTANT_WEIGHT 0
#define LINEAR_WEIGHT 1
#define NARROW_LINEAR_WEIGHT 2
#define NARROW_EXP_WEIGHT 3

__device__ inline float constant_weight(float distance) {
	return 1.f;
}

__device__ inline float linear_weight(float distance, float trunc_dist, float near_dist){
	
    if(distance <= -trunc_dist){
        return 0.f;
    }else if((distance >= -trunc_dist) && (distance <= -near_dist)){
        return (trunc_dist + distance) / (trunc_dist - near_dist);
    }else{
        return 1.f;
    }
}

__device__ inline float narrow_linear_weight(float distance, float trunc_dist, float near_dist){
	
    distance = distance >= 0? distance:-distance;
    if(distance >= trunc_dist){
        return 0.f;
    }else if((distance < trunc_dist) && (distance > near_dist)){
        return (trunc_dist - distance) / (trunc_dist - near_dist);
    }else{
        return 1;
    } 
}

__device__ inline float narrow_exp_weight(float distance, float trunc_dist, float near_dist, float noise){
	
    distance = distance >= 0? distance:-distance;

    if(distance >= trunc_dist){
        return 0;
    }else if((distance < trunc_dist) && (distance > near_dist)){
        return expf(-noise*(distance-near_dist)*(distance-near_dist));
    }else{
        return 1;
    }
}

__device__ inline float compute_weight(float distance, float trunc_dist, float near_dist, uint32_t weight_type, float noise =0.1f){

    switch (weight_type)
    {
        case CONSTANT_WEIGHT:
            return constant_weight(distance);
        case LINEAR_WEIGHT:
            return linear_weight(distance,trunc_dist, near_dist);
        case NARROW_LINEAR_WEIGHT:
            return narrow_linear_weight(distance,trunc_dist, near_dist);
        case NARROW_EXP_WEIGHT:
            return narrow_exp_weight(distance, trunc_dist, near_dist, noise);
        default:
            return linear_weight(distance, trunc_dist,near_dist);
   }
}

////////////////////////////////////////////////////
/////////////           utils          /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// nears/fars: [N]
// scalar_t should always be float in use.
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), aabb.data_ptr<scalar_t>(), N, min_near, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>());
    }));
}


// rays_o/d: [N, 3]
// radius: float
// coords: [N, 2]
template <typename scalar_t>
__global__ void kernel_sph_from_ray(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const float radius,
    const uint32_t N,
    scalar_t * coords
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    coords += n * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // solve t from || o + td || = radius
    const float A = dx * dx + dy * dy + dz * dz;
    const float B = ox * dx + oy * dy + oz * dz; // in fact B / 2
    const float C = ox * ox + oy * oy + oz * oz - radius * radius;

    const float t = (- B + sqrtf(B * B - A * C)) / A; // always use the larger solution (positive)

    // solve theta, phi (assume y is the up axis)
    const float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
    const float theta = atan2(sqrtf(x * x + z * z), y); // [0, PI)
    const float phi = atan2(z, x); // [-PI, PI)

    // normalize to [-1, 1]
    coords[0] = 2 * theta * RPI() - 1;
    coords[1] = phi * RPI();
}


void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "sph_from_ray", ([&] {
        kernel_sph_from_ray<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), radius, N, coords.data_ptr<scalar_t>());
    }));
}


// coords: int32, [N, 3]
// indices: int32, [N]
__global__ void kernel_morton3D(
    const int * __restrict__ coords,
    const uint32_t N,
    int * indices
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;
    indices[n] = __morton3D(coords[0], coords[1], coords[2]);
}


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D<<<div_round_up(N, N_THREAD), N_THREAD>>>(coords.data_ptr<int>(), N, indices.data_ptr<int>());
}


// indices: int32, [N]
// coords: int32, [N, 3]
__global__ void kernel_morton3D_invert(
    const int * __restrict__ indices,
    const uint32_t N,
    int * coords
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;

    const int ind = indices[n];

    coords[0] = __morton3D_invert(ind >> 0);
    coords[1] = __morton3D_invert(ind >> 1);
    coords[2] = __morton3D_invert(ind >> 2);
}


void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D_invert<<<div_round_up(N, N_THREAD), N_THREAD>>>(indices.data_ptr<int>(), N, coords.data_ptr<int>());
}


// grid: float, [C, H, H, H]
// N: int, C * H * H * H / 8
// density_thresh: float
// bitfield: uint8, [N]
template <typename scalar_t>
__global__ void kernel_packbits(
    const scalar_t * __restrict__ grid,
    const uint32_t N,
    const float density_thresh,
    uint8_t * bitfield
) {
    // parallel per byte
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    grid += n * 8;

    uint8_t bits = 0;

    #pragma unroll
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (grid[i] > density_thresh) ? ((uint8_t)1 << i) : 0;
    }

    bitfield[n] = bits;
}


void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grid.scalar_type(), "packbits", ([&] {
        kernel_packbits<<<div_round_up(N, N_THREAD), N_THREAD>>>(grid.data_ptr<scalar_t>(), N, density_thresh, bitfield.data_ptr<uint8_t>());
    }));
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const uint8_t * __restrict__ grid,
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M,
    const scalar_t* __restrict__ nears, 
    const scalar_t* __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays,
    int * counter,
    const scalar_t* __restrict__ noises
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;
    
    float t0 = near;
    
    // perturb
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);
    
    while (t < far && num_steps < max_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        //if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, num_steps);

        if (occ) {
            num_steps++;
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;

            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }

    //printf("[n=%d] num_steps=%d, near=%f, far=%f, dt=%f, max_steps=%f\n", n, num_steps, near, far, dt_min, (far - near) / dt_min);

    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);
    
    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps > M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;

    t = t0;
    uint32_t step = 0;

    float last_t = t;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max); 
            } while (t < tt);
        }
    }
}

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises) {

    static constexpr uint32_t N_THREAD = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), bound, dt_gamma, max_steps, N, C, H, M, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>(), noises.data_ptr<scalar_t>());
    }));
}


// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ normals,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh, 
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * image,
    scalar_t * normal
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index * 3] = 0;
        image[index * 3 + 1] = 0;
        image[index * 3 + 2] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * 3;
    normals += offset * 3;
    deltas += offset * 2;

    // accumulate 
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, t = 0, d = 0;
    scalar_t nx = 0, ny = 0, nz = 0;
    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        nx += weight * normals[0];
        ny += weight * normals[1];
        nz += weight * normals[2];
        
        t += deltas[1]; // real delta
        d += weight * t;
        
        ws += weight;
        
        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;

    normal[index * 3] = nx;
    normal[index * 3 + 1] = ny;
    normal[index * 3 + 2] = nz;
}


void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor normals,  const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image, at::Tensor normal) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), normals.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), M, N, T_thresh, weights_sum.data_ptr<scalar_t>(),
         depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), normal.data_ptr<scalar_t>());
    }));
}


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here 
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ grad_normal,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ normals, 
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,
    const scalar_t * __restrict__ normal,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs,
    scalar_t * grad_normals
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > M) return;

    grad_weights_sum += index;
    grad_image += index * 3;
    grad_normal += index * 3;
    
    weights_sum += index;
    image += index * 3;
    normal += index * 3;
    sigmas += offset;
    rgbs += offset * 3;
    normals += offset * 3;
    deltas += offset * 2;
    grad_sigmas += offset;
    grad_rgbs += offset * 3;
    grad_normals += offset * 3;


    // accumulate 
    uint32_t step = 0;
    
    scalar_t T = 1.0f;
    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2], ws_final = weights_sum[0];
    const scalar_t nx_final = normal[0], ny_final = normal[1], nz_final = normal[2];

    scalar_t r = 0, g = 0, b = 0, ws = 0;
    scalar_t nx = 0, ny = 0, nz = 0;

    while (step < num_steps) {
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        nx += weight * normals[2];
        ny += weight * normals[2];
        nz += weight * normals[2];

        ws += weight;

        T *= 1.0f - alpha;
        
        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        grad_rgbs[0] = grad_image[0] * weight;
        grad_rgbs[1] = grad_image[1] * weight;
        grad_rgbs[2] = grad_image[2] * weight;


        grad_normals[0] = grad_normal[0] * weight;
        grad_normals[1] = grad_normal[1] * weight;
        grad_normals[2] = grad_normal[2] * weight;



        // write grad_sigmas
        grad_sigmas[0] = deltas[0] * (
            grad_image[0] * (T * rgbs[0] - (r_final - r)) + 
            grad_image[1] * (T * rgbs[1] - (g_final - g)) + 
            grad_image[2] * (T * rgbs[2] - (b_final - b)) +
            grad_normal[0] * (T * normals[0] - (nx_final - nx)) +
            grad_normal[1] * (T * normals[1] - (ny_final - ny)) +
            grad_normal[2] * (T * normals[2] - (nz_final - nz)) +
            grad_weights_sum[0] * (1 - ws_final)
        );

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
        // minimal remained transmittence
        if (T < T_thresh) break;
        
        // locate
        sigmas++;
        rgbs += 3;
        normals += 3;

        deltas += 2;
        grad_sigmas++;
        grad_rgbs += 3;
        grad_normals += 3;

        step++;
    }
}


void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image,const at::Tensor grad_normal, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor normals,
  const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image,const at::Tensor normal, const uint32_t M, const uint32_t N, const float T_thresh,
   at::Tensor grad_sigmas, at::Tensor grad_rgbs,at::Tensor grad_normals) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_image.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(),
         grad_image.data_ptr<scalar_t>(), grad_normal.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), normals.data_ptr<scalar_t>(),
          deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), normal.data_ptr<scalar_t>(),  M, N, T_thresh,
           grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>(), grad_normals.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    const scalar_t* __restrict__ rays_t, 
    const scalar_t* __restrict__ rays_o, 
    const scalar_t* __restrict__ rays_d, 
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H,
    const uint8_t * __restrict__ grid,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas,
    const scalar_t* __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];
    
    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;
    
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;
    
    float t = rays_t[index]; // current ray's t
    const float far = fars[index];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;

    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}


void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor near, const at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises) {
    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), bound, dt_gamma, max_steps, C, H, grid.data_ptr<uint8_t>(), near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), noises.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh,
    int* rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs,
    const scalar_t* __restrict__ normals, 
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image, scalar_t* normal
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    
    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    normals += n * n_step * 3;
    deltas += n * n_step * 2;
    
    rays_t += index;
    weights_sum += index;
    depth += index;
    image += index * 3;
    normal += index * 3;

    scalar_t t = rays_t[0]; // current ray's t
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];

    scalar_t nx = normal[0];
    scalar_t ny = normal[1];
    scalar_t nz = normal[2];

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        nx += weight * normals[0];
        ny += weight * normals[1];
        nz += weight * normals[2];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;

    normal[0] = nx;
    normal[1] = ny;
    normal[2] = nz;


}


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive,
 at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor normals, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor normal) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh,
        rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(),normals.data_ptr<scalar_t>(),
         deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), normal.data_ptr<scalar_t>());
    }));
}

template <typename scalar_t>
__global__ void kernel_integrate_tsdf(
    const scalar_t * __restrict__ rays_o, // [N, 3]
    const scalar_t * __restrict__ rays_d, // [N, 3]
    const scalar_t * __restrict__ depths, // [N,]  
    const uint32_t grid_size,
    const float trunc,
    const float narrow,
    const uint32_t N,
    const uint32_t weight_type,
    float* tsdf_grid,
    float* weight_grid
){
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;
    
    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    depths += n;

    const float3 pos = make_float3(rays_o[0],rays_o[1],rays_o[2]);
    const float3 dir = make_float3(rays_d[0],rays_d[1],rays_d[2]);
    const float3 inv_dir = 1.f / dir;
    
    // surface
    float t = 0;
    if(depths[0] < 0) return; // invalid depth
    const float3 surf = pos + depths[0] * dir;
    const float3 next_voxel_dir = 0.5f + 0.5f * signf(dir);

    while(true){
        // update voxel
        const float3 p = pos + t * dir;
        if(p.x >= 1.f || p.x <= -1.f ||
           p.y >= 1.f || p.y <= -1.f ||
           p.z >= 1.f || p.z <= -1.f) break; 
        
        // grid coordinate: [0, grid_size-1]^3
        const int3 coord = make_int3(clamp(0.5f * (p +1) * (grid_size -1.f), 0.0f, grid_size -1.0f)); 

        // Find voxel's center
        const float3 voxel_center = ((make_float3(coord) + 0.5f)/ 
                        (grid_size - 1.0f) * 2 - 1);
        const float diff = dot(dir, surf - voxel_center);
        
        if(diff < -trunc) break; // stop update
        
        float dist = clamp(diff, -trunc, trunc);
        float weight = compute_weight(dist, trunc, narrow, weight_type);

        const uint32_t idx = __morton3D(coord.x, coord.y, coord.z);
        update_distance_weight_safe(&tsdf_grid[idx],&weight_grid[idx], dist, weight);
        // march next voxel
        const float3 next =
          (((make_float3(coord) + next_voxel_dir) / (grid_size - 1.0f) * 2 - 1) -
               p) * inv_dir;
        t = t + fmaxf(0.f, fminf(next)) + 1e-6;
    }
}

// TSDF grid
void integrate_tsdf(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor depths, const uint32_t grid_size, const float trunc,
    const float narrow, const uint32_t N, const uint32_t weight_type,
    at::Tensor tsdf_grid, at::Tensor weight_grid){

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "integrate_tsdf", ([&] {
        kernel_integrate_tsdf<<<div_round_up(N, N_THREAD), N_THREAD>>>(
            rays_o.data_ptr<scalar_t>(),
            rays_d.data_ptr<scalar_t>(),
            depths.data_ptr<scalar_t>(),
            grid_size,
            trunc,
            narrow,
            N,
            weight_type,
            tsdf_grid.data_ptr<float>(),
            weight_grid.data_ptr<float>());
    }));
}

template <typename scalar_t>
__global__ void kernel_march_ray_tsdf(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,    
    const scalar_t * __restrict__ tsdf_grid,        
    const float dist_surf,
    const float min_dt,
    const float max_dt,
    const uint32_t nb_margin,
    const uint32_t max_walk_step,
    const scalar_t * __restrict__ noise,
    const uint32_t grid_size,
    const uint32_t max_steps,
    const uint32_t N,
    const scalar_t * __restrict__ nears,
    const scalar_t * __restrict__ fars,
    // outputs
    scalar_t * xyzs,
    scalar_t * dirs,
    scalar_t * deltas,
    int * rays,
    int * counter
){

    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;
    
    const float3 pos = make_float3(rays_o[n*3],rays_o[n*3+1],rays_o[n*3+2]);
    const float3 dir = make_float3(rays_d[n*3],rays_d[n*3+1],rays_d[n*3+2]);
    
    uint32_t n_steps = 0; uint32_t n_walk_step = 0;
    const float t1 = nears[n];
    const float t2 = fars[n];    
    
    float t = t1;
    t += (min_dt + max_dt) * noise[n] * 0.5f;
    float sdf = 0.f; float abs_sdf = 0.f;

    while((n_steps < max_steps) && (t < t2)){
        const float3 p = pos + t * dir;
        // grid coordinate: [0, grid_size-1]^3
        const int3 coord = 
            make_int3(clamp(0.5f * (p +1) * (grid_size -1.f), 0.0f, grid_size -1.0f)); 

        const uint32_t idx = __morton3D(coord.x, coord.y, coord.z);
        sdf = tsdf_grid[idx]; 

        if(sdf >= dist_surf){
            t += dist_surf / 2.f;
        }else{
            n_steps++;
            abs_sdf = sdf > 0 ? sdf : -sdf;
            const float step_size = (max_dt - min_dt)* abs_sdf / dist_surf + min_dt;
            t += step_size;
        }
        // check whether the point is inside on the surface.
        // To determine it certainly, search neighbor's sdf
        // If all values of neighbors is negative, the point would be inside and stop marching algorithm  
        if (sdf < 0){
            bool stop = true;
            const uint32_t size = 2*nb_margin + 1;
            const uint32_t n_nb = size*size*size;            
            for(int nb_idx = 0; nb_idx < n_nb; nb_idx++)
            {   
                /*
                0 => (-nb_marin,-nb_marin,-nb_marin)
                1 => (-nb_marin+1,-nb_marin,-nb_marin)
                2 => (-nb_marin+2,-nb_marin,-nb_marin)
                .
                .
                (2*nb_margin+1)^3-1 => (nb_margin,nb_margin,nb_margin) 
                */
                uint32_t i = nb_idx % size - nb_margin;
                uint32_t j = (nb_idx % (size*size)) / size - nb_margin;
                uint32_t k = nb_idx / (size*size) - nb_margin;

                int3 nb = clamp(make_int3(coord.x+i,coord.y+j,coord.z+k), 0, grid_size-1);
                if(tsdf_grid[__morton3D(nb.x,nb.y,nb.z)] >=0.f){
                    stop = false;
                    n_walk_step = 0; // count again from 0
                    break;
                }
            }
            if(stop) {
                n_walk_step++;
                if(max_walk_step == n_walk_step) break;
            }
        }
    }

    const uint32_t point_index = atomicAdd(counter, n_steps);
    const uint32_t ray_index = atomicAdd(counter + 1, 1);

    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = n_steps;
    
    t = t1;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;

    uint32_t step = 0;

    float last_t = t;

    while((t < t2) && (step < n_steps)){
        const float3 p = pos + t * dir;
        // grid coordinate: [0, grid_size-1]^3
        const int3 coord = make_int3(clamp(0.5f * (p +1) * (grid_size -1.f), 0.0f, grid_size -1.0f)); 

        const uint32_t idx = __morton3D(coord.x, coord.y, coord.z);
        sdf = tsdf_grid[idx]; 

        if(sdf >= dist_surf){
            t += dist_surf / 2.f;
        }else{
            step++;
            abs_sdf = sdf > 0 ? sdf : -sdf;
            const float step_size = (max_dt - min_dt)* abs_sdf / dist_surf + min_dt;

            xyzs[0] = p.x;  xyzs[1] = p.y;  xyzs[2] = p.z;
            dirs[0] = dir.x; dirs[1] = dir.y; dirs[2] = dir.z;
            t += step_size;
            deltas[0] = step_size;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            xyzs += 3;
            dirs += 3;
            deltas += 2;
        }
    }

}


void march_rays_tsdf(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor tsdf_grid, const float dist_surf, const float min_dt, const float max_dt, const uint32_t nb_margin, const int max_walk_step, const at::Tensor noise, const uint32_t grid_size, const uint32_t max_steps, const uint32_t N, const at::Tensor nears, const at::Tensor fars, 
    //outputs
    at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter){

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_ray_tsdf", ([&] {
        kernel_march_ray_tsdf<<<div_round_up(N, N_THREAD), N_THREAD>>>(
            rays_o.data_ptr<scalar_t>(),
            rays_d.data_ptr<scalar_t>(),
            tsdf_grid.data_ptr<scalar_t>(),
            dist_surf,
            min_dt,
            max_dt,
            nb_margin,
            max_walk_step,
            noise.data_ptr<scalar_t>(),
            grid_size,
            max_steps,
            N,
            nears.data_ptr<scalar_t>(),
            fars.data_ptr<scalar_t>(),
            xyzs.data_ptr<scalar_t>(),
            dirs.data_ptr<scalar_t>(),
            deltas.data_ptr<scalar_t>(),
            rays.data_ptr<int>(),
            counter.data_ptr<int>()
            );
    }));
}


template <typename scalar_t>
__global__ void kernel_march_rays_tsdf_uniform(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,    
    const scalar_t * __restrict__ tsdf_grid,        
    const float dist_surf,
    const float dt,
    const uint32_t nb_margin,
    const uint32_t max_walk_step,
    const scalar_t * __restrict__ noise,
    const uint32_t grid_size,
    const uint32_t max_steps,
    const uint32_t N,
    const scalar_t * __restrict__ nears,
    const scalar_t * __restrict__ fars,
    // outputs
    scalar_t * xyzs,
    scalar_t * dirs,
    scalar_t * deltas,
    scalar_t * zs,
    int * rays,
    int * counter,
    scalar_t * intervals
){

    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    intervals += n*2;
    const float3 pos = make_float3(rays_o[n*3],rays_o[n*3+1],rays_o[n*3+2]);
    const float3 dir = make_float3(rays_d[n*3],rays_d[n*3+1],rays_d[n*3+2]);
    const float3 inv_dir = 1.f / dir;
    const float3 next_voxel_dir = 0.5f + 0.5f * signf(dir);
    

    uint32_t n_walk_step = 0;
    const float t1 = nears[n], t2 = fars[n];    
    float t = t1;
    intervals[0] = t1;
    float sdf = 0.f;
    
    // 1st pass: reduce near
    // skip empty space(sdf is larger)
    while(t < t2){
        const float3 p = pos + t * dir;
        if(p.x >= 1.f || p.x <= -1.f ||
           p.y >= 1.f || p.y <= -1.f ||
           p.z >= 1.f || p.z <= -1.f) break; 
        // grid coordinate: [0, grid_size-1]^3
        const int3 coord = make_int3(clamp(0.5f * (p +1) * (grid_size -1.f), 0.0f, grid_size -1.0f)); 

        const uint32_t idx = __morton3D(coord.x, coord.y, coord.z);
        sdf = tsdf_grid[idx]; 

        if(sdf <= dist_surf){
            break;
        }
        intervals[0] = t;
        // marching until to find non-empty space
        const float3 next =
          (((make_float3(coord) + next_voxel_dir) / (grid_size - 1.0f) * 2 - 1) -
               p) * inv_dir;
        t = t + fmaxf(0.f, fminf(next)) + 1e-6;
    }

    // 2nd pass: reduce far
    // find inside on surface
    while(t < t2){
        const float3 p = pos + t * dir;
        // grid coordinate: [0, grid_size-1]^3
        const int3 coord = make_int3(clamp(0.5f * (p +1) * (grid_size -1.f), 0.0f, grid_size -1.0f)); 

        const uint32_t idx = __morton3D(coord.x, coord.y, coord.z);
        sdf = tsdf_grid[idx];

        if (sdf < 0){
            bool stop = true;
            const uint32_t size = 2*nb_margin + 1;
            const uint32_t n_nb = size*size*size;            
            for(int nb_idx = 0; nb_idx < n_nb; nb_idx++)
            {   
                /*
                0 => (-nb_marin,-nb_marin,-nb_marin)
                1 => (-nb_marin+1,-nb_marin,-nb_marin)
                2 => (-nb_marin+2,-nb_marin,-nb_marin)
                .
                .
                (2*nb_margin+1)^3-1 => (nb_margin,nb_margin,nb_margin) 
                */
                uint32_t i = nb_idx % size - nb_margin;
                uint32_t j = (nb_idx % (size*size)) / size - nb_margin;
                uint32_t k = nb_idx / (size*size) - nb_margin;

                int3 nb = clamp(make_int3(coord.x+i,coord.y+j,coord.z+k), 0, grid_size-1);
                if(tsdf_grid[__morton3D(nb.x,nb.y,nb.z)] >=0.f){
                    stop = false;
                    n_walk_step = 0; // count again from 0
                    break;
                }
            }
            if(stop) {
                n_walk_step++;
                if(max_walk_step == n_walk_step) break;
            }
        }
        const float3 next =
          (((make_float3(coord) + next_voxel_dir) / (grid_size - 1.0f) * 2 - 1) -
               p) * inv_dir;
        t = t + fmaxf(0.f, fminf(next)) + 1e-6;
    }
    
    intervals[1] = t;

    // sample uiformly in the reduced interval
    float length = intervals[1] - intervals[0];
    uint32_t n_steps = uint32_t(length / dt);
    float step_size = dt;
    if(n_steps > max_steps) {
        n_steps = max_steps;
        step_size = length / n_steps;  
    }

    const uint32_t point_index = atomicAdd(counter, n_steps);
    const uint32_t ray_index = atomicAdd(counter + 1, 1);

    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = n_steps;
    

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;
    zs += point_index;

    uint32_t step = 0;
    
    t = intervals[0];
    float last_t = 0;
    // store the result
    while(step < n_steps){
        step++;
        const float3 p = pos + t * dir;
        
        xyzs[0] = p.x;  xyzs[1] = p.y;  xyzs[2] = p.z;
        dirs[0] = dir.x; dirs[1] = dir.y; dirs[2] = dir.z;
        zs[0] += t;
        t += step_size;
        deltas[0] = step_size;
        deltas[1] = t - last_t; // used to calc depth
        last_t = t;
        xyzs += 3;
        dirs += 3;
        deltas += 2;
        zs += 1;
    }
    
}

void march_rays_tsdf_uniform(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor tsdf_grid, const float dist_surf, const float dt, const uint32_t nb_margin,
    const uint32_t max_walk_step, const at::Tensor noise, const uint32_t grid_size, const uint32_t max_steps, const uint32_t N, at::Tensor nears, at::Tensor fars, 
    //outputs
    at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas,at::Tensor zs,
    at::Tensor rays, at::Tensor counter, at::Tensor intervals){
    
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_ray_tsdf_uniform", ([&] {
        kernel_march_rays_tsdf_uniform<<<div_round_up(N, N_THREAD), N_THREAD>>>(
            rays_o.data_ptr<scalar_t>(),
            rays_d.data_ptr<scalar_t>(),
            tsdf_grid.data_ptr<scalar_t>(),
            dist_surf,
            dt,
            nb_margin,
            max_walk_step,
            noise.data_ptr<scalar_t>(),
            grid_size,
            max_steps,
            N,
            nears.data_ptr<scalar_t>(),
            fars.data_ptr<scalar_t>(),
            xyzs.data_ptr<scalar_t>(),
            dirs.data_ptr<scalar_t>(),
            deltas.data_ptr<scalar_t>(),
            zs.data_ptr<scalar_t>(),
            rays.data_ptr<int>(),
            counter.data_ptr<int>(),
            intervals.data_ptr<scalar_t>()
        );
    }));


}


template <typename scalar_t>
__global__ void kernel_evaluate(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ alphas,
    const scalar_t * __restrict__ deltas,
    const scalar_t * __restrict__ nears,
    const int * __restrict__ rays,
    const uint32_t n_fine_samples,
    const uint32_t N,
    scalar_t * weights,
    scalar_t * weights_sum,
    scalar_t * zs_fine,
    scalar_t * xyzs_all,
    scalar_t * dirs_all,
    scalar_t * deltas_all,
    scalar_t * zs_all,
    int * rays_all,
    int * counter_all
){
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];
    weights += offset;
    deltas += offset * 2;
    zs_fine += index * n_fine_samples;
    weights_sum += index;
    
    const int point_idx_all = atomicAdd(counter_all, num_steps+n_fine_samples);
    rays_all[index * 3] = index;
    rays_all[index * 3 + 1] = point_idx_all;
    rays_all[index * 3 + 2] = num_steps + n_fine_samples;

    const float3 pos = make_float3(rays_o[index*3],rays_o[index*3+1],rays_o[index*3+2]);
    const float3 dir = make_float3(rays_d[index*3],rays_d[index*3+1],rays_d[index*3+2]);

    float near = nears[index];
    float cum_alpha = 1.f;
    float weight_sum = 1e-5f;
    
    // compute weight: T*alpha 
    for(uint32_t s = 0; s < num_steps; s++){
        const float alpha = max(alphas[s], 0.f);
        weights[s] = alpha * cum_alpha; 
        weight_sum += weights[s];
        cum_alpha *= 1.f - alpha;
    }
    weight_sum = min(weight_sum, 1.f);
    weights_sum[0] = weight_sum;
    float below_cdf = weights[0] / weight_sum;
    float above_cdf = weights[0] / weight_sum;
    uint32_t above_index = 0;
    // scalar_t denom = 0.f;

    float t_last = 0;
    float t = near;
    int s = point_idx_all;

    xyzs_all += point_idx_all * 3;
    dirs_all += point_idx_all * 3;
    deltas_all += point_idx_all * 2;
    zs_all += point_idx_all;
    float denom = 0.;

    for(uint32_t i = 0; i < n_fine_samples; i++){
        const float u = weights[0] / weight_sum + (1. - weights[0]/weight_sum) * (0.5 / float(n_fine_samples) + float(i) / float(n_fine_samples));
        while(above_cdf < u){
            t_last = t;
            t += deltas[above_index*2];
            below_cdf = above_cdf;
            above_cdf += weights[++above_index] / weight_sum;
        }
        denom = above_cdf - below_cdf >= 1e-5? above_cdf - below_cdf: 1.f;
        // if (denom < 1e-5) denome = 1.f;
        zs_fine[i] = t_last + ((u-below_cdf)/denom)*(deltas[(above_index-1)*2]);
    }
    // copy coarse zs
    t = near;
    for(int i = 0; i < num_steps; i++){
        zs_all[i] = t;
        t += deltas[i*2];
    }
    // merge coarse zs and fine zs
    int i_c = num_steps-1;
    int i_f = n_fine_samples-1;
    int k = num_steps + n_fine_samples-1;

    while(i_c >=0 && i_f >= 0){
        if(zs_all[i_c] > zs_fine[i_f]){
            zs_all[k] = zs_all[i_c];
            i_c--;
        }else{
            zs_all[k] = zs_fine[i_f];
            i_f--;
        }
        k--;
    }
    while(i_f >= 0){
        zs_all[k]= zs_fine[i_f];
        i_f--;
        k--;
    }

    for(int i = 0; i < num_steps + n_fine_samples;i++){
        const float3 p = pos + zs_all[i] * dir;
        xyzs_all[0] = p.x;   xyzs_all[1] = p.y;   xyzs_all[2] = p.z;
        dirs_all[0] = dir.x; dirs_all[1] = dir.y; dirs_all[2] = dir.z;
        if(i == 0){
            deltas_all[0] = deltas[0];
            deltas_all[1] = deltas[1];
        }else{
            deltas_all[0] = zs_all[i]-zs_all[i-1];
            deltas_all[1] = zs_all[i]-zs_all[i-1];
        }
        xyzs_all +=3;
        dirs_all +=3;
        deltas_all +=2;
    } 
}

void evaluate(
    const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor alphas, const at::Tensor deltas, const at::Tensor nears, const at::Tensor rays, const uint32_t n_fine_samples, const uint32_t N,
    // outputs
    at::Tensor weights,at::Tensor weights_sum, at::Tensor zs_fine,
    at::Tensor xyzs_all, at::Tensor dirs_all, at::Tensor deltas_all,at::Tensor zs_all, at::Tensor rays_all,at::Tensor step_counter_all){

   static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    alphas.scalar_type(), "evaluate", ([&] {
        kernel_evaluate<<<div_round_up(N, N_THREAD), N_THREAD>>>(
            rays_o.data_ptr<scalar_t>(),
            rays_d.data_ptr<scalar_t>(),
            alphas.data_ptr<scalar_t>(),
            deltas.data_ptr<scalar_t>(),
            nears.data_ptr<scalar_t>(),
            rays.data_ptr<int>(),
            n_fine_samples,
            N,
            weights.data_ptr<scalar_t>(),
            weights_sum.data_ptr<scalar_t>(),
            zs_fine.data_ptr<scalar_t>(),
            xyzs_all.data_ptr<scalar_t>(),
            dirs_all.data_ptr<scalar_t>(),
            deltas_all.data_ptr<scalar_t>(),
            zs_all.data_ptr<scalar_t>(),
            rays_all.data_ptr<int>(),
            step_counter_all.data_ptr<int>()
        );
    }));
}

template <typename scalar_t>
__global__ void kernel_volume_render(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,
    const scalar_t * __restrict__ deltas,
    const scalar_t * __restrict__ zs,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh, 
    scalar_t * alphas,
    scalar_t * weights,
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index * 3] = 0;
        image[index * 3 + 1] = 0;
        image[index * 3 + 2] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;
    zs += offset;
    weights += offset;
    alphas += offset;

    // accumulate 
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, d = 0;
    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;
        weights[0] = weight;
        alphas[0] = alpha;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        d += weight * zs[0];
        
        ws += weight;
        
        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        // printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, zs=%f, d=%f\n", n, step, alpha, weight, T, zs[0], d);

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        zs++;
        weights++;
        alphas++;
        step++;
    }
    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d / (ws + 1.0e-10);
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}

void volume_render(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor zs, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor alphas,at::Tensor weights, at::Tensor weights_sum, at::Tensor depth, at::Tensor image){

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "volume_render", ([&] {
        kernel_volume_render<<<div_round_up(N, N_THREAD), N_THREAD>>>(
            sigmas.data_ptr<scalar_t>(),
            rgbs.data_ptr<scalar_t>(),
            deltas.data_ptr<scalar_t>(),
            zs.data_ptr<scalar_t>(),
            rays.data_ptr<int>(),
            M,
            N,
            T_thresh,
            alphas.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            weights_sum.data_ptr<scalar_t>(),
            depth.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>()
            );
    }));
}

// template <typename scalar_t>
// __global__ void kernel_compute_important_samples(
//     const scalar_t * __restrict__ weights,
//     const int * __restrict__ rays,
//     const uint32_t M, const uint32_t N, 
//     int * important_samples
// ) {
//     // parallel per ray
//     const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
//     if (n >= N) return;

//     // locate 
//     uint32_t index = rays[n * 3];
//     uint32_t offset = rays[n * 3 + 1];
//     uint32_t num_steps = rays[n * 3 + 2];

//     int step = 0;

//     while(step < num_steps)

   
// }


// void compute_important_samples(const at::Tensor weights, const at::Tensor rays, const uint32_t M, const uint32_t N, at::Tensor important_samples){
//     static constexpr uint32_t N_THREAD = 128;
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//     weights.scalar_type(), "compute_important_samples", ([&] {
//         kernel_compute_important_samples<<<div_round_up(N, N_THREAD), N_THREAD>>>(
//             weights.data_ptr<scalar_t>(),
//             rays.data_ptr<int>(),
//             M,
//             N,
//             important_samples.data_ptr<scalar_t>()
//         );
//     }));
// }
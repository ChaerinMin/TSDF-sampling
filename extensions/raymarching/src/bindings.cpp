#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.def("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    // train
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("composite_rays_train_forward", &composite_rays_train_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_backward", &composite_rays_train_backward, "composite_rays_train_backward (CUDA)");
    // infer
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
    // tsdf
    m.def("integrate_tsdf", &integrate_tsdf, "integrate tsdf & weight grid (CUDA)");
    m.def("march_rays_tsdf", &march_rays_tsdf, "tsdf march rays (CUDA)");
    m.def("march_rays_tsdf_uniform", &march_rays_tsdf_uniform, "tsdf march rays unifromly (CUDA)");
    m.def("evaluate", &evaluate, "evaluate rays (CUDA)");    
    m.def("volume_render", &volume_render, "volume_render (CUDA)");    
    // m.def("compute_important_samples", &compute_important_samples, "compute_important_samples (CUDA)");    
}
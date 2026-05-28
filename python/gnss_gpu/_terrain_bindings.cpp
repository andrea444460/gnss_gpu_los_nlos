#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "gnss_gpu/terrain_cuda.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_terrain, m) {
  m.doc() = "CUDA terrain/elevation preprocess bindings";

  m.def(
      "terrain_prefilter_batch",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> rx_ecef,
         py::array_t<double, py::array::c_style | py::array::forcecast> sat_ecef,
         py::array_t<float, py::array::c_style | py::array::forcecast> dem_h,
         double dem_lat0_deg,
         double dem_lon0_deg,
         double dem_lat_step_deg,
         double dem_lon_step_deg,
         double max_distance_m,
         double sample_step_m,
         double elevation_mask_rad,
         double margin_rad) {
        auto brx = rx_ecef.request();
        auto bsat = sat_ecef.request();
        auto bdem = dem_h.request();

        if (brx.ndim != 2 || brx.shape[1] != 3)
          throw std::runtime_error("rx_ecef must have shape (N,3)");
        if (bsat.ndim != 3 || bsat.shape[2] != 3)
          throw std::runtime_error("sat_ecef must have shape (N,n_sat,3)");
        if (bdem.ndim != 2)
          throw std::runtime_error("dem_h must have shape (H,W)");
        if (brx.shape[0] != bsat.shape[0])
          throw std::runtime_error("rx_ecef and sat_ecef must share leading N");

        int n_epoch = (int)brx.shape[0];
        int n_sat = (int)bsat.shape[1];
        int dem_rows = (int)bdem.shape[0];
        int dem_cols = (int)bdem.shape[1];

        auto visible = py::array_t<int>({n_epoch, n_sat});
        auto terrain_blocked = py::array_t<int>({n_epoch, n_sat});
        auto terrain_blocked_visible = py::array_t<int>({n_epoch, n_sat});
        auto sat_masked = py::array_t<double>({n_epoch, n_sat, 3});

        gnss_gpu::terrain_prefilter_batch(
            static_cast<const double*>(brx.ptr),
            static_cast<const double*>(bsat.ptr),
            n_epoch,
            n_sat,
            static_cast<const float*>(bdem.ptr),
            dem_rows,
            dem_cols,
            dem_lat0_deg,
            dem_lon0_deg,
            dem_lat_step_deg,
            dem_lon_step_deg,
            max_distance_m,
            sample_step_m,
            elevation_mask_rad,
            margin_rad,
            static_cast<int*>(visible.mutable_data()),
            static_cast<int*>(terrain_blocked.mutable_data()),
            static_cast<int*>(terrain_blocked_visible.mutable_data()),
            static_cast<double*>(sat_masked.mutable_data()));

        return py::make_tuple(visible, terrain_blocked, terrain_blocked_visible, sat_masked);
      },
      "Run CUDA terrain/elevation preprocess in batch",
      py::arg("rx_ecef"),
      py::arg("sat_ecef"),
      py::arg("dem_h"),
      py::arg("dem_lat0_deg"),
      py::arg("dem_lon0_deg"),
      py::arg("dem_lat_step_deg"),
      py::arg("dem_lon_step_deg"),
      py::arg("max_distance_m"),
      py::arg("sample_step_m"),
      py::arg("elevation_mask_rad"),
      py::arg("margin_rad"));
}

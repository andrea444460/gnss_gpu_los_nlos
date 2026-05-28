#include "gnss_gpu/terrain_cuda.h"

#include "gnss_gpu/cuda_check.h"

#include <cmath>
#include <cfloat>
#include <vector>

namespace gnss_gpu {

namespace {

constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84F = 1.0 / 298.257223563;
constexpr double kWgs84E2 = 2.0 * kWgs84F - kWgs84F * kWgs84F;

__device__ inline double d_clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ void ecef_to_lla_rad(double x, double y, double z, double* lat, double* lon, double* alt) {
  *lon = atan2(y, x);
  double p = sqrt(x * x + y * y);
  double phi = atan2(z, p * (1.0 - kWgs84E2));
  for (int i = 0; i < 10; ++i) {
    double s = sin(phi);
    double n = kWgs84A / sqrt(1.0 - kWgs84E2 * s * s);
    phi = atan2(z + kWgs84E2 * n * s, p);
  }
  double s = sin(phi);
  double c = cos(phi);
  double n = kWgs84A / sqrt(1.0 - kWgs84E2 * s * s);
  *alt = (fabs(c) > 1e-10) ? (p / c - n) : (fabs(z) - n * (1.0 - kWgs84E2));
  *lat = phi;
}

__device__ void sat_elevation_azimuth(
    const double* rx, const double* sat, double* el_rad, double* az_rad) {
  double lat, lon, alt;
  (void)alt;
  ecef_to_lla_rad(rx[0], rx[1], rx[2], &lat, &lon, &alt);
  double sin_lat = sin(lat), cos_lat = cos(lat);
  double sin_lon = sin(lon), cos_lon = cos(lon);

  double dx = sat[0] - rx[0];
  double dy = sat[1] - rx[1];
  double dz = sat[2] - rx[2];

  // ENU
  double e = -sin_lon * dx + cos_lon * dy;
  double n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
  double u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;

  double horiz = sqrt(e * e + n * n);
  *el_rad = atan2(u, horiz);
  *az_rad = atan2(e, n);
}

__device__ inline bool dem_bilinear(
    const float* dem_h,
    int dem_rows,
    int dem_cols,
    double dem_lat0_deg,
    double dem_lon0_deg,
    double dem_lat_step_deg,
    double dem_lon_step_deg,
    double q_lat_deg,
    double q_lon_deg,
    double* z_out) {
  if (dem_rows < 2 || dem_cols < 2) return false;
  if (!isfinite(q_lat_deg) || !isfinite(q_lon_deg)) return false;

  double r = (q_lat_deg - dem_lat0_deg) / dem_lat_step_deg;
  double c = (q_lon_deg - dem_lon0_deg) / dem_lon_step_deg;
  if (!isfinite(r) || !isfinite(c)) return false;
  if (r < 0.0 || c < 0.0 || r >= (double)(dem_rows - 1) || c >= (double)(dem_cols - 1)) return false;

  int r0 = (int)floor(r);
  int c0 = (int)floor(c);
  int r1 = r0 + 1;
  int c1 = c0 + 1;
  double fr = r - (double)r0;
  double fc = c - (double)c0;

  float z00 = dem_h[r0 * dem_cols + c0];
  float z01 = dem_h[r0 * dem_cols + c1];
  float z10 = dem_h[r1 * dem_cols + c0];
  float z11 = dem_h[r1 * dem_cols + c1];
  if (!isfinite(z00) || !isfinite(z01) || !isfinite(z10) || !isfinite(z11)) return false;

  double z0 = (1.0 - fc) * (double)z00 + fc * (double)z01;
  double z1 = (1.0 - fc) * (double)z10 + fc * (double)z11;
  *z_out = (1.0 - fr) * z0 + fr * z1;
  return true;
}

__device__ bool terrain_visible_raymarch(
    const double* rx,
    double az_rad,
    double el_rad,
    const float* dem_h,
    int dem_rows,
    int dem_cols,
    double dem_lat0_deg,
    double dem_lon0_deg,
    double dem_lat_step_deg,
    double dem_lon_step_deg,
    double max_distance_m,
    double sample_step_m,
    double margin_rad) {
  double lat, lon, alt;
  ecef_to_lla_rad(rx[0], rx[1], rx[2], &lat, &lon, &alt);
  double lat_deg = lat * (180.0 / M_PI);
  double lon_deg = lon * (180.0 / M_PI);
  double cos_lat = cos(lat);
  if (fabs(cos_lat) < 1e-6) cos_lat = (cos_lat >= 0.0 ? 1e-6 : -1e-6);

  double horizon = -M_PI * 0.5;
  for (double d = sample_step_m; d <= max_distance_m + 1e-9; d += sample_step_m) {
    double north = d * cos(az_rad);
    double east = d * sin(az_rad);
    double q_lat = lat_deg + north / 111320.0;
    double q_lon = lon_deg + east / (111320.0 * cos_lat);
    double z_dem = NAN;
    if (!dem_bilinear(
            dem_h, dem_rows, dem_cols,
            dem_lat0_deg, dem_lon0_deg,
            dem_lat_step_deg, dem_lon_step_deg,
            q_lat, q_lon, &z_dem)) {
      continue;
    }
    double ang = atan2(z_dem - alt, d);
    horizon = ang > horizon ? ang : horizon;
  }
  return el_rad >= (horizon + margin_rad);
}

__global__ void terrain_prefilter_kernel(
    const double* rx_ecef,
    const double* sat_ecef,
    int n_epoch,
    int n_sat,
    const float* dem_h,
    int dem_rows,
    int dem_cols,
    double dem_lat0_deg,
    double dem_lon0_deg,
    double dem_lat_step_deg,
    double dem_lon_step_deg,
    double max_distance_m,
    double sample_step_m,
    double elevation_mask_rad,
    double margin_rad,
    int* visible,
    int* terrain_blocked,
    int* terrain_blocked_visible,
    double* sat_masked) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (tid >= total) return;

  int eid = tid / n_sat;
  int sid = tid % n_sat;

  const double* rx = rx_ecef + eid * 3;
  const double* sat = sat_ecef + (eid * n_sat + sid) * 3;
  double* sat_out = sat_masked + (eid * n_sat + sid) * 3;

  if (!isfinite(rx[0]) || !isfinite(rx[1]) || !isfinite(rx[2]) ||
      !isfinite(sat[0]) || !isfinite(sat[1]) || !isfinite(sat[2])) {
    visible[tid] = 0;
    terrain_blocked[tid] = 0;
    terrain_blocked_visible[tid] = 0;
    sat_out[0] = NAN;
    sat_out[1] = NAN;
    sat_out[2] = NAN;
    return;
  }

  double el, az;
  sat_elevation_azimuth(rx, sat, &el, &az);
  double az_mod = fmod(az, 2.0 * M_PI);
  if (az_mod < 0.0) az_mod += 2.0 * M_PI;

  bool el_visible = (el >= elevation_mask_rad);
  bool terr_visible = terrain_visible_raymarch(
      rx,
      az_mod,
      el,
      dem_h,
      dem_rows,
      dem_cols,
      dem_lat0_deg,
      dem_lon0_deg,
      dem_lat_step_deg,
      dem_lon_step_deg,
      max_distance_m,
      sample_step_m,
      margin_rad);

  bool terr_blocked = !terr_visible;
  bool terr_blocked_vis = el_visible && terr_blocked;
  bool vis = el_visible && terr_visible;

  visible[tid] = vis ? 1 : 0;
  terrain_blocked[tid] = terr_blocked ? 1 : 0;
  terrain_blocked_visible[tid] = terr_blocked_vis ? 1 : 0;
  if (vis) {
    sat_out[0] = sat[0];
    sat_out[1] = sat[1];
    sat_out[2] = sat[2];
  } else {
    sat_out[0] = NAN;
    sat_out[1] = NAN;
    sat_out[2] = NAN;
  }
}

}  // namespace

void terrain_prefilter_batch(
    const double* rx_ecef,
    const double* sat_ecef,
    int n_epoch,
    int n_sat,
    const float* dem_h,
    int dem_rows,
    int dem_cols,
    double dem_lat0_deg,
    double dem_lon0_deg,
    double dem_lat_step_deg,
    double dem_lon_step_deg,
    double max_distance_m,
    double sample_step_m,
    double elevation_mask_rad,
    double margin_rad,
    int* visible,
    int* terrain_blocked,
    int* terrain_blocked_visible,
    double* sat_masked) {
  if (n_epoch <= 0 || n_sat <= 0) return;

  size_t sz_rx = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_dem = (size_t)dem_rows * dem_cols * sizeof(float);
  size_t sz_mask = (size_t)n_epoch * n_sat * sizeof(int);
  size_t sz_sat_out = sz_sat;

  double* d_rx = nullptr;
  double* d_sat = nullptr;
  float* d_dem = nullptr;
  int* d_visible = nullptr;
  int* d_tblk = nullptr;
  int* d_tblk_vis = nullptr;
  double* d_sat_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_dem, sz_dem));
  CUDA_CHECK(cudaMalloc(&d_visible, sz_mask));
  CUDA_CHECK(cudaMalloc(&d_tblk, sz_mask));
  CUDA_CHECK(cudaMalloc(&d_tblk_vis, sz_mask));
  CUDA_CHECK(cudaMalloc(&d_sat_out, sz_sat_out));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dem, dem_h, sz_dem, cudaMemcpyHostToDevice));

  int total = n_epoch * n_sat;
  int block = 256;
  int grid = (total + block - 1) / block;
  terrain_prefilter_kernel<<<grid, block>>>(
      d_rx,
      d_sat,
      n_epoch,
      n_sat,
      d_dem,
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
      d_visible,
      d_tblk,
      d_tblk_vis,
      d_sat_out);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(visible, d_visible, sz_mask, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(terrain_blocked, d_tblk, sz_mask, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(terrain_blocked_visible, d_tblk_vis, sz_mask, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sat_masked, d_sat_out, sz_sat_out, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_dem));
  CUDA_CHECK(cudaFree(d_visible));
  CUDA_CHECK(cudaFree(d_tblk));
  CUDA_CHECK(cudaFree(d_tblk_vis));
  CUDA_CHECK(cudaFree(d_sat_out));
}

}  // namespace gnss_gpu

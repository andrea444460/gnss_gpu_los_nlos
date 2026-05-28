#pragma once

namespace gnss_gpu {

// Batched terrain + elevation preprocess on CUDA.
//
// Inputs:
// - rx_ecef: [n_epoch, 3]
// - sat_ecef: [n_epoch, n_sat, 3]
// - dem_h: [dem_rows, dem_cols] elevation grid in meters (WGS84 lat/lon grid)
// - dem_lat0_deg/dem_lon0_deg: top-left pixel center coordinates [deg]
// - dem_lat_step_deg/dem_lon_step_deg: per-pixel step [deg]
//
// Outputs:
// - visible: [n_epoch, n_sat] (1 if above elevation mask and terrain-visible)
// - terrain_blocked: [n_epoch, n_sat] (1 if terrain blocks LOS)
// - terrain_blocked_visible: [n_epoch, n_sat] (1 if blocked and above elevation mask)
// - sat_masked: [n_epoch, n_sat, 3] sat_ecef with NaN for not-visible entries
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
    double* sat_masked);

}  // namespace gnss_gpu

#ifndef GUFO_CORE_HIP_OCCUPANCY_HPP_
#define GUFO_CORE_HIP_OCCUPANCY_HPP_

/// Minimum resident waves per SIMD for a __launch_bounds__ hint tuned on
/// gfx1151, whose SIMDs hold 1536 VGPR rows: a minimum of n waves gives each
/// wave up to 1536 / n registers. gfx1150 has 1024 rows, so the same minimum
/// would leave each wave two thirds of the registers it was tuned with and
/// make the compiler spill (WmmaCausalAttention's six waves: 240 VGPRs on
/// gfx1151, 160 plus 72 spilled on gfx1150, 1.6x slower). Scale the minimum
/// there so every wave keeps its gfx1151 register budget; occupancy drops in
/// proportion, as it does for the gfx1151 code objects on that GPU.
/// gfx1151 code is unchanged.
#if defined(__gfx1150__)
#define GUFO_MIN_WAVES_PER_SIMD(n) ((n) * 2 / 3 > 0 ? (n) * 2 / 3 : 1)
#else
#define GUFO_MIN_WAVES_PER_SIMD(n) (n)
#endif

#endif  // GUFO_CORE_HIP_OCCUPANCY_HPP_

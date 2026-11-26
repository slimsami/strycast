#ifndef GUFO_CORE_DIAGNOSTICS_GPU_ARCHITECTURE_H_
#define GUFO_CORE_DIAGNOSTICS_GPU_ARCHITECTURE_H_

#include <string_view>

namespace gufo::diagnostics {

/// The GPU architecture in a HIP gcnArchName such as "gfx1151:xnack-" when
/// Gufo builds kernels for it, or an empty view. gfx1151 is Strix Halo
/// (Radeon 8050S/8060S), the production target; gfx1150 is Strix Point
/// (Radeon 880M/890M), the same RDNA 3.5 ISA with a smaller register file,
/// which needs its own code objects (CMAKE_HIP_ARCHITECTURES).
[[nodiscard]] constexpr std::string_view SupportedGpuArchitecture(
    std::string_view gcn_arch_name) {
  const std::string_view arch =
      gcn_arch_name.substr(0, gcn_arch_name.find(':'));
  if (arch == "gfx1151" || arch == "gfx1150")
    return arch;
  return {};
}

[[nodiscard]] constexpr bool IsSupportedGpuArchitecture(
    std::string_view architecture) {
  return !SupportedGpuArchitecture(architecture).empty();
}

/// Marketing family of a supported architecture, for messages.
[[nodiscard]] constexpr std::string_view GpuArchitectureFamily(
    std::string_view architecture) {
  if (architecture == "gfx1151")
    return "AMD Strix Halo";
  if (architecture == "gfx1150")
    return "AMD Strix Point";
  return "unsupported";
}

/// Requirement text for diagnostics and error messages.
inline constexpr std::string_view kSupportedGpuArchitecturesText =
    "gfx1151 (Strix Halo, Radeon 8050S/8060S) or gfx1150 (Strix Point, "
    "Radeon 880M/890M)";

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_GPU_ARCHITECTURE_H_

#pragma once
#include "modules/fire_control/control_types.hpp"

#include <nlohmann/json.hpp>

namespace rmvision_rl {
/** @brief Versioned semantic projection of the shared observation, never simulator evaluation.
 * Absolute clocks/identities are transport metadata, not model features. Missing ages are null.
 * This is not a frozen normalized tensor layout; fixed short history is a later adapter concern.
 */
[[nodiscard]] nlohmann::json Observation(const mv::modules::PolicyObservation& observation);
}  // namespace rmvision_rl

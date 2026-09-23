#pragma once
#include "modules/fire_control/control_types.hpp"

#include <nlohmann/json.hpp>

namespace rmvision_rl {
/**
 * @brief 将火控策略观测转换为带版本的 JSON。
 *
 * 仅输出策略可见字段；绝对时钟和标识符属于传输元数据，不作为模型特征。
 * 不可用的数据年龄编码为 null，归一化和短期历史由 Python 适配层处理。
 *
 * @param observation 当前控制周期的策略观测。
 * @return 策略通信使用的结构化观测，不含仿真评估真值。
 */
[[nodiscard]] nlohmann::json Observation(const mv::modules::PolicyObservation& observation);
}  // namespace rmvision_rl

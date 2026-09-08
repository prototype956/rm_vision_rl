#ifdef PHASE1_NEW
#include "modules/fire_control/control_session.hpp"
#else
#include "legacy_session.hpp"
#endif
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
using namespace mv;
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const auto f = modules::ParseFireControlConfig(YAML::LoadFile(std::string(argv[1])+"/fire_control.yaml"));
  const auto p = modules::ParseGimbalTrajectoryPlannerConfig(YAML::LoadFile(std::string(argv[1])+"/gimbal_trajectory_planner.yaml"));
  std::cout << std::setprecision(17);
  for (int scene=0; scene<8; ++scene) {
    modules::ControlSession control(f,p);
    hal::GimbalFeedback feedback; feedback.valid=true;
    for (int step=0;step<600;++step) {
      auto now=std::chrono::steady_clock::time_point(std::chrono::milliseconds(1000+step*10));
      double t=step*.01;
      modules::ControlInputSnapshot input;
      input.external_control_enabled=true;
      auto& e=input.prediction;
      e.sequence=step; e.source_steady_time=now;
      e.state=modules::TrackerState::TRACKING;
      e.label=modules::ArmorLabel::THREE; e.type=geometry::ArmorType::SMALL;
      e.radii_m={.21,.23}; e.armor_tilt_rad=-.265;
      e.center_world={4., scene%2 ? std::sin(t)*1.2:0., .1};
      e.velocity_world={0., scene%2 ? std::cos(t)*1.2:0.,0.};
      e.yaw_velocity_rad_s=scene>=2?2.:0.;
      e.orientation_world=Eigen::AngleAxisd(std::numbers::pi+e.yaw_velocity_rad_s*t,geometry::Vector3::UnitZ());
      if(scene==4 && step>=200 && step<210) e.state=modules::TrackerState::TEMP_LOST;
      if(scene==5 && step>=200 && step<240) e.source_steady_time=now-std::chrono::milliseconds(200);
      if(scene==6 && step>=200 && step<250) input.external_control_enabled=false;
      if(scene==7 && step>=200 && step<250) e.center_covariance_world=Eigen::Matrix3d::Identity();
      feedback.timestamp=now;
#ifdef PHASE1_NEW
      input.referee.valid=true; input.referee.received_at=now; input.referee.fire_permitted=true;
      input.referee.alive=true; input.referee.unlimited=true; input.referee.heat_limit=88;

#else

#endif
      input.world_t_gimbal.rotation=Eigen::AngleAxisd(feedback.yaw,geometry::Vector3::UnitZ())*Eigen::AngleAxisd(-feedback.pitch,geometry::Vector3::UnitY());
      auto result=control.Step(input,{},true,now,1000000000ULL+step*10000000ULL);
      control.AcknowledgePublication(result,!(scene==6 && step>=350 && step<355),now);
      const auto& o=result.output; const auto& a=result.diagnostics.armor_selection;
      std::cout<<scene<<','<<step<<','<<o.selected_slot<<','<<int(o.tracker_state)<<','<<int(o.reject_reason)<<','<<o.command.valid<<','<<o.command.fire<<','<<int(o.command_source)<<','<<o.stable_cycles<<','<<a.pending_slot<<','<<int(a.decision)<<','<<o.target_yaw<<','<<o.target_pitch<<','<<o.command.yaw<<','<<o.command.pitch<<','<<o.command.yaw_velocity<<','<<o.command.pitch_velocity<<','<<o.command.yaw_acceleration<<','<<o.command.pitch_acceleration<<','<<o.ballistic.fly_time_s<<','<<o.fallback_active<<','<<o.fallback_source_slot<<','<<o.fallback_trajectory_index<<','<<o.published_valid;
      for(const auto& point:o.plan.trajectory) std::cout<<','<<point.yaw<<','<<point.pitch<<','<<point.yaw_velocity<<','<<point.pitch_velocity;
      std::cout<<'\n';
      if(o.command.valid) {feedback.yaw=o.command.yaw;feedback.pitch=o.command.pitch;feedback.yaw_velocity=o.command.yaw_velocity;feedback.pitch_velocity=o.command.pitch_velocity;}
    }
  }
}

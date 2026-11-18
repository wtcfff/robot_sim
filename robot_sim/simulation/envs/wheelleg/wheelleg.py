# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import quat_rotate_inverse, quat_rotate, quat_mul, quat_from_euler_xyz

from robot_sim.simulation.envs.base.legged_robot import LeggedRobot
from .wheelleg_config import WheellegCfg


class Wheelleg(LeggedRobot):
    """Open-chain wheel-legged robot with leg position and wheel velocity control."""

    wheel_indices = (3, 7)
    leg_indices = (0, 1, 2, 4, 5, 6)

    from robot_sim.simulation.terrain import TERRAIN_TYPES as _terrain_names

    def _parse_cfg(self, cfg):
        from robot_sim.simulation.terrain import resolve_terrain
        resolve_terrain(cfg.terrain)
        super()._parse_cfg(cfg)

    def _get_env_origins(self):
        super()._get_env_origins()
        if self.cfg.terrain.mesh_type not in ("heightfield", "trimesh"):
            return

        # 基任务按旧版 20 列写死分类；本任务改为按生成器的累计比例分类。
        # 与 Terrain.curiculum() 使用同一个列选择公式，支持改比例和列数。
        column_types = torch.tensor(self.terrain.column_ids, device=self.device)
        env_types = column_types[self.terrain_types]
        self.terrain_type_indices = {
            name: (env_types == index).nonzero(as_tuple=False).flatten()
            for index, name in enumerate(self._terrain_names)
        }
        indices = self.terrain_type_indices
        self.flat_idx = indices["flat"]
        self.smooth_slope_idx = indices["smooth_slope"]
        self.rough_slope_idx = indices["rough_slope"]
        # 保留基任务以中心出生后的运动方向命名的阶梯索引。
        self.stair_up_idx = indices["stairs_negative"]
        self.stair_down_idx = indices["stairs_positive"]
        self.discrete_idx = indices["discrete_obstacles"]
        # 延续原课程分组：中心高的阶梯属于基础组；新增踏石、沟、坑归高级组。
        basic_types = {"flat", "smooth_slope", "rough_slope", "stairs_positive"}
        self.basic_terrain_idx = torch.cat(
            [indices[name] for name in self._terrain_names if name in basic_types]
        )
        self.advanced_terrain_idx = torch.cat(
            [indices[name] for name in self._terrain_names if name not in basic_types]
        )

    def __init__(self, cfg: WheellegCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )
        self.last_policy_wheel_torques = torch.zeros(
            self.num_envs, len(self.wheel_indices), device=self.device
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self._resample_control_randomization(env_ids)
            self.action_fifo[env_ids] = 0.0
            self.actions[env_ids] = 0.0
            self.torques[env_ids] = 0.0
            self.dof_acc[env_ids] = 0.0
            self.last_dof_vel[env_ids] = self.dof_vel[env_ids]
            self.last_policy_wheel_torques[env_ids] = 0.0
            self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
            self.wheel_state_valid[env_ids] = False
            self.push_wrench_body[env_ids] = 0.0
            self.applied_wrench_body[env_ids] = 0.0
            self.push_end_step[env_ids] = 0
            self.next_push_step[env_ids] = self._sample_push_intervals(len(env_ids))
            self._record_sensor_sample(env_ids)
            # The parent fills history with uncorrupted data; replace reset rows.
            initial = self.compute_proprioception_observations()[env_ids].clone()
            initial[:, :6] = self.sensor_fifo[env_ids, 0, :6]
            initial[:, 10:24] = self.sensor_fifo[env_ids, 0, 6:]
            self.obs_history[env_ids] = initial.repeat(1, self.obs_history_length)

    def compute_reward(self):
        """Use a wider local clip only for the stand-still positive reward."""
        self.rew_buf[:] = 0.0
        local_clip_names = {"stand_still"}
        global_clip = self.cfg.rewards.clip_single_reward
        local_clip = self.cfg.rewards.local_clip_single_reward
        for name, reward_function in zip(self.reward_names, self.reward_functions):
            rew = reward_function() * self.reward_scales[name]
            clip_value = local_clip if name in local_clip_names else global_clip
            rew = torch.clip(
                rew,
                -clip_value * self.dt,
                clip_value * self.dt,
            )
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        self.last_policy_wheel_torques.copy_(
            self.torques[:, self.wheel_indices]
        )

    def _init_buffers(self):
        self.cfg.control.damping = dict(self.cfg.control.damping)
        self.cfg.control.damping["foot_Joint"] = self.cfg.control.wheel_velocity_kp
        super()._init_buffers()
        self.nominal_p_gains = torch.tensor([
            next((value for key, value in self.cfg.control.stiffness.items() if key in name), 0.0)
            for name in self.dof_names], device=self.device)
        self.nominal_d_gains = torch.tensor([
            next((value for key, value in self.cfg.control.damping.items() if key in name), 0.0)
            for name in self.dof_names], device=self.device)
        self.wheel_velocity_gains = self.d_gains[:, self.wheel_indices].clone()
        action_max = self._delay_limits(self.cfg.domain_rand.delay_ms_range)[1]
        self.action_fifo = torch.zeros(self.num_envs, action_max + 1, self.num_actions, device=self.device)
        obs_max = self._delay_limits(self.cfg.domain_rand.observation_delay_ms_range)[1]
        self.sensor_fifo = torch.zeros(self.num_envs, obs_max + 1, 20, device=self.device)
        self.sensor_noise_scale = torch.cat((self.noise_scale_vec[:6], self.noise_scale_vec[10:24]))
        self.observation_delay_idx = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.wheel_state_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.push_wrench_body = torch.zeros(self.num_envs, 6, device=self.device)
        self.applied_wrench_body = torch.zeros_like(self.push_wrench_body)
        self.push_end_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.next_push_step = self._sample_push_intervals(self.num_envs)
        self.output_torque_noise_std = torch.full((self.num_dof,), self.cfg.domain_rand.leg_output_torque_noise_std, device=self.device)
        self.output_torque_noise_std[list(self.wheel_indices)] = self.cfg.domain_rand.wheel_output_torque_noise_std
        self._resample_control_randomization(torch.arange(self.num_envs, device=self.device))
        self._record_sensor_sample(torch.arange(self.num_envs, device=self.device))

    def _compute_torques(self, actions):
        """Leg position PD and wheel velocity P, followed by motor/URDF limits."""
        leg = self.leg_indices
        wheel = self.wheel_indices
        torques = torch.zeros_like(actions)
        leg_position_targets = (
            actions[:, leg] * self.cfg.control.pos_action_scale
            + self.default_dof_pos[:, leg]
        )
        torques[:, leg] = (
            self.p_gains[:, leg] * (leg_position_targets - self.dof_pos[:, leg])
            - self.d_gains[:, leg] * self.dof_vel[:, leg]
        )
        max_wheel_speed = (
            self.cfg.control.wheel_no_load_speed_rpm * (2.0 * torch.pi / 60.0)
        )
        wheel_velocity_targets = torch.clamp(
            actions[:, wheel] * self.cfg.control.vel_action_scale,
            min=-max_wheel_speed,
            max=max_wheel_speed,
        )
        torques[:, wheel] = self.wheel_velocity_gains * (
            wheel_velocity_targets - self.dof_vel[:, wheel]
        )
        torques = torques * self.torques_scale
        if self.cfg.domain_rand.randomize_output_torque_noise:
            torques += torch.randn_like(torques) * self.output_torque_noise_std
        torques = torch.clip(
            torques,
            -self.torque_limits,
            self.torque_limits,
        )
        return torques

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # Internal order remains [vx, yaw_rate, height, heading, reserved_vy].
        # Assign directly: the reserved command must not consume RNG samples.
        self.commands[env_ids, 4] = 0.0

    def compute_command_observations(self):
        """Return [vx, reserved zero vy, yaw_rate, height] at policy scales."""
        motion_and_height = self.commands[:, :3] * self.commands_scale
        return torch.cat(
            (motion_and_height[:, :1], torch.zeros_like(self.commands[:, :1]),
             motion_and_height[:, 1:]), dim=-1
        )

    def compute_proprioception_observations(self):
        return torch.cat(
            (self.base_ang_vel * self.obs_scales.ang_vel,
             self.projected_gravity,
             self.compute_command_observations(),
             (self.dof_pos[:, self.leg_indices]
              - self.default_dof_pos[:, self.leg_indices]) * self.obs_scales.dof_pos,
             self.dof_vel * self.obs_scales.dof_vel,
             self.actions), dim=-1
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        if not self.add_noise:
            return noise_vec

        scales = cfg.noise.noise_scales
        level = cfg.noise.noise_level
        n = cfg.env.num_actions
        dof_pos_start = 10
        dof_vel_start = dof_pos_start + len(self.leg_indices)
        dof_vel_end = dof_vel_start + n
        noise_vec[:3] = scales.ang_vel * level * self.obs_scales.ang_vel
        noise_vec[3:6] = scales.gravity * level
        noise_vec[dof_pos_start:dof_vel_start] = (
            scales.dof_pos * level * self.obs_scales.dof_pos
        )
        noise_vec[dof_vel_start:dof_vel_end] = (
            scales.dof_vel * level * self.obs_scales.dof_vel
        )
        for wheel_index in self.wheel_indices:
            noise_vec[dof_vel_start + wheel_index] = (
                scales.wheel_dof_vel * level * self.obs_scales.dof_vel
            )
        return noise_vec

    def _uniform(self, shape, bounds):
        return torch.empty(shape, device=self.device).uniform_(*bounds)

    def _create_envs(self):
        super()._create_envs()
        if self.num_bodies != self.cfg.env.num_robot_bodies:
            raise ValueError("wheelleg privileged layout requires the nine-body asset")
        self.body_names = self.gym.get_actor_rigid_body_names(
            self.envs[0], self.actor_handles[0]
        )
        self.body_material = torch.zeros(self.num_envs, self.num_bodies, 2, device=self.device)
        dr = self.cfg.domain_rand
        material_samples = torch.empty(self.num_envs, self.num_bodies, 2)
        for body_id, name in enumerate(self.body_names):
            wheel = "foot_Link" in name
            material_samples[:, body_id, 0].uniform_(*(dr.friction_range if wheel else dr.body_friction_range))
            material_samples[:, body_id, 1].uniform_(*(dr.restitution_range if wheel else dr.body_restitution_range))
        for env_id, (env, actor) in enumerate(zip(self.envs, self.actor_handles)):
            shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
            mapping = self.gym.get_actor_rigid_body_shape_indices(env, actor)
            for body_id, span in enumerate(mapping):
                first = shapes[span.start] if span.count else None
                friction = (float(material_samples[env_id, body_id, 0]) if dr.randomize_friction
                            else (first.friction if first else 0.0))
                restitution = (float(material_samples[env_id, body_id, 1]) if dr.randomize_restitution
                               else (first.restitution if first else 0.0))
                material_samples[env_id, body_id, 0] = friction
                material_samples[env_id, body_id, 1] = restitution
                for shape in shapes[span.start:span.start + span.count]:
                    shape.friction = friction
                    shape.restitution = restitution
            self.gym.set_actor_rigid_shape_properties(env, actor, shapes)
        self.body_material = material_samples.to(self.device)
        # Compatibility fields are actual wheel material, not fictitious zeros.
        self.friction_coef = self.body_material[:, self.feet_indices[0], 0]
        self.restitution_coef = self.body_material[:, self.feet_indices[0], 1]

    def _process_rigid_shape_props(self, props, env_id):
        # Per-body material assignment needs the actor's actual shape mapping.
        return props

    def _process_rigid_body_props(self, props, env_id):
        dr = self.cfg.domain_rand
        if env_id == 0:
            # Batch CPU sampling avoids a CUDA synchronization for every link.
            self._mass_samples = torch.ones(self.num_envs, len(props))
            self._com_samples = torch.zeros(self.num_envs, len(props), 3)
            if dr.randomize_link_mass:
                self._mass_samples.uniform_(*dr.link_mass_range)
                self._mass_samples[:, 0].uniform_(*dr.base_mass_range)
            if dr.randomize_link_com:
                self._com_samples.uniform_(*dr.link_com_range)
                self._com_samples[:, 0] = torch.empty(self.num_envs, 3).uniform_(-1.0, 1.0) * torch.tensor(dr.base_com_xyz)
            self.body_mass_ratio = self._mass_samples.to(self.device)
            self.body_com_offset = self._com_samples.to(self.device)
            self.body_masses = torch.zeros_like(self.body_mass_ratio)
        for body_id, prop in enumerate(props):
            prop.mass *= float(self._mass_samples[env_id, body_id])
            self.body_masses[env_id, body_id] = prop.mass
            offset = self._com_samples[env_id, body_id]
            prop.com.x += float(offset[0])
            prop.com.y += float(offset[1])
            prop.com.z += float(offset[2])
        self.base_mass[env_id] = props[0].mass
        self.base_com[env_id] = self.body_com_offset[env_id, 0]
        # The caller uses recomputeInertia=True; no second independent mass scale.
        return props

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            self._joint_friction_samples = torch.tensor(props["friction"].copy()).repeat(self.num_envs, 1)
            if self.cfg.domain_rand.randomize_joint_friction:
                self._joint_friction_samples.uniform_(*self.cfg.domain_rand.leg_joint_friction_range)
                self._joint_friction_samples[:, list(self.wheel_indices)] = torch.empty(self.num_envs, 2).uniform_(*self.cfg.domain_rand.wheel_joint_friction_range)
            self.joint_friction_truth = self._joint_friction_samples.to(self.device)
        props["friction"][:] = self._joint_friction_samples[env_id].numpy()
        return props

    def _delay_limits(self, bounds_ms):
        step_ms = self.sim_params.dt * 1000.0
        low = math.ceil(bounds_ms[0] / step_ms - 1e-9)
        high = math.floor(bounds_ms[1] / step_ms + 1e-9)
        if low < 0 or high < low:
            raise ValueError("Delay range must contain nonnegative integer physics steps")
        return low, high

    def _resample_control_randomization(self, env_ids):
        dr = self.cfg.domain_rand
        count = len(env_ids)
        self.p_gains[env_ids] = self.nominal_p_gains
        self.d_gains[env_ids] = self.nominal_d_gains
        if dr.randomize_Kp:
            self.p_gains[env_ids] *= self._uniform((count, self.num_dof), dr.randomize_Kp_range)
        if dr.randomize_Kd:
            self.d_gains[env_ids] *= self._uniform((count, self.num_dof), dr.randomize_Kd_range)
        self.wheel_velocity_gains[env_ids] = self.d_gains[env_ids][:, self.wheel_indices]
        self.torques_scale[env_ids] = 1.0
        if dr.randomize_motor_torque:
            self.torques_scale[env_ids] = self._uniform((count, self.num_dof), dr.randomize_motor_torque_range)
            self.torques_scale[env_ids[:, None], torch.tensor(self.wheel_indices, device=self.device)] = self._uniform((count, 2), dr.wheel_motor_strength_range)
        self.action_delay_idx[env_ids] = 0
        if dr.randomize_action_delay:
            low, high = self._delay_limits(dr.delay_ms_range)
            self.action_delay_idx[env_ids] = torch.randint(low, high + 1, (count,), device=self.device)
        self.observation_delay_idx[env_ids] = 0
        if dr.randomize_observation_delay:
            low, high = self._delay_limits(dr.observation_delay_ms_range)
            self.observation_delay_idx[env_ids] = torch.randint(low, high + 1, (count, 2), device=self.device)

    def _sensor_sample(self):
        gyro = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 10:13])
        gravity = quat_rotate_inverse(self.root_states[:, 3:7], self.gravity_vec)
        return torch.cat((gyro * self.obs_scales.ang_vel, gravity,
                          (self.dof_pos[:, self.leg_indices] - self.default_dof_pos[:, self.leg_indices]) * self.obs_scales.dof_pos,
                          self.dof_vel * self.obs_scales.dof_vel), dim=-1)

    def _record_sensor_sample(self, env_ids=None):
        sample = self._sensor_sample()
        if self.add_noise:
            sample += (2 * torch.rand_like(sample) - 1) * self.sensor_noise_scale
        if env_ids is None:
            self.sensor_fifo[:, 1:] = self.sensor_fifo[:, :-1].clone()
            self.sensor_fifo[:, 0] = sample
        else:
            # Warm start every lag with the new episode, never the old episode.
            self.sensor_fifo[env_ids] = sample[env_ids, None, :]

    def compute_dof_vel(self):
        super().compute_dof_vel()
        if hasattr(self, "sensor_fifo"):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self._record_sensor_sample()
            self.wheel_state_valid[:] = True

    def _push_robots(self):
        # Called once per physics step. Bounded body-frame wrench for 0.1 s.
        now = self.envs_steps_buf
        start = now >= self.next_push_step
        if start.any():
            ids = start.nonzero(as_tuple=False).flatten()
            self.push_wrench_body[ids] = 0.0
            self.push_wrench_body[ids, :2] = self._uniform((len(ids), 2), self.cfg.domain_rand.push_force_range)
            self.push_wrench_body[ids, 5] = self._uniform((len(ids),), self.cfg.domain_rand.push_torque_range)
            self.push_end_step[ids] = now[ids] + max(1, round(self.cfg.domain_rand.push_duration_s / self.sim_params.dt))
            self.next_push_step[ids] = now[ids] + self._sample_push_intervals(len(ids))
        active = now < self.push_end_step
        self.applied_wrench_body = self.push_wrench_body * active[:, None]
        self.rigid_body_external_forces.zero_()
        self.rigid_body_external_torques.zero_()
        self.rigid_body_external_forces[:, 0] = quat_rotate(self.base_quat, self.applied_wrench_body[:, :3])
        self.rigid_body_external_torques[:, 0] = quat_rotate(self.base_quat, self.applied_wrench_body[:, 3:])
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques), gymapi.ENV_SPACE
        )

    def _sample_push_intervals(self, count):
        seconds = self._uniform((count,), self.cfg.domain_rand.push_interval_range_s)
        return torch.ceil(seconds / self.sim_params.dt).long()

    def _reset_dofs(self, env_ids):
        dr = self.cfg.domain_rand
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.0
        if dr.randomize_initial_state:
            self.dof_pos[env_ids[:, None], torch.tensor(self.leg_indices, device=self.device)] += self._uniform((len(env_ids), 6), dr.initial_leg_position_offset_range)
            leg_positions = self.dof_pos[env_ids][:, self.leg_indices]
            self.dof_pos[env_ids[:, None], torch.tensor(self.leg_indices, device=self.device)] = torch.clamp(
                leg_positions, self.dof_pos_limits[list(self.leg_indices), 0], self.dof_pos_limits[list(self.leg_indices), 1]
            )
            self.dof_pos[env_ids[:, None], torch.tensor(self.wheel_indices, device=self.device)] = self._uniform((len(env_ids), 2), dr.initial_wheel_position_range)
            self.dof_vel[env_ids] = self._uniform((len(env_ids), self.num_dof), dr.initial_joint_velocity_range)
        ids = env_ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(ids), len(ids))

    def _reset_root_states(self, env_ids):
        dr = self.cfg.domain_rand
        count = len(env_ids)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if dr.randomize_initial_state:
            self.root_states[env_ids, :2] += self._uniform((count, 2), dr.initial_xy_range)
            self.root_states[env_ids, 2] += self._uniform((count,), dr.initial_z_range)
            roll_pitch = self._uniform((count, 2), dr.initial_roll_pitch_range)
            yaw = self._uniform((count,), dr.initial_yaw_range)
            self.root_states[env_ids, 3:7] = quat_mul(
                self.base_init_state[3:7].expand(count, -1),
                quat_from_euler_xyz(roll_pitch[:, 0], roll_pitch[:, 1], yaw)
            )
            self.root_states[env_ids, 7:10] = self._uniform((count, 3), dr.initial_linear_velocity_range)
            self.root_states[env_ids, 10:13] = self._uniform((count, 3), dr.initial_angular_velocity_range)
        ids = env_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(ids), count)
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _compute_privileged_observations(self, clean_obs):
        wheel_force_world = self.contact_forces[:, self.feet_indices].clone()
        wheel_velocity_world = self.rigid_body_states[:, self.feet_indices, 7:10].clone()
        # Rigid-body/contact tensors can still describe the previous episode at reset.
        wheel_force_world[~self.wheel_state_valid] = 0.0
        wheel_velocity_world[~self.wheel_state_valid] = 0.0
        rotations = self.base_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
        force_body = quat_rotate_inverse(rotations, wheel_force_world.reshape(-1, 3)).reshape(self.num_envs, 6)
        velocity_body = quat_rotate_inverse(rotations, wheel_velocity_world.reshape(-1, 3)).reshape(self.num_envs, 6)
        contacts = (torch.norm(wheel_force_world, dim=-1) > self.cfg.rewards.wheel_contact_force_threshold).float()
        heights = torch.clamp(self.root_states[:, 2:3] - 0.5 - self.measured_heights, -1.0, 1.0) * self.obs_scales.height_measurements
        blocks = (
            self.base_lin_vel * self.obs_scales.lin_vel, clean_obs,
            self.last_actions[:, :, 0], self.last_actions[:, :, 1],
            self.dof_acc * self.obs_scales.dof_acc, heights,
            self.torques * self.obs_scales.torque,
            self.default_dof_pos - self.raw_default_dof_pos,
            self.body_mass_ratio - 1.0, self.body_com_offset.flatten(1) * 10.0,
            self.body_material.flatten(1), contacts, force_body * 0.01,
            velocity_body * self.obs_scales.lin_vel,
            torch.cat((self.p_gains[:, self.leg_indices] / self.nominal_p_gains[list(self.leg_indices)].clamp_min(1e-6),
                       self.d_gains[:, self.leg_indices] / self.nominal_d_gains[list(self.leg_indices)].clamp_min(1e-6), self.wheel_velocity_gains / self.cfg.control.wheel_velocity_kp), dim=-1),
            self.torques_scale, self.joint_friction_truth,
            torch.cat((self.action_delay_idx[:, None], self.observation_delay_idx), dim=-1).float() * self.sim_params.dt * 100.0,
            self.applied_wrench_body * torch.tensor([0.05, 0.05, 0.05, 1.0, 1.0, 1.0], device=self.device),
        )
        result = torch.cat(blocks, dim=-1)
        if result.shape[1] != self.num_privileged_obs:
            raise RuntimeError("wheelleg privileged observation dimension mismatch")
        return result

    def compute_observations(self, update_history=True):
        clean_obs = self.compute_proprioception_observations()
        self.privileged_obs_buf = self._compute_privileged_observations(clean_obs)
        self.obs_buf = clean_obs.clone()
        rows = torch.arange(self.num_envs, device=self.device)
        self.obs_buf[:, :6] = self.sensor_fifo[rows, self.observation_delay_idx[:, 0], :6]
        self.obs_buf[:, 10:24] = self.sensor_fifo[rows, self.observation_delay_idx[:, 1], 6:]
        clip = self.cfg.normalization.clip_observations
        self.obs_buf.clamp_(-clip, clip)
        self.privileged_obs_buf.clamp_(-clip, clip)
        if not update_history or not self.obs_history_length:
            return
        update = ((self.envs_steps_buf // self.cfg.control.decimation) % self.cfg.env.obs_history_dec) == 0
        self.obs_history[update] = torch.cat((self.obs_history[update, self.num_obs:], self.obs_buf[update]), dim=-1)

    def _reward_dof_vel(self):
        leg_indices = [i for i in range(self.num_dof) if i not in self.wheel_indices]
        return torch.sum(torch.square(self.dof_vel[:, leg_indices]), dim=1)

    def _reward_dof_acc(self):
        """Penalize acceleration of the leg joints only."""
        leg_indices = [i for i in range(self.num_dof) if i not in self.wheel_indices]
        return torch.sum(torch.square(self.dof_acc[:, leg_indices]), dim=1)

    def _reward_wheel_dof_acc(self):
        """Penalize actual angular acceleration of the two wheel joints."""
        return torch.sum(torch.square(self.dof_acc[:, self.wheel_indices]), dim=1)

    def _reward_torques(self):
        """Penalize torque applied to the leg joints only."""
        leg_indices = [i for i in range(self.num_dof) if i not in self.wheel_indices]
        return torch.sum(torch.square(self.torques[:, leg_indices]), dim=1)

    def _reward_wheel_torques(self):
        """Penalize torque applied to the wheel joints."""
        return torch.sum(torch.square(self.torques[:, self.wheel_indices]), dim=1)

    def _reward_wheel_base_rel_vel(self):
        """Penalize contacting wheel-center fore-aft speed relative to base.

        Subtracting the base-origin velocity and rotational transport term
        isolates wheel motion caused by the leg joints. The result is expressed
        in the base frame so its x component follows the robot fore-aft axis.
        Wheels without a sufficient contact force do not contribute.
        """
        wheel_states = self.rigid_body_states[:, self.feet_indices]
        wheel_positions = wheel_states[..., :3]
        base_position = self.root_states[:, None, :3]
        relative_positions = wheel_positions - base_position
        relative_velocity_world = (
            wheel_states[..., 7:10]
            - self.root_states[:, None, 7:10]
            - torch.cross(
                self.root_states[:, None, 10:13].expand_as(relative_positions),
                relative_positions,
                dim=-1,
            )
        )
        base_quat = self.base_quat[:, None, :].expand(
            -1, relative_positions.shape[1], -1
        )
        relative_velocity_body = quat_rotate_inverse(
            base_quat.reshape(-1, 4), relative_velocity_world.reshape(-1, 3)
        ).reshape_as(relative_velocity_world)

        excess_speed = (
            relative_velocity_body[..., 0].abs()
            - self.cfg.rewards.wheel_base_rel_vel_deadband
        ).clamp(min=0.0)
        normalized_speed = excess_speed / self.cfg.rewards.wheel_base_rel_vel_scale
        contact = torch.linalg.vector_norm(
            self.contact_forces[:, self.feet_indices], dim=-1
        ) > self.cfg.rewards.wheel_contact_force_threshold
        return torch.sum(torch.square(normalized_speed) * contact, dim=1)

    def _reward_wheel_base_rel_pos(self):
        """Penalize wheel centers over 3 cm from their default-pose x positions."""
        wheel_positions = self.rigid_body_states[:, self.feet_indices, :3]
        relative_positions = wheel_positions - self.root_states[:, None, :3]
        base_quat = self.base_quat[:, None, :].expand(
            -1, relative_positions.shape[1], -1
        )
        relative_positions_body = quat_rotate_inverse(
            base_quat.reshape(-1, 4), relative_positions.reshape(-1, 3)
        ).reshape_as(relative_positions)

        nominal_x = torch.as_tensor(
            self.cfg.rewards.wheel_base_rel_pos_nominal_x,
            device=self.device,
            dtype=relative_positions_body.dtype,
        ).view(1, -1)
        default_pose_deviation = (relative_positions_body[..., 0] - nominal_x).abs()
        excess_x = (
            default_pose_deviation - self.cfg.rewards.wheel_base_rel_pos_limit
        ).clamp(min=0.0)
        normalized_excess = excess_x / self.cfg.rewards.wheel_base_rel_pos_scale
        contact = torch.linalg.vector_norm(
            self.contact_forces[:, self.feet_indices], dim=-1
        ) > self.cfg.rewards.wheel_contact_force_threshold
        return torch.sum(torch.square(normalized_excess) * contact, dim=1)

    def _reward_wheel_torque_rate(self):
        """Penalize per-policy-step changes in applied wheel torque."""
        torque_delta = (
            self.torques[:, self.wheel_indices]
            - self.last_policy_wheel_torques
        )
        normalized_delta = torque_delta / self.cfg.rewards.wheel_torque_rate_scale
        return torch.sum(torch.square(normalized_delta), dim=1)

    def _reward_action_rate(self):
        """Penalize action changes for the leg joints only."""
        leg_indices = [i for i in range(self.num_dof) if i not in self.wheel_indices]
        return torch.sum(
            torch.square(
                self.actions[:, leg_indices]
                - self.last_actions[:, leg_indices, 0]
            ),
            dim=1,
        )

    def _reward_wheel_action_rate(self):
        """Penalize first differences of wheel actions only."""
        return torch.sum(
            torch.square(
                self.actions[:, self.wheel_indices]
                - self.last_actions[:, self.wheel_indices, 0]
            ),
            dim=1,
        )

    def _reward_base_height(self):
        """Give a smooth positive reward for tracking the commanded height."""
        height_error = self.base_height - self.commands[:, 2]
        return torch.exp(
            -torch.square(height_error) / self.cfg.rewards.base_height_sigma
        )

    def _reward_action_smooth(self):
        leg_indices = [i for i in range(self.num_dof) if i not in self.wheel_indices]
        return torch.sum(
            torch.square(
                self.actions[:, leg_indices]
                - 2 * self.last_actions[:, leg_indices, 0]
                + self.last_actions[:, leg_indices, 1]
            ),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        leg_indices = [i for i in range(self.num_dof) if i not in self.wheel_indices]
        dof_pos = self.dof_pos[:, leg_indices]
        limits = self.dof_pos_limits[leg_indices]
        below_limits = (limits[:, 0] - dof_pos).clamp(min=0.0)
        above_limits = (dof_pos - limits[:, 1]).clamp(min=0.0)
        return torch.sum(below_limits + above_limits, dim=1)

    def _reward_stand_still(self):
        """Reward low planar speed when the commanded motion is near zero."""
        zero_motion_command = (
            torch.linalg.vector_norm(self.commands[:, :2], dim=1)
            < self.cfg.rewards.stand_still_command_threshold
        )
        speed = torch.linalg.vector_norm(self.base_lin_vel[:, :2], dim=1)
        normalized_speed = speed / self.cfg.rewards.stand_still_speed_scale
        return zero_motion_command * torch.exp(-torch.square(normalized_speed))

    def _reward_stand_still_drift(self):
        """Penalize fore-aft base velocity under a zero motion command."""
        zero_motion_command = (
            (self.commands[:, 0].abs() < self.cfg.rewards.stand_still_command_threshold)
            & (
                self.commands[:, 1].abs()
                < self.cfg.rewards.direction_yaw_command_threshold
            )
        )
        normalized_drift = (
            self.base_lin_vel[:, 0]
            / self.cfg.rewards.stand_still_drift_speed_scale
        )
        return zero_motion_command * torch.square(normalized_drift)

    def _reward_stand_still_acc(self):
        """Penalize planar base acceleration while both motion commands are zero."""
        zero_motion_command = (
            torch.linalg.vector_norm(self.commands[:, :2], dim=1)
            < self.cfg.rewards.stand_still_command_threshold
        )
        linear_acc_world = (
            self.root_states[:, 7:10] - self.last_root_vel[:, :3]
        ) / self.dt
        linear_acc_base = quat_rotate_inverse(self.base_quat, linear_acc_world)
        planar_acc = torch.linalg.vector_norm(linear_acc_base[:, :2], dim=1)
        normalized_acc = planar_acc / self.cfg.rewards.stand_still_acc_scale
        smooth_bounded_penalty = 1.0 - torch.exp(-torch.square(normalized_acc))
        return zero_motion_command * smooth_bounded_penalty

    def _wheelleg_foot_y_offsets(self):
        """Return each wheel's absolute base-frame y distance from the center."""
        wheel_pos = self.rigid_body_states[:, self.feet_indices, :3]
        wheel_pos_relative = wheel_pos - self.root_states[:, None, :3]
        body_quat = self.base_quat[:, None, :].expand(
            -1, wheel_pos_relative.shape[1], -1
        )
        wheel_pos_body = quat_rotate_inverse(
            body_quat.reshape(-1, 4), wheel_pos_relative.reshape(-1, 3)
        ).reshape_as(wheel_pos_relative)
        return wheel_pos_body[..., 1].abs()

    def _reward_wheelleg_track(self):
        """Constrain nominal wheel offsets most strongly in settled standing."""
        foot_y_error = torch.abs(
            self._wheelleg_foot_y_offsets() - self.cfg.rewards.wheel_track_half_width
        )
        normalized_error = foot_y_error / self.cfg.rewards.wheel_track_penalty_softness
        penalty = torch.square(normalized_error).clamp(max=1.0).max(dim=1).values
        return penalty * self._wheelleg_standing_gate()

    def _wheelleg_standing_gate(self):
        """Smoothly relax stance costs, without letting recovery disable them.

        Separate command scales avoid mixing m/s and rad/s in a hard cutoff.
        Actual planar speed is deliberately not a gate: drifting must not
        switch off the stance cost. base_com is a randomization parameter,
        not a measurement of the complete robot's center of mass.
        """
        command_error = (
            self.commands[:, 0] / self.cfg.rewards.wheel_track_gate_lin_command_scale
        ).square() + (
            self.commands[:, 1] / self.cfg.rewards.wheel_track_gate_yaw_command_scale
        ).square()
        upright_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        angular_speed_squared = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        settled_gate = torch.exp(
            -command_error
            - upright_error / self.cfg.rewards.wheel_track_gate_upright_sigma
            - angular_speed_squared / self.cfg.rewards.wheel_track_gate_ang_vel_sigma
        )
        minimum = self.cfg.rewards.wheel_track_gate_min
        return minimum + (1.0 - minimum) * settled_gate

    def _reward_tracking_lin_vel_direction(self):
        """Penalize planar velocity whose direction disagrees with vx command."""
        command_speed = self.commands[:, 0].abs()
        measured_speed = torch.linalg.vector_norm(self.base_lin_vel[:, :2], dim=1)
        valid = (
            (command_speed >= self.cfg.rewards.direction_command_threshold)
            & (
                self.commands[:, 1].abs()
                < self.cfg.rewards.direction_yaw_command_threshold
            )
            & (measured_speed >= self.cfg.rewards.direction_speed_threshold)
        )
        commanded_direction = torch.sign(self.commands[:, 0])
        cosine = (
            commanded_direction * self.base_lin_vel[:, 0]
            / measured_speed.clamp(min=1.0e-6)
        ).clamp(min=-1.0, max=1.0)
        return torch.where(valid, 1.0 - cosine, torch.zeros_like(cosine))

    def deployment_config(self):
        """Record actual policy order and nominal control, not sampled random gains."""
        scales = self.cfg.normalization.obs_scales
        joints = []
        for i, name in enumerate(self.dof_names):
            wheel = i in self.wheel_indices
            joints.append(dict(name=name, mode='velocity' if wheel else 'position',
                default_position=float(self.raw_default_dof_pos[i]),
                action_scale=self.cfg.control.vel_action_scale if wheel else self.cfg.control.pos_action_scale,
                kp=float(self.nominal_d_gains[i] if wheel else self.nominal_p_gains[i]),
                kd=0.0 if wheel else float(self.nominal_d_gains[i]),
                effort_limit=float(self.torque_limits[i]),
                target_limit=self.cfg.control.wheel_no_load_speed_rpm * 2 * math.pi / 60 if wheel else None))
        return dict(simulation_dt=self.sim_params.dt, control_decimation=self.cfg.control.decimation,
            clip_actions=self.cfg.normalization.clip_actions,
            clip_observations=self.cfg.normalization.clip_observations,
            velocity_source='wrapped_position_difference' if self.cfg.env.dof_vel_use_pos_diff else 'simulator',
            joints=joints,
            observations=[dict(source='angular_velocity', scale=scales.ang_vel, size=3),
                dict(source='gravity', scale=1.0, size=3),
                dict(source='commands', scale=[scales.lin_vel,scales.lin_vel,scales.ang_vel,scales.height_measurements], size=4),
                dict(source='joint_position_offset', indices=list(self.leg_indices), scale=scales.dof_pos, size=6),
                dict(source='joint_velocity', scale=scales.dof_vel, size=8),
                dict(source='previous_action', scale=1.0, size=8)],
            initialization=dict(base_position=list(self.cfg.init_state.pos),
                base_quaternion_xyzw=list(self.cfg.init_state.rot),
                joint_velocity='zero', base_velocity='zero', previous_action='zero',
                commands=[0.0,0.0,0.0,float(self.cfg.commands.ranges.height[0])],
                settling_policy_steps=0))

    def set_external_commands(self, command):
        """Accept [vx, vy, yaw_rate, height] without changing the task layout."""
        self.commands[:, 0] = command[0]
        self.commands[:, 1] = command[2]
        self.commands[:, 2] = command[3]
        self.commands[:, 4] = 0.0
        self.compute_observations(update_history=False)

    def evaluation_metrics(self):
        return dict(linear_error=(self.base_lin_vel[:, 0]-self.commands[:, 0]).abs(),
                    yaw_error=(self.base_ang_vel[:, 2]-self.commands[:, 1]).abs(),
                    height_error=(self.base_height-self.commands[:, 2]).abs())

    def evaluation_commands(self, time_s):
        """A fixed four-phase command sequence, identical across methods."""
        phases=((0.0,0.0,0.0),(0.5,0.0,0.0),(0.0,0.0,0.6),(-0.5,0.0,0.0))
        command=list(phases[int(time_s//5)%len(phases)])
        return command+[float(self.cfg.commands.ranges.height[0])]

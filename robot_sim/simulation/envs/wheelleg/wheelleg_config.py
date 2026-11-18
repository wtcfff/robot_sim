# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from robot_sim.simulation.envs.base.legged_robot_config import (
    LeggedRobotCfg,
)


class WheellegCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 3 gyro + 3 gravity + 4 commands + 6 leg positions + 8 speeds + 8 actions.
        num_observations = 32
        num_actions = 8
        # Named critic blocks; truth-only inputs never enter the actor/history.
        num_robot_bodies = 9
        privileged_obs_blocks = (
            ("base_linear_velocity", 3), ("clean_proprioception", 32),
            ("previous_actions", 16), ("joint_acceleration", 8),
            ("terrain_heights", 77), ("applied_torques", 8),
            ("default_position_offsets", 8), ("body_mass_ratios", 9),
            ("body_com_offsets", 27), ("contact_material", 18),
            ("wheel_contacts", 2), ("wheel_contact_forces", 6),
            ("wheel_center_velocities", 6), ("controller_gains", 14),
            ("motor_strength_truth", 8), ("joint_friction", 8),
            ("delays", 3), ("external_wrench", 6),
        )
        num_privileged_obs = sum(size for _, size in privileged_obs_blocks)

    def disable_training_randomization(self):
        """Nominal simulation for play; call before creating the environment."""
        for name in ("randomize_link_mass", "randomize_link_com", "randomize_friction",
                     "randomize_restitution", "randomize_joint_friction", "randomize_Kp",
                     "randomize_Kd", "randomize_motor_torque", "randomize_output_torque_noise",
                     "randomize_action_delay", "randomize_observation_delay", "push_robots",
                     "randomize_initial_state"):
            setattr(self.domain_rand, name, False)
        self.noise.add_noise = False

    class terrain(LeggedRobotCfg.terrain):
        # Preserve the reference task's four enabled terrain types and ratios.
        selected_ids = [0, 1, 2, 5]
        weights = None

    class sim(LeggedRobotCfg.sim):
        # Use the same 10 ms policy period with finer physics integration.
        dt = 0.0025

    class commands:
        # 指令课程参数与取值范围均在 wheelleg 内独立定义。
        curriculum = True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        # Observed commands: [vx, vy, yaw_rate, height]. The reserved vy is
        # always zero; keep the inherited internal yaw/height/heading indices.
        num_commands = 4
        resampling_time = 5.0
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            # Training samples the commanded base height from this range.
            height = [0.360, 0.360]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.360]
        default_joint_angles = {
            "R_hip_Joint": 0.0,
            "R_thigh_Joint": 1.242,
            "R_calf_Joint": -1.975,
            "R_foot_Joint": 0.0,
            "L_hip_Joint": 0.0,
            "L_thigh_Joint": 1.242,
            "L_calf_Joint": -1.975,
            "L_foot_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        decimation = 4
        action_scale = 0.5
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        wheel_no_load_speed_rpm = 210.0
        # Velocity-error proportional gain [Nm s/rad]. The inherited gain
        # initializer stores it in d_gains and applies randomize_Kd unchanged.
        wheel_velocity_kp = 0.5
        # Leg joints use position PD; wheel joints use velocity P control.
        stiffness = {
            "hip_Joint": 40.0,
            "thigh_Joint": 40.0,
            "calf_Joint": 40.0,
            "foot_Joint": 0.0,
        }
        damping = {
            "hip_Joint": 1.0,
            "thigh_Joint": 1.0,
            "calf_Joint": 1.0,
            "foot_Joint": wheel_velocity_kp,
        }

    class asset(LeggedRobotCfg.asset):
        file = "{ROOT_DIR}/resources/robots/wheelleg/urdf/wheelleg_sw.urdf"
        name = "Wheelleg"
        foot_name = "foot_Link"
        penalize_contacts_on = ["base_link", "thigh_Link", "calf_Link"]
        terminate_after_contacts_on = []
        self_collisions = 0
        flip_visual_attachments = False
        collapse_fixed_joints = False
        replace_cylinder_with_capsule = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Physical parameters are sampled once per created environment.
        randomize_link_mass = True
        base_mass_range = [0.9, 1.3]
        link_mass_range = [0.9, 1.1]
        randomize_link_com = True
        base_com_xyz = [0.04, 0.02, 0.02]  # m, symmetric per axis
        link_com_range = [-0.01, 0.01]  # m per axis, excluding base
        randomize_friction = True
        friction_range = [0.5, 1.2]  # wheel contact
        body_friction_range = [0.01, 0.1]
        body_restitution_range = [0.02, 0.2]
        randomize_restitution = True
        restitution_range = [0.02, 0.2]
        randomize_joint_friction = True
        leg_joint_friction_range = [0.0, 0.2]
        wheel_joint_friction_range = [0.0, 0.05]
        # Avoid stacking inherited base mass/inertia/COM randomizations.
        randomize_base_mass = False
        randomize_inertia = False
        randomize_base_com = False
        randomize_default_dof_pos = False

        # Controller gains and motor strength are separate sampled quantities.
        # They, and the delays, are resampled at episode reset.
        randomize_Kp = True
        randomize_Kp_range = [0.75, 1.25]
        randomize_Kd = True
        randomize_Kd_range = [0.75, 1.25]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.1]  # legs
        wheel_motor_strength_range = [0.9, 1.1]
        randomize_output_torque_noise = True
        leg_output_torque_noise_std = 0.1  # Nm, Gaussian per physics step
        wheel_output_torque_noise_std = 0.05  # Nm
        randomize_action_delay = True
        delay_ms_range = [0.0, 15.0]  # integer physics steps, inclusive
        randomize_observation_delay = True
        observation_delay_ms_range = [0.0, 20.0]  # IMU/motor groups

        push_robots = True
        push_interval_range_s = [5.0, 10.0]
        push_duration_s = 0.1
        push_force_range = [-20.0, 20.0]  # body-frame horizontal force, N
        push_torque_range = [-1.0, 1.0]  # body-frame yaw torque, Nm

        randomize_initial_state = True
        initial_xy_range = [-0.5, 0.5]  # m from terrain origin
        initial_z_range = [0.0, 0.03]  # m above nominal base height
        initial_roll_pitch_range = [-0.1, 0.1]  # rad
        initial_yaw_range = [-3.141592653589793, 3.141592653589793]
        initial_linear_velocity_range = [-0.2, 0.2]  # m/s
        initial_angular_velocity_range = [-0.2, 0.2]  # rad/s
        initial_leg_position_offset_range = [-0.1, 0.1]  # rad
        initial_wheel_position_range = [-3.141592653589793, 3.141592653589793]
        initial_joint_velocity_range = [-0.5, 0.5]  # rad/s

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            ang_vel = 0.25  # rad/s, uniform +/- before scaling
            gravity = 0.05
            dof_pos = 0.025  # rad, legs only
            dof_vel = 0.5  # rad/s, legs
            wheel_dof_vel = 1.0  # rad/s
            lin_vel = 0.1  # retained for compatibility; critic truth stays clean
            height_measurements = 0.0  # critic's 77 terrain samples stay clean

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            wheelleg_track = -1.0
            # Available for later experiments; zero scales are not evaluated.
            feet_contact_forces = 0.0
            stumble = 0.0
            termination = 0.0
            tracking_ang_vel_pbrs = 0.0
            tracking_lin_vel_pbrs = 0.0
            tracking_lin_vel_direction = 0.0
            stand_still_acc = 0.0
            base_height = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            dof_vel = -0.0005
            dof_acc = -2.5e-7
            wheel_dof_acc = -1.0e-6
            torques = -0.0001
            wheel_torques = -0.0002
            # Penalize wheel-center motion relative to the base along body x.
            wheel_base_rel_vel = 0.0
            # Softly constrain wheel centers to +/- 3 cm along the base x axis.
            wheel_base_rel_pos = -0.5
            # Learn to avoid abrupt wheel-torque changes across policy steps.
            wheel_torque_rate = 0.0
            # Penalize absolute mechanical power on every joint.
            # The common _reward_power implementation sums |tau * qdot|.
            power = -2.0e-5
            action_rate = -0.01
            wheel_action_rate = -0.02
            action_smooth = -0.01
            # Activate soft velocity/torque limit penalties. These use the
            # per-joint URDF limits and begin below the hard controller limits.
            dof_vel_limits = -0.1
            torque_limits = -0.1
            stand_still = 1.0
            stand_still_drift = -1.0
            collision = -1.0
            dof_pos_limits = -1.0

        only_positive_rewards = False
        clip_single_reward = 1.0
        local_clip_single_reward = 2.0
        tracking_sigma = 0.25
        base_height_sigma = 0.01
        soft_dof_pos_limit = 0.97
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        # Kept for deployment-side compatibility. Training reads the target
        # from commands.ranges.height instead of this value.
        base_height_target = 0.360
        max_contact_force = 250.0
        stand_still_speed_scale = 0.05
        # Direct zero-command fore-aft velocity penalty scale.
        stand_still_drift_speed_scale = 0.01
        # Each wheel is expected to stay at |y| = 0.1335 m in the base frame.
        wheel_track_half_width = 0.1335
        wheel_track_penalty_softness = 0.015
        # Fade the nominal stance cost smoothly during motion and recovery,
        # retaining a floor so leaning or oscillating cannot disable it.
        wheel_track_gate_min = 0.2
        wheel_track_gate_lin_command_scale = 0.3  # m/s
        wheel_track_gate_yaw_command_scale = 0.5  # rad/s
        wheel_track_gate_upright_sigma = 0.03
        wheel_track_gate_ang_vel_sigma = 0.04  # (rad/s)^2
        wheel_contact_force_threshold = 1.0  # N; used by relative wheel costs
        wheel_base_rel_vel_deadband = 0.05
        wheel_base_rel_vel_scale = 0.1
        wheel_base_rel_pos_limit = 0.03
        wheel_base_rel_pos_scale = 0.03
        # Normalize torque differences by 3 Nm per policy step when enabled.
        wheel_torque_rate_scale = 3.0
        # Default-pose wheel-center x in the base frame, derived from the URDF
        # hip offset and the 1.242/-1.975 rad static-support default angles.
        wheel_base_rel_pos_nominal_x = [-0.094, -0.094]
        direction_command_threshold = 0.1
        direction_yaw_command_threshold = 0.1
        direction_speed_threshold = 0.05
        stand_still_command_threshold = 0.1
        stand_still_acc_scale = 1.0


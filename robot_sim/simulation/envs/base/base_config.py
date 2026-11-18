# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import inspect
import copy

class BaseConfig:
    """Materialize nested class-style defaults into independent instance trees."""
    def __init__(self):
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key.startswith("_"):
                continue
            value = getattr(obj, key)
            if inspect.isclass(value):
                instance = value()
                BaseConfig.init_member_classes(instance)
                setattr(obj, key, instance)
            elif not callable(value):
                setattr(obj, key, copy.deepcopy(value))


class BaseEnvCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 0
        num_privileged_obs = None
        obs_history_length = 0  # number of observations stacked together
        obs_history_dec = 1
        num_actions = 0
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 1

    class terrain:
        # 独立声明地形配置，避免继承基任务的地形默认值或共享可变列表。
        mesh_type = "trimesh"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25.0
        curriculum = True
        measure_heights = True
        measured_points_x = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5,
        ]
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5  # 初始等级随机取 0~5；重置时按表现升降级。
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 18  # 九类等比例时，每类两列；行方向为难度等级。
        selected_ids = None  # Stable IDs 0..8; None means all.
        weights = None       # Equal weights unless explicitly configured.
        slope_treshold = 0.75

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


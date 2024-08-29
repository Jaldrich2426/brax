# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Trains an ant to run in the +x direction."""

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco


class Franka(PipelineEnv):


  def __init__(
      self,
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs,
  ):
    path = epath.resource_path('brax') / 'envs/franka_emika_panda/panda_nohand.xml'
    my_model = mujoco.MjModel.from_xml_path(str(path))
    sys = mjcf.load_model(my_model)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.005})
      n_frames = 10

    if backend == 'mjx':
      sys = sys.tree_replace({
          'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
          'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
          'opt.iterations': 1,
          'opt.ls_iterations': 4,
      })

    if backend == 'positional':
      # TODO: does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jp.zeros(3)
    metrics = {
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'ee_pose': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt

    # min_z, max_z = self._healthy_z_range
    # is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
    # is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
    # if self._terminate_when_unhealthy:
    #   healthy_reward = self._healthy_reward
    # else:
    #   healthy_reward = self._healthy_reward * is_healthy
    # ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    # contact_cost = 0.0

    obs = self._get_obs(pipeline_state)
    # reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    # done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    # done = jp.linalg.norm(pipeline_state.q - self.joint_target_state) < self.target_threshold
    state.metrics.update(
        # reward_forward=forward_reward,
        # reward_survive=healthy_reward,
        # reward_ctrl=-ctrl_cost,
        # reward_contact=-contact_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        # forward_reward=forward_reward,
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=0.0, done=0.0
    )

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe ant body position and velocities."""
    qpos = pipeline_state.q
    qvel = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      qpos = pipeline_state.q[2:]

    return jp.concatenate([qpos] + [qvel])
from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from brax.envs import walker2d

"""
  ### Action Space

  | Num | Action                                 | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|----------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                    | hinge | torque (N m) |
  | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                      | hinge | torque (N m) |
  | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                     | hinge | torque (N m) |
  | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint               | hinge | torque (N m) |
  | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                 | hinge | torque (N m) |
  | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                | hinge | torque (N m) |

  ### Observation Space

  | Num | Observation                                      | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|--------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | ?                                                | -Inf | Inf | rootz (torso)                  | slide | position (m)             |
  | 1   | ?                                                | -Inf | Inf | rootz (torso)                  | slide | position (m)             |
  | 2   | angle of the top                                 | -Inf | Inf | rooty (torso)                  | hinge | angle (rad)              |
  | 3   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                    | hinge | angle (rad)              |
  | 4   | angle of the leg joint                           | -Inf | Inf | leg_joint                      | hinge | angle (rad)              |
  | 5   | angle of the foot joint                          | -Inf | Inf | foot_joint                     | hinge | angle (rad)              |
  | 6   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint               | hinge | angle (rad)              |
  | 7   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                 | hinge | angle (rad)              |
  | 8   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                | hinge | angle (rad)              |
  | 9   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
  | 10  | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
  | 11  | angular velocity of the angle of the top         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
  | 12  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                    | hinge | angular velocity (rad/s) |
  | 13  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                      | hinge | angular velocity (rad/s) |
  | 14  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                     | hinge | angular velocity (rad/s) |
  | 15  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint               | hinge | angular velocity (rad/s) |
  | 16  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                 | hinge | angular velocity (rad/s) |
  | 17  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                | hinge | angular velocity (rad/s) |
    18    x-coordinate of the top (distance of hopper) 
    19    x coord of ?
    20    x coord of ?
    21    x coord of ?
    22    x coord of ?
    23    x coord of ?
    24    x coord of ?
    25    z-coordinate of the top (height of hopper)
    26    z coord of ?
    27    z coord of ?
    28    z coord of ?
    29    z coord of ?
    30    z coord of ?
    31    z coord of ?
"""
class Walker2DMPC(walker2d.Walker2d):

    def __init__(
        self,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        backend='generalized',
        **kwargs
        ):
        super().__init__(forward_reward_weight=forward_reward_weight,ctrl_cost_weight=ctrl_cost_weight,healthy_reward=healthy_reward,terminate_when_unhealthy=terminate_when_unhealthy,healthy_z_range=healthy_z_range,healthy_angle_range=healthy_angle_range,reset_noise_scale=reset_noise_scale,exclude_current_positions_from_observation=exclude_current_positions_from_observation,backend=backend,**kwargs)

    def set_state(self,input_state):
        # print("set state input state is:",input_state,type(input_state))
        pipeline_state = self.pipeline_init(input_state.pipeline_state.q, input_state.pipeline_state.qd)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_ctrl': zero,
            'reward_healthy': zero,
            'x_position': zero,
            'x_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)
    
    def generate_state_from_tensor(self, q,qd):
        # print(state)
        # print(type(state))
        # print(state.shape)
        # # print(state)
        # # state=jp.array(state)
        # # print(state)
        # # print(state.shape)
        # print(state[0])
        # print(type(state[0]))
        # q = state[0:9]
        # qd = state[9:18]
        # print("conversion")
        # print(tensor.shape)
        # print(q.shape)
        # print(qd.shape)
        # print("sliced states:")
        # print(q)
        # print(qd)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_ctrl': zero,
            'reward_healthy': zero,
            'x_position': zero,
            'x_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def set_state_and_step(self,input_state,action):
        state = self.set_state(input_state)
        return self.step(state,action)
    
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        # position = position.at[0].set(pipeline_state.x.pos[0, 0]) # set to x pos of body
        # position = position.at[1].set(pipeline_state.x.pos[0, 2]) # set to z pos of body
        velocity = jp.clip(pipeline_state.qd, -10, 10)
        x_body = pipeline_state.x.pos[:, 0]
        z_body = pipeline_state.x.pos[:, 2]
        # print("x_body",x_body.shape)
        # print("z_body",z_body.shape)

        # if self._exclude_current_positions_from_observation:
        #     position = position[1:]

        return jp.concatenate((position, velocity,x_body,z_body))

    def get_obs_grad(self, pipeline_grad,action_grad) -> jax.Array:
        """Returns gradient of the environment observations.
        
        state Gradient: 32x32
            q: 9x9
            qd: 9x9
            x: 7x1
            z: 7x1
        action Gradient: 32x6

        State Gradient: [dq/dq,  dq/dqd,  dq/dx,  dq/dz;
                         dqd/dq, dqd/dqd, dqd/dx, dqd/dz;
                         dx/dq,  dx/dqd,  dx/dx,  dx/dz;
                         dz/dq,  dz/dqd,  dz/dx,  dz/dz]

                         [9x9   9x9     9x7    9x7;
                          9x9   9x9     9x7    9x7;
                          7x9   7x9     7x7    7x7;
                          7x9   7x9     7x7    7x7]

        Action Gradient: [dq/du, dqd/du, dx/du, dz/du]

        """

        # State Portion

        # print(pipeline_grad.pipeline_state.q.pipeline_state)
        # print(pipeline_grad.pipeline_state.qd)
        # print(pipeline_grad.pipeline_state.x.pos.pipeline_state.x.pos)
        # print(pipeline_grad.pipeline_state.x.pos.pipeline_state.x.pos.shape)
        q_handle=pipeline_grad.pipeline_state.q.pipeline_state
        qd_handle=pipeline_grad.pipeline_state.qd.pipeline_state
        x_handle=pipeline_grad.pipeline_state.x.pos.pipeline_state
        z_handle=pipeline_grad.pipeline_state.x.pos.pipeline_state

        dq_dq = q_handle.q
        dq_dqd = q_handle.qd
        dq_dx = q_handle.x.pos[:,:,0]
        dq_dz = q_handle.x.pos[:,:,2]

        # print("grad shapes")
        # print(dq_dq.shape)
        # print(dq_dqd.shape)
        # print(dq_dx.shape)
        # print(dq_dz.shape)

        # print("helper",q_handle.x.pos.shape)
        dqd_dq = qd_handle.q
        dqd_dqd = qd_handle.qd
        dqd_dx = qd_handle.x.pos[:,:,0]
        dqd_dz = qd_handle.x.pos[:,:,2]

        # print(dqd_dq.shape)
        # print(dqd_dqd.shape)
        # print(dqd_dx.shape)
        # print(dqd_dz.shape)

        dx_dq = x_handle.q[:,0,:]
        dx_dqd = x_handle.qd[:,0,:]
        dx_dx = x_handle.x.pos[:,0,:,0]
        dx_dz = x_handle.x.pos[:,0,:,2]

        # print(dx_dq.shape)
        # print(dx_dqd.shape)
        # print(dx_dx.shape)
        # print(dx_dz.shape)

        # print("helper",x_handle.x.pos.shape)
        

        dz_dq = z_handle.q[:,2,:]
        dz_dqd = z_handle.qd[:,2,:]
        dz_dx = z_handle.x.pos[:,2,:,0]
        dz_dz = z_handle.x.pos[:,2,:,2]

        # print(dz_dq.shape)
        # print(dz_dqd.shape)
        # print(dz_dx.shape)
        # print(dz_dz.shape)

        full_state_grad = jp.concatenate((jp.concatenate((dq_dq,dq_dqd,dq_dx,dq_dz),axis=-1),
                                         jp.concatenate((dqd_dq,dqd_dqd,dqd_dx,dqd_dz),axis=-1),
                                         jp.concatenate((dx_dq,dx_dqd,dx_dx,dx_dz),axis=-1),
                                         jp.concatenate((dz_dq,dz_dqd,dz_dx,dz_dz),axis=-1),),axis=0)
        # print(full_state_grad.shape)

        # position_grad = jp.concatenate((dq_dq,dq_dqd),axis=-1)
        # velocity_grad = jp.concatenate((dqd_dq,dqd_dqd),axis=-1)
        
        # z_term = jp.concatenate((pipeline_grad.pipeline_state.x.pos.pipeline_state.q[0,2,:],pipeline_grad.pipeline_state.x.pos.pipeline_state.qd[0,2,:]),axis=0)[jp.newaxis,...]
        # x_term = jp.concatenate((pipeline_grad.pipeline_state.x.pos.pipeline_state.q[0,0,:],pipeline_grad.pipeline_state.x.pos.pipeline_state.qd[0,0,:]),axis=0)[jp.newaxis,...]
        # print(z_term.shape)
        # dq_dxpos = pipeline_grad.pipeline_state.q.pipeline_state.x.pos
        # dqd_dxpos = pipeline_grad.pipeline_state.qd.pipeline_state.x.pos
        # dxpos_dq = pipeline_grad.pipeline_state.x.pos.pipeline_state.q
        # dxpos_dqd = pipeline_grad.pipeline_state.x.pos.pipeline_state.qd
        # dxpos_dxpos = pipeline_grad.pipeline_state.x.pos.pipeline_state.x.pos
        # print(dq_dxpos.shape)
        # print(dqd_dxpos.shape)
        # print(dxpos_dq.shape)
        # print(dxpos_dqd.shape)
        # print(dxpos_dxpos.shape)
        # print(pipeline_grad.pipeline_state.x.pos.pipeline_state.q)
        
        # position_grad=position_grad.at[1].set(z_term)
        # position_grad=position_grad.at[0].set(x_term)


        # if self._exclude_current_positions_from_observation:
        #     position_grad = position_grad[1:]
        
        # Action portion
        action_term = jp.concatenate((action_grad.pipeline_state.q,action_grad.pipeline_state.qd),axis=0)
        # print(action_grad.pipeline_state.x.pos.shape)
        # print(action_grad.pipeline_state.x.pos)
        action_z_term = action_grad.pipeline_state.x.pos[:,2]
        # print(action_z_term.shape)
        # print(action_term.shape)
        # action_term = action_term.at[1].set(action_z_term)

        action_x_term = action_grad.pipeline_state.x.pos[:,0]
        # action_term = action_term.at[0].set(action_x_term)

        # if self._exclude_current_positions_from_observation:
        #     action_term = action_term[1:]
        # print("state_grad",jp.concatenate((position_grad, velocity_grad,x_term,z_term), axis=0).shape)
        # print("action_grad",jp.concatenate((action_term,action_x_term,action_z_term), axis=0).shape)
        return full_state_grad, jp.concatenate((action_term,action_x_term,action_z_term), axis=0)
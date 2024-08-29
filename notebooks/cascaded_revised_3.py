import functools
import jax
import os

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

from IPython.display import HTML, clear_output,display, IFrame

import brax


import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sacaction_grad
from torch2jax import j2t, t2j
from mrf_swarm.sim.robot import DynamicsModel
import torch
from functools import singledispatch, update_wrapper
from brax.envs.base import PipelineEnv, State
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# jax.config.update("jax_traceback_filtering", "off")
# class MultiDispatch:
#     def __init__(self, default):
#         self.default = default
#         self.dispatch = {}
#         update_wrapper(self, default)

#     def register(self, *types):
#         def wrapper(func):
#             self.dispatch[types] = func
#             return func
#         return wrapper

#     def __call__(self, *args):
#         types = tuple(type(arg) for arg in args)
#         func = self.dispatch.get(types, self.default)
#         return func(*args)

class BraxHandler():

    def __init__(self, env_name, backend,rng_seed=0):
        self.env_name = env_name
        self.backend = backend
        # self.env = envs.create(env_name=env_name, backend=backend)
        # rng = jax.random.PRNGKey(seed=rng_seed)

        # self.jit_env_set_and_step = jax.jit(self.env.set_state_and_step)
        # self.jit_env_step = jax.jit(self.env.step)
        # self.jit_env_reset = jax.jit(self.env.reset)

        # self.jit_env_rng = jax.random.PRNGKey(seed=1)
        # self.init_state = self.jit_env_reset(rng=rng)
        self.rollout=[]
        # self.rollout.append(self.init_state.pipeline_state)

        # self.action_grad1 = jax.jit(jax.jacobian(self.step_obs,argnums=1))
        # self.state_grad1 = jax.jit(jax.jacobian(self.step_obs, argnums=0))
        # self.action_grad = jax.jit(jax.jacobian(self.full_grad,argnums=1))
        # self.state_grad = jax.jit(jax.jacobian(self.full_grad, argnums=0))
        # self.jit_generate_state_from_tensor = jax.jit(self.env.generate_state_from_tensor)

    # def step_obs(self,state, act):
    #     return self.perform_step(state, act).obs
    
    # def full_grad(self,state, act):
    #     grad = self.perform_step(state, act)
    #     # print(grad.pipeline_state)
    #     return grad

    # def perform_step(self,input_state, act, render=False):
    #     # Step environment
    #     state = self.jit_env_step(input_state, act)
    #     if render:
    #         self.rollout.append(state.pipeline_state)
    #     return state
    
    # def perform_step_w_grad(self,input_state, act, render=False):
    #     # Step environment
    #     next_state = self.jit_env_step(input_state, act)

    #     # Compute gradients
    #     pipeline_state_grad = self.state_grad(input_state, act)
    #     action_grad = self.action_grad(input_state, act)
       
    #     # Convert gradients to useful data 

    #     obs_grad,action_grad = self.env.get_obs_grad(pipeline_state_grad,action_grad)
        
        
    #     # Maintain History if specified
    #     if render:
    #         self.add_to_rollout(next_state)

    #     return next_state, obs_grad, action_grad

    def get_rollout(self):
        return self.rollout
    
    def add_to_rollout(self, state):
        self.rollout.append(state.pipeline_state)
    
    def save_rollout(self, filename):
        with open(filename, 'w') as f:
            f.write(html.render(self.env.sys.tree_replace({'opt.timestep': self.env.dt}), self.rollout))

class BraxTorchStateMapper():
    def __init__(self, state,action, state_grad, act_grad):
        self.brax_state = state
        self.brax_action = action

        # Convert fields to torch tensors
        self.x = j2t(self.brax_state.obs)
        self.u = j2t(self.brax_action)
        self.act_grad=j2t(act_grad)
        self.state_grad=j2t(state_grad)

class BraxHelperModel(DynamicsModel):
    def __init__(self, brax_handler,env=None,horizon=1,render=False,rng_seed=0):
        self.brax_handler=brax_handler
        self.horizon=horizon
        self.render=render
        # self.generate_state_from_tensor = self.brax_handler.jit_generate_state_from_tensor
        if(env is None):
            self.env = envs.create(env_name=env_name, backend=backend)
        else:
            self.env=env
        rng = jax.random.PRNGKey(seed=rng_seed)

        self.jit_env_set_and_step = jax.jit(self.env.set_state_and_step)
        self.jit_env_step = jax.jit(self.env.step)
        self.jit_env_reset = jax.jit(self.env.reset)

        self.jit_env_rng = jax.random.PRNGKey(seed=rng_seed)
        self.init_state = self.jit_env_reset(rng=rng)
        self.generate_state_from_tensor = jax.jit(self.env.generate_state_from_tensor)

        brax_handler.rollout.append(self.init_state.pipeline_state)

        self.action_grad1 = jax.jit(jax.jacobian(self.step_obs,argnums=1))
        self.state_grad1 = jax.jit(jax.jacobian(self.step_obs, argnums=0))
        self.action_grad = jax.jit(jax.jacobian(self.full_grad,argnums=1))
        self.state_grad = jax.jit(jax.jacobian(self.full_grad, argnums=0))
        # self.rollout_history=[]
        # self.rollout_history.append(self.init_state.pipeline_state)

    def step_obs(self,state, act):
        return self.perform_step(state, act).obs
    
    def full_grad(self,state, act):
        grad = self.perform_step(state, act)
        # print(grad.pipeline_state)
        return grad

    # # TODO: scrap torch rollouts and instead maintain a particle set of all brax states
    # def rollout_torch(self, torch_state, torch_action):
    #     # print(torch_state)
    #     # print(torch_action)
    #     input_state = self.generate_state_from_tensor(t2j(torch_state[:9].detach().clone()),t2j(torch_state[9:18].detach().clone()))
    #     # input_state.info['steps'] = jp.zeros(1)
    #     # input_state.info['truncation'] = jp.zeros(1)
    #     self.brax_handler.env.reset(input_state)
    #     return self.rollout(input_state, t2j(torch_action))
    def get_env(self):
        return self.env

    def set_env(self, env):
        self.env = env

    def perform_step(self,input_state, act, render=False):
        # Step environment
        state = self.jit_env_step(input_state, act)
        if render:
            self.brax_handler.add_to_rollout(state.pipeline_state)
        return state
    
    def perform_step_w_grad(self,input_state, act, render=False):
        # Step environment
        next_state = self.jit_env_step(input_state, act)

        # Compute gradients
        pipeline_state_grad = self.state_grad(input_state, act)
        action_grad = self.action_grad(input_state, act)
       
        # Convert gradients to useful data 

        obs_grad,action_grad = self.env.get_obs_grad(pipeline_state_grad,action_grad)
        
        
        # Maintain History if specified
        if render:
            self.brax_handler.add_to_rollout(next_state)

        return next_state, obs_grad, action_grad

    def rollout(self, 
                input_state: State, 
                action: jp.ndarray):
        states=[]
        torch_states=[]
        state=input_state
        for i in range(self.horizon):
            if(action.ndim==1):
                next_state = self.perform_step(state,action,render=self.render)
            else:
                next_state = self.perform_step(state,action[i],render=self.render)

            torched_state = j2t(next_state.obs)
            torched_action = j2t(action[i])
            
            if(action.ndim==1):
                torched_action=torched_action[None,...]
          
            combined_state_action = torch.cat((torched_state,torched_action),dim=0)

            # memory of states
            torch_states.append(combined_state_action)
            states.append(next_state)
            
            state=next_state

        return torch_states,states

    def rollout_w_grad(self, input_state, action):
        states=[]
        obs_grads=[]
        action_grads=[]
        state=input_state
        torch_states=[]
        # for key in input_state.info:
        #     print(key, input_state.info[key])
        if(action.ndim==1):
            action=action[jp.newaxis,...]
        for i in range(self.horizon):
            next_state,obs_grad, action_grad = self.perform_step_w_grad(state,action[i],render=self.render)
            states.append(next_state)
            torched_state = j2t(next_state.obs)
            torched_action = j2t(action[i])
            # print(torched_action.shape)
            combined_state_action = torch.cat((torched_state,torched_action),dim=0)
            torch_states.append(combined_state_action)

            torched_obs_grad = j2t(obs_grad)
            torched_action_grad = j2t(action_grad)
            obs_grads.append(torched_obs_grad)
            action_grads.append(torched_action_grad)
            state=next_state
        return action_grads,obs_grads, torch_states,states
    
    def is_linear(self):
        return False

class BraxModel(DynamicsModel):
    def __init__(self, env_name, backend,rng_seed=0,horizon=1):
        self.brax_handler=BraxHandler(env_name, backend,rng_seed)
        self.horizon=horizon
        # self.state_list=[]

        # Model Versions
        # print(f"brax model horizon: {self.horizon}")
        self.render_model=BraxHelperModel(self.brax_handler,horizon=1,render=True)

        self.one_step_model=BraxHelperModel(self.brax_handler,horizon=1,env=self.render_model.get_env())
        self.horizon_steps_model=BraxHelperModel(self.brax_handler,horizon=self.horizon,env=self.render_model.get_env())
        # print("horizon steps model horizon: ",self.horizon_steps_model.horizon)

    def copy_envs(self):
        env = self.render_model.get_env()

        self.one_step_model.set_env(env)
        self.horizon_steps_model.set_env(env)

    def is_linear(self):
        return False



import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mrf_swarm.envs import PointSwarm
from mrf_swarm.controllers.mpc import SteinMPC
from mrf_swarm.sim.robot import LinearPointRobotModel
from mrf_swarm.sim.map import DiffMap
from plotting import draw_belief_traj, draw_paritcles
from utils.point_swarm import make_costs, make_terminal_costs
from mrf_swarm.costs.base_costs import CompositeSumCost

from cascaded_cost_factors import DynamicsPairwiseFactor
from cascaded_cost_factors import CascadedLoopySVBP
from tqdm import tqdm
from outputs_to_gif import plot_as_gif
from utils.walker_cost import make_walking_costs,make_terminal_walking_costs
from cascaded_cost_factors import CascadedMPC, CascadedMPCBrax
import torch_bp.bp as bp
import torch_bp.distributions as dist

from torch_bp.graph import factors, Graph
from torch_bp.util.plotting import plot_dists, plot_graph, plot_particles
from torch_bp.inference.kernels import RBFMedianKernel
from mrf_swarm.factors.trajectory_factors import UnaryRobotTrajectoryFactor
from torch2jax import j2t, t2j
import faulthandler; faulthandler.enable()


# mega outside loop

for trial in range(1):
    # print("Trial: ",trial)
    FIG_WIDTH = 6
    DT = 0.1
    SIM_TIME = 20
    HORIZON = 3
    NUM_PARTICLES = 2
    tensor_kwargs = {"device": 'cuda', "dtype": torch.float}

    torch.random.manual_seed(0)

    #Initialize state along with brax

    env_name = 'walker2d_mpc'
    backend = 'generalized'
    # brax_handler = BraxHandler(env_name, backend)
    brax_model=BraxModel(env_name, backend,horizon=HORIZON)
    # jit_env_step = jax.jit(brax_model..env.step)
    brax_state = brax_model.render_model.init_state
    torch_state= j2t(brax_state.obs)
    steps = 10
    # print(torch_state)

    # out_dir = "/home/jacealdr/repos/CascadedCostBrax/repos/images"

    c_pos_x=3.
    c_q=1.
    c_qd=1.
    c_q_term=1.
    c_qd_term=1.
    c_vel_x=3.
    c_u=0.05
    c_term_x=3.
    c_pos_z=10.
    c_vel_z=0.
    c_term_z=10.
    c_torso_angle=0.
    x_goal=0
    z_goal=1.2
    c_term_torso_angle=0.


    cascading_costs = make_walking_costs(c_pos_x=c_pos_x,c_q=c_q,c_qd=c_qd, c_q_term=c_q_term,c_qd_term=c_qd_term,c_vel_x=c_vel_x, c_u=c_u, c_term_x=c_term_x, c_pos_z=c_pos_z, c_vel_z=c_vel_z,c_term_z=c_term_z, goal_x=x_goal,goal_z=z_goal, use_terminal_cost=False,tensor_kwargs=tensor_kwargs)
    final_costs = make_terminal_walking_costs( c_q_term=c_q_term,c_qd_term=c_qd_term,c_term_x=c_term_x,c_term_z=c_term_z,goal_x=x_goal,goal_z=z_goal, tensor_kwargs=tensor_kwargs)
    full_costs = make_walking_costs(c_pos_x=c_pos_x, c_vel_x=c_vel_x,c_q=c_q,c_qd=c_qd, c_q_term=c_q_term,c_qd_term=c_qd_term,c_u=c_u, c_term_x=c_term_x, c_pos_z=c_pos_z, c_vel_z=c_vel_z,c_term_z=c_term_z, goal_x=x_goal,goal_z=z_goal, use_terminal_cost=True,tensor_kwargs=tensor_kwargs)
    gamma = 1. / np.sqrt(2*2)
    rbf_kernel = RBFMedianKernel(gamma=gamma)

    # TODO fix passing brax/notebooks/cascaded.py
    mpc=CascadedMPCBrax(cascading_costs,final_costs, brax_model.one_step_model, brax_model.horizon_steps_model,rbf_kernel,
                    brax_state,num_particles=NUM_PARTICLES,horizon=HORIZON,dim=6,
                    init_cov=0.5, optim_params={"lr": 0.05}, full_costs=full_costs, one_cost=cascading_costs, term_cost=final_costs, tensor_kwargs=tensor_kwargs)
    total_cost=0

    POS_TOL = 0.5
    VEL_TOL = 0.2

    # for i in tqdm(range(HORIZON)):
    #     for _ in range(1):
    #         mpc.solve(num_iter=1, normalize=True, precompute=False)
    #         # Do plotting?
    #     mpc.shift(torch_state)
    #     print("finished intialization step:",i)
    # particle_seed = mpc.get_particles()
    # mpc.sbp.reset(particle_seed)
    combined_cost_eval = CompositeSumCost(costs=cascading_costs, sigma=1,
                                                tensor_kwargs=tensor_kwargs)
    # print("running mpc")
    pbar = tqdm(range(steps))
    brax_state=brax_model.brax_handler.rollout[-1]
    for i in pbar:
        
        # print(starting_pipeline_state)
        # starting_state = j2t(brax_model.brax_handler.env._get_obs(starting_pipeline_state))
        # print(starting_state)
        mpc.solve(num_iter=1, normalize=True, precompute=False)
        # mpc.shift(state)
        # print("about to get best state action")
        state_action=mpc.get_best_state_action()
        state=state_action[0,0:32]
        action=state_action[0,32:]
        action=torch.zeros_like(action)
        pbar.set_description(f"Trial: {trial} | X: {state_action[0,18].item()} | Z {state_action[0,25].item()}")
        # print("pre rollout")
        # print("x,z: ",state_action[0,18],state_action[0,25])
        # print("state action: ",state_action[0,:])
        # print("nan test:",torch.isnan(state_action[0,:]).any())
        if(torch.isnan(state_action[0,:]).any()):
            break
        next_state_action,next_brax_state=brax_model.render_model.rollout(brax_state,t2j(action))
        
        # next_brax_state = jit_env_step(brax_state, t2j(action))
        # state = j2t(next_brax_state.obs)
        # state_action = torch.cat((state,action),dim=0)
        brax_model.brax_handler.save_rollout(f"Full_MPC_test_longer_steps_working.html")
        brax_model.copy_envs()
        mpc.action_history.append(action)
        mpc.state_history.append(state)
        # normalized_cost = combined_cost_eval(state_action[0])
        # total_cost+=normalized_cost
        # TODO Do save to rollout
        # print(next_state_action)
        mpc.shift(state)
        brax_state=next_brax_state
    brax_model.brax_handler.save_rollout(f"Full_MPC_test_longer_steps_final_{trial}.html")
        # # if at terminal position and velocity is low, end early
        # if torch.norm(state[0:2] - x_goal) < POS_TOL and torch.norm(state[2:4]) < VEL_TOL:
        #     print("Reached goal at iteration ", i)
        #     break

    # running_cost=combined_cost_evalrunning_cost
    # print("running initialization")


    # for i in range(steps):
    #     # Pick an action
    #     act = jp.ones(brax_handler.env.action_size, dtype=float)

    #     # Perform a step with gradients
    #     new_state,obs_grad,action_grad = brax_handler.perform_step_w_grad(state, act,render=True)
    
    #     # Assign new state (if continually running)
    #     state = new_state
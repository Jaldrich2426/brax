import functools
import jax
from jax import numpy as jp

from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo


env_name = 'walker2d'
backend = 'generalized'  # @param ['generalized', 'positional', 'spring', 'mjx']

env = envs.get_environment(env_name=env_name,
                           backend=backend)

state = env.reset(rng=jax.random.PRNGKey(seed=0))

# train_fn = functools.partial(ppo.train,  num_timesteps=50_000_000, num_evals=10, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1,
#                              unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1)

# make_inference_fn, params, _ = train_fn(environment=env)
# inference_fn = make_inference_fn(params)

env = envs.create(env_name=env_name, backend=backend)

def trimmed_state_step(state, action):
    return env.step(state, action).obs

rng = jax.random.PRNGKey(seed=1)
state = env.reset(rng=rng)

# act_rng, rng = jax.random.split(rng)
# act, _ = inference_fn(state.obs, act_rng)
act = jp.zeros((env.action_size,))
# act = jp.ones((env.action_size,))*.5

new_q = trimmed_state_step(state, act)
print(new_q)
print(jax.jacobian(trimmed_state_step,argnums=1)(state, act))
print(jax.jacobian(trimmed_state_step, argnums=0)(state, act).obs)
# print(jp.linalg.norm(jax.jacobian(trimmed_state_step, argnums=0)(state, act).obs))
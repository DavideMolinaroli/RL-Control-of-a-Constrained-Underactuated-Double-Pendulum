import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac.policies import MlpPolicy

from model.model_parameters import model_parameters
from model.symbolic_plant import SymbolicDoublePendulum
from simulation.gym_env import DoublePendulumEnv, double_pendulum_dynamics_func
from simulation.simulation import Simulator
from utils.reset_functions import noisy_reset_func, zero_reset_func

log_dir = f"log_data/SAC_training"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

robot = "pendubot"

print(f"Training {robot} using SAC")
# Set random seed to zero for reproducibility
set_random_seed(0)

max_velocity = 20.0
max_torque = 10.0
torque_limit = [max_torque, 0.0]
model_par_path = f"parameters/{robot}_parameters.yml"

# Model parameters
mpar = model_parameters(filepath=model_par_path)

print("Loading model parameters...")
dt = 0.01
max_steps = 1_500
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 2
obs_space = gym.spaces.Box(
    np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
)
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))

# tuning parameter
n_envs = 25
training_steps = 2_500_000
verbose = 1
eval_freq = 5000
n_eval_episodes = 1
learning_rate = 0.005 

print(
    f"Training settings: training step = {training_steps}, learning rate = {learning_rate}"
)

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

# initialize vectorized environment
env = DoublePendulumEnv(
    dynamics_func=dynamics_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training env
envs = make_vec_env(
    seed=0,
    env_id=DoublePendulumEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reset_func": noisy_reset_func,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
    },
)

# evaluation env, same as training env
eval_env = DoublePendulumEnv(
    dynamics_func=dynamics_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="model_default",
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

# train
agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, f"logs_{robot}"),
    learning_rate=learning_rate,
)

agent.learn(total_timesteps=training_steps, callback=eval_callback, progress_bar=True)
save_agent = "model_default/model_end_phase1"
agent.save(save_agent, include=["replay_buffer"])

eval_freq = 1500

envs_phase2 = make_vec_env(
    seed=0,
    env_id=DoublePendulumEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reset_func": zero_reset_func,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
        "disturbance": True,
        "terminates": False,
        "unscaled_constraints":[4*np.pi, 4*np.pi, 20, 20],
        "theta_slack": np.pi/6,
        "penalty_k" : 1
    },
)

eval_env_phase2 = DoublePendulumEnv(
    dynamics_func=dynamics_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
    disturbance=True,
    terminates=False,
    unscaled_constraints = [4*np.pi, 4*np.pi, 20, 20],
    theta_slack=np.pi/6,
    penalty_k=1
)

eval_callback_phase2 = EvalCallback(
    eval_env_phase2,
    best_model_save_path="model_penalty_pi6",
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

# Continue training same agent
agent.set_env(envs_phase2)
agent.learn(total_timesteps=training_steps, callback=eval_callback_phase2, progress_bar=True)
save_agent = "model_penalty_pi6/model_end_phase2"
agent.save(save_agent)
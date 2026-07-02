import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from env.quadrotor_env import QuadrotorVecEnv
from controllers.controllers import *

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = "px4-30dr-rand-dt"

    env = QuadrotorVecEnv(
        num_envs=100, 
        controller="px4", 
        device=device)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
        log_std_init=0,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        n_steps=1000,
        batch_size=5000,
        n_epochs=10,
        gamma=0.999,
        tensorboard_log="outputs/tb/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=7812,
        save_path="outputs",
        name_prefix=name,

    )

    model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)
    model.save(name)
    env.close()

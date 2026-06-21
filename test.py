import time
from stable_baselines3 import PPO
from env.quadrotor_env import QuadrotorEnv

model = PPO.load("outputs/UZH_best")

env = QuadrotorEnv(render_mode="human")
obs, _ = env.reset()

episode = 0
ep_reward = 0
ep_steps = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_reward += reward
    ep_steps  += 1

    if terminated or truncated:
        episode += 1
        print(f"Episode {episode:3d} | steps: {ep_steps:4d} | reward: {ep_reward:.1f}")
        ep_reward = 0
        ep_steps  = 0
        obs, _ = env.reset()
        time.sleep(0.3)  

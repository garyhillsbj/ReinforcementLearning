import gym
import os
from stable_baselines3 import ppo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

envname='CartPole-v0'
env=gym.make(envname)
episodes=5
for episode in range(episodes+1):
    state=env.reset()
    done=False
    score=0
    while not done:
        env.render()
        action=env.action_space.sample()
        n_state, reward, done, tr,info=env.step(action)
        score+=reward
    print(score)
env.close()
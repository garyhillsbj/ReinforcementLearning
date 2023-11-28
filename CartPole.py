# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 02:45:12 2023

@author: Administrator
"""

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class CP_RL:
    def __init__(self, log_path, models_path,train_or_test):
        self.models_path=models_path
        if train_or_test:
            self.env=gym.make('CartPole-v0')
        else:
            self.env=gym.make('CartPole-v0', render_mode='human')
        self.env=DummyVecEnv([lambda:self.env])
        self.model=self.building_model()
    def building_model(self):
        model=PPO('MlpPolicy',self.env,verbose=1,tensorboard_log=log_path)
        return model
    def load_model(self):
        self.model=PPO.load(self.models_path)
    def save_model(self):
        self.model.save(self.models_path)
        del self.model
    def learning_model(self, totalsteps):
        self.model.learn(total_timesteps=totalsteps)
    def testing_model(self,episodes):
        for episode in range(episodes+1):
            obs=self.env.reset()
            score=0
            done=False
            while not done:
                act,_=self.model.predict(obs)
                obs, rew, done, inf= self.env.step(act)
                score+=rew
    def exit_env(self):
        self.env.close()        

if __name__ == "__main__":
    log_path=os.path.join('CPRL_training','logs')
    models_path=os.path.join('CPRL_training','save_models','CPRL_models')
    CPRL=CP_RL(log_path, models_path,0)
    # CPRL.learning_model(50000)
    # CPRL.save_model()
    CPRL.load_model()
    CPRL.testing_model(5)
    CPRL.exit_env()
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 04:18:00 2023

@author: Administrator
"""

import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

class ATARI_RL:
    def __init__(self, log_path, models_path,train_or_test):
        self.models_path=models_path
        if train_or_test:
            self.env=gym.make('Breakout-v0')
        else:
            self.env=gym.make('Breakout-v0', render_mode='human')
        self.env=make_atari_env('Breakout-v0',n_envs=4,seed=0)
        self.env=VecFrameStack(self.env,n_stack=4)
        self.model=self.building_model()
    def building_model(self):
        model=A2C('CnnPolicy',self.env,verbose=1,tensorboard_log=log_path)
        return model
    def load_model(self):
        self.model=A2C.load(self.models_path,self.env)
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
                #self.env.render_mode('human')
                #self.env.render()
                act,_=self.model.predict(obs)
                #act=self.env.action_space.sample()
                obs, rew, done, inf= self.env.step(act)
                score+=rew
    def exit_env(self):
        self.env.close()        

if __name__ == "__main__":
    log_path=os.path.join('ATARIRL_training','logs')
    models_path=os.path.join('ATARIRL_training','save_models','ATARIRL_models')
    ATARIRL=ATARI_RL(log_path, models_path,0)
    # ATARIRL.learning_model(80000)
    # ATARIRL.save_model()
    #ATARIRL.load_model()
    ATARIRL.testing_model(5)
    ATARIRL.exit_env()
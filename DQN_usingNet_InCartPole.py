import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

env=gym.make('CartPole-v0',render_mode='human')
# env=gym.make('CartPole-v0')
state_size=env.observation_space.shape[0]
action_size=env.action_space.n
sample_size=512
batch_size=16

class DQN:
    def __init__(self,in_size,out_size):
        self.in_size=in_size
        self.out_size=out_size
        self.learning_rate=0.9
        self.build_model()
    def build_model(self):
        self.model=Sequential()
        self.model.add(Dense(32,input_dim=self.in_size,activation='relu',kernel_initializer='he_uniform'))
        self.model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
        self.model.add(Dense(self.out_size,activation='linear',kernel_initializer='he_uniform'))
        self.model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
    def train_model(self,input_data, test_data):
        self.model.fit(input_data,test_data,batch_size=batch_size,epochs=1,verbose=1)
    def load_model(self,name):
        self.model.load_weights(name)
    def save_model(self,name):
        self.model.set_weights(name)
class ReplayMemory:
    def __init__(self):
        self.data=deque(maxlen=1000)
    def size(self):
        return len(self.data)
    def save(self,state,action,reward,next_state,done):
        self.data.append((state,action,reward,next_state,done))
    def load(self):
        return random.sample(self.data,sample_size)
class Agent:
    def __init__(self,epsides=500,timesteps_per_episode=50,gamma=0.95,epsilon=1.0,decay_factor=0.995):    
        self.episodes=epsides
        self.epsilon=epsilon
        self.decay_factor=decay_factor
        self.timesteps_per_episode=timesteps_per_episode
        self.gamma=gamma
        self.dqn_policy=DQN(state_size,action_size)
        self.dqn_target=DQN(state_size,action_size)
        self.memory=ReplayMemory()
        self.state=np.zeros((1,state_size))
        self.action=0
    def predictAction_Egreedy(self,state):
        prob=np.random.random()
        action=np.random.randint(0,action_size)
        if prob>self.epsilon:
            action=np.argmax(self.dqn_policy.model.predict(state)[0])
        return action
    def update_targetmodel(self):
        self.dqn_target.model.set_weights(self.dqn_policy.model.get_weights())
    def train_tempmodel(self):
        if self.memory.size()>sample_size:
            samples=self.memory.load()
            state=np.zeros((sample_size,state_size))
            action=np.zeros(sample_size)
            reward=np.zeros(sample_size)
            next_state=np.zeros((sample_size,state_size))
            done=np.zeros(sample_size)
            for i in range(sample_size):
                state[i],action[i],reward[i],next_state[i],done[i]=samples[i]
            state=np.reshape(state,(sample_size,state_size))
            next_state=np.reshape(next_state,(sample_size,state_size))
            q_policy=self.dqn_target.model.predict(state)
            q_target=self.dqn_target.model.predict(next_state)
            for i in range(sample_size):
                if not done[i]:
                    q_policy[i][int(action[i])]=reward[i]+self.gamma*np.max(q_target[i])
            self.dqn_policy.train_model(state, q_policy)
    def train_agent(self):
        for episode in range(self.episodes):
            self.state=np.reshape(env.reset()[0], (1,state_size))
            done=False
            rewards=0
            while not done:
                action=self.predictAction_Egreedy(self.state)
                next_state,reward,terminal,truncated,inf=env.step(action)
                done=terminal or truncated
                next_state=np.reshape(next_state, (1,state_size))
                self.memory.save(self.state[0], action, reward, next_state[0], done)
                self.train_tempmodel()
                self.state=next_state
                rewards+=reward
            self.update_targetmodel()
            if self.epsilon>0.01:
                self.epsilon*=self.decay_factor
            print('epi=',episode+1,'      rew=',rewards)
    def test_agent(self):
        self.state=np.reshape(env.reset()[0], (1,state_size))
        done=False
        rewards=0
        while not done:
            action=self.predictAction_Egreedy(self.state)
            next_state,reward,terminal,truncated,inf=env.step(action)
            done=terminal or truncated
            next_state=np.reshape(next_state, (1,state_size))
            self.state=next_state
            rewards+=reward
        print('-------------------rew=',rewards)
if __name__=="__main__":
    agent=Agent()
    agent.train_agent()
    agent.test_agent()

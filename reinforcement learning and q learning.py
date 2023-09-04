#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[3]:


import gym
import numpy as np
import random


# In[4]:


Creating the Environment


# In[5]:


env = gym.make("Taxi-v3")
env.render()


# Hyperparameters

# In[6]:


total_episodes = 500000
total_test_episodes = 100


# Reinforcement Learning

# In[7]:


env.reset()
total_epochs = []
total_penalties = []
total_reward = []
frames = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    epochs, penalties, reward = 0,0,0
    
    while not done:
        action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        
        if reward == -10:
            penalties += 1
            
            epochs += 1
            
            total_reward.append(reward)
            
            frames.append({'episode':episode,'frames':env.render(mode='ansi'),'state':state,'action':action,'reward':reward})
            
            state = new_state
            total_penalties.append(penalties)
            total_epochs.append(epochs)


# Visualization

# In[8]:


from IPython.display import clear_output
from time import sleep
 
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        
        if i == total_test_episodes:
            break
            
        print(f"Episode: {frame['episode']}")
        print(frame['frames'])
        print(f"Timestep: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.6)
        
print_frames(frames)


# In[ ]:


print(f"Results after {total_test_episodes} episode:")
print(f"Average timesteps per episode: {sum(total_epochs)/total_test_episodes}")
print(f"Average reward per timesteps: {sum(total_reward)/sum(total_epochs)}")
print(f"Average penalties per episode: {sum(total_penalties)/total_test_episodes}")
print(f"Total Reward: {sum(total_reward)/total_test_episodes}")


# In[10]:


avg_ts_ncl = sum(total_epochs)/total_test_episodes
avg_reward_nrl = sum(total_reward)/sum(total_epochs)
avg_penalties_nrl = sum(total_penalties)/total_test_episodes


# Q learning

# In[9]:


max_steps = 99 #Maximum steps for each episode
learning_rate = 0.7
gamma = 0.6

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.03


# In[11]:


action_size = env.action_space.n
print("action size = ",action_size)

state_size = env.observation_space.n
print("state size = ",state_size)


# In[12]:


qtable = np.zeros((state_size,action_size))
print(qtable.shape)


# In[ ]:


for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False\44
    
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)
        
        if exp_exp_tradeoff>epsilon:
            action = np.argmax(qtable[state,:])
            
        else:
            action = env.action_space.sample()
            
        new_state,reward,done,info = env.step(action)
        
        qtable[state,action] = qtable[state,action]+ learning_rate*(reward+gamma*(np.max(qtable[new_state,:])) -qtable[state,action])
                                                                    
        state = new_state
        
        if done == True
            break
            
        epsilon = min_epsilon+(max_epsilon-min_epsilon) *np.exp(-decay_rate*episode)


# In[13]:


qtable


# In[16]:


env.reset()
total_epochs = []
total_penalties = []
total_reward = []
frames = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    epochs, penalties, reward = 0,0,0
    
    while not done:
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if reward == -10:
            penalties += 1
            
            epochs += 1
            
            total_reward.append(reward)
            
            frames.append({'episode':episode,'frames':env.render(mode='ansi'),'state':state,'action':action,'reward':reward})
            
            state = new_state
            total_penalties.append(penalties)
            total_epochs.append(epochs)


# In[17]:


from IPython.display import clear_output
from time import sleep
 
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        
        if frame['episode'] == total_test_episodes:
            break
            
        print(f"Episode: {frame['episode']}")
        print(frame['frames'])
        print(f"Timestep: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.0)
        
print_frames(frames)


# In[18]:


print(f"Results after {total_test_episodes} episode:")
print(f"Average timesteps per episode: {sum(total_epochs)/total_test_episodes}")
print(f"Average reward per timesteps: {sum(total_reward)/sum(total_epochs)}")
print(f"Average penalties per episode: {sum(total_penalties)/total_test_episodes}")
print(f"Total Reward: {sum(total_reward)/total_test_episodes}")


# In[19]:


avg_ts_ncl = sum(total_epochs)/total_test_episodes
avg_reward_nrl = sum(total_reward)/sum(total_epochs)
avg_penalties_nrl = sum(total_penalties)/total_test_episode


# In[20]:


import pandas as pd
import seaborn as sns


# In[ ]:


models = [("Random agent's performance",avg_reward_nr1,avg_penalties_nr1,avg_ts_nr1),("Q-learning agent's performance",avg_reward_rl,avg_penalties_rl,avg_ts_r1)]


# In[3]:


Algorithms = pd.Dataframe(data=models,columns=["Model", "Average rewards per move", "Average number of penalties per episode", "Average number of timesteps per trip"])


# In[ ]:


Algorithms


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x = 'Model',y = 'Average rewards per move',data = Algorithms)
plt.show()


# In[1]:


plt.figure(figsize=(10,10))
sns.barplot(x = 'Model',y = 'Average number of penalties per episode',data = Algorithms)
plt.show()


# In[ ]:





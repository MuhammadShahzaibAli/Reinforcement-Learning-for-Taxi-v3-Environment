# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:57:41 2023

@author: Shahzaib
"""

import gym
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random

def render(env):
    plt.imshow(env.render())
    plt.show()

env = gym.make("Taxi-v3", render_mode='rgb_array')
env.reset()
render(env)

action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)

# Q-Network

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(state_size)),
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dense(action_size,activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')
model.summary()

# Evaluation policy Q Network

def eval_dqn(dqn_,mod, num_of_episodes_, max_steps_,show=False):
    rewards=np.empty(num_of_episodes_)
    for e in range(num_of_episodes_):
        st=env.reset()[0]
        val=0
        for t in range(max_steps_):
            ac=np.argmax(mod.predict(tf.one_hot([st,st],state_size),verbose=0)[0])
            st,re,done,info=env.step(ac)[:4]
            val+=re
            if show:
                render(dqn_)
            if done:
                break
        rewards[e]=val
    return np.mean(rewards),np.min(rewards),np.max(rewards),np.std(rewards)

# Evaluation policy Q Learning

def eval_policy(en,qt,gam,it,ep,show=False):
    val_r=np.empty(ep)
    for e in range(ep):
        st=env.reset()[0]
        val=0
        for t in range(it):
            ac=np.argmax(qt[st])
            st,re,done,info=env.step(ac)[:4]
            val+=re
            if show:
                render(en)
            if done:
                break
        val_r[e]=val
    return np.mean(val_r),np.min(val_r),np.max(val_r),np.std(val_r)

# Q-Learning

qtable=np.zeros((state_size,action_size))
y=np.zeros((state_size,action_size))

episodes=3000
interactions=100
epsilon=1
alpha=0.5
gamma=0.9
hist=[]
re=[]


for episode in range(episodes):
    state=env.reset()[0]
    step=0
    done=False
    epsilon=0.99**(episode/5)
    
    for interact in range(interactions):
        
        if np.random.uniform(0,1)>epsilon:
            action=np.argmax(qtable[state,:])
        else:
            action=np.random.randint(0,action_size)
        
        new_state,reward,done,info=env.step(action)[:4]
        
        # update the Q-value
        
        qtable[state,action]=qtable[state,action]+alpha*(reward+gamma*np.max(qtable[new_state,:])-qtable[state,action])
        
        state=new_state
        
        if done==True:
            break
    
    # Q Learning Evaluation
    
    if episode % 25 == 0 or episode == 1:
        val_mean, val_min, val_max, val_std = eval_policy(env, qtable, gamma, 20, 500)
        hist.append([episode, val_mean,val_min,val_max,val_std])
        print(f'Q Learning mean reward for episode {episode} is {val_mean} \n')
        
        
env.reset()
        
        
hist = np.array(hist)
print(hist.shape)
fig, ax = plt.subplots()
ax.plot(hist[:,0],hist[:,1])
ax.fill_between(hist[:,0], (hist[:,1]-hist[:,4]), (hist[:,1]+hist[:,4]), color='b', alpha=.1)
plt.show()


x=np.arange(500)
x_one_hot=to_categorical(x,500)

# Q Network Learning and Evaluation

for ep in range(700):
    index=random.sample(range(500),400)
    model.fit(x_one_hot[index],qtable[index,:],epochs=15,verbose=1)

    if ep % 25 == 0 or ep == 1:
        dnq_mean, dnq_min, dnq_max, dnq_std=eval_dqn(env,model,10,50)
        re.append([ep, dnq_mean,dnq_std])
        print(f'Q Network mean reward for episode {ep} is {dnq_mean} \n')

re = np.array(re)
print(re.shape)
fig, ax = plt.subplots()
ax.plot(re[:,0],re[:,1])
ax.fill_between(re[:,0], (re[:,1]-re[:,2]), (re[:,1]+re[:,2]), color='b', alpha=.1)
plt.show()

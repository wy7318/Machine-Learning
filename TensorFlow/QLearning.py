'''Reinforcement Learning
Instead of inputting tons of data, let the agent to learn. i.e. Learning how to play a game by actually playing it
1) Environment : This is what our agent will explore. i.e. Level of the game, Or maze
2) Agent : Entiry that's exploring the environment. i.e. Mario is agent in Mario game
3) State : Where you are in the environment. i.e. position of agent is at in the environment
4) Action : i.e. Moving to the left or right no did nothing
5) Reward : What agent is trying to maximize
Goal is, 'Agent' explore the 'Environment' and go through some 'States' by 'Action' to achieve maximum 'Reward'

Q-Learning : Creating table of 'Actions' and 'States', predicting the rewards of each state by the action

Learning the Q-Table
1. Randomly picking a valid action
2. Using the current Q Table to find the best action
We need both actions to be balanced properly. Otherwise, the agent will always try to maximize its reward for the current state/action,
leading to 'local maxima'


Updating Q Value
Q[state, action] = Q[state, action] + A*(reward + r*max(Q[newState, :]) - Q[state, action])
a : stands fo 'learning rate'
r : stands for the 'discount factor'

Learning Rate : Numeric constant that defines how much change is permitted on each QTable update.
High learning rate means that each update will introduce a large change to the current state-action value.
Small learning rate means that each update has a more subtle change

Discount Factor : Balance how much focus is put on the current and future reward. High discount factor means that future rewards will be considered more heavily

'''

import gym
env = gym.make('FrozenLake-v1')     # Using FrozenLake environment

print(env.observation_space.n)      # get number of states
print(env.action_space.n)           # get number of actions

env.reset()                         # reset environment to default state
action = env.action_space.sample()  # get a random action
new_state, reward, done, info = env.step(action)      # take action, notice it return information
state = env.render()                        #render the GUI for the environment

import numpy as np
import time

STATES = env.observation_space.n
ACTIONS = env.action_space.n
Q = np.zeros((STATES, ACTIONS))     #Creating a matrix with all 0s

#Constants
EPISODES = 1500                    # how many times to run the environment from the beginning
MAX_STEPS = 100                     # max number of steps allowed for each run of environment

RENDER = False                      # see training or not. False being not watching

LEARNING_RATE = 0.81                # learning rate
GAMMA = 0.96                        # discount rate

#Picking an action
# 1. Randomly picking a valid action
# 2. Using the current Q Table to find the best action

epsilon = 0.9                       # start with a 90% chance of picking a random action

# code to pick action
if np.random.uniform(0, 1) < epsilon:       #we will check if a randomly selected value is less than epsilon
    action = env.action_space.sample()      # take random action
else:
    action = np.argmax(Q[state, :])         # use Q table to pick best action based on current values

Q[state, action] = Q[state, action] + LEARNING_RATE*(reward + GAMMA*np.max(Q[new_state, :]) - Q[state, action])


# Actual Example
rewards = []
for episode in range(EPISODES):         # for every episode

    state = env.reset()
    for _ in range(MAX_STEPS):          # explore environment until max_step

        if RENDER:
            env.render()

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)          # we don't care about info value

        Q[state, action] = Q[state, action] + LEARNING_RATE*(reward + GAMMA*np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break       #reached goal

print(Q)
'''Q Table
[[4.59813031e-01 1.43715586e-02 1.27186852e-02 1.46891221e-02]
 [1.81243694e-03 6.52244337e-03 6.29982821e-03 1.36802041e-02]
 [5.87836885e-03 6.18733003e-03 1.34546963e-03 1.20697554e-02]
 [3.73634893e-03 1.30626865e-03 1.18069443e-03 1.03409139e-02]
 [2.93069654e-01 6.71050399e-03 1.12702589e-02 1.01050618e-02]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.96365440e-05 1.42492001e-03 2.66453123e-05 3.79639615e-05]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [3.32658306e-03 2.31541518e-03 4.20572483e-03 1.67677295e-01]
 [8.36797003e-03 3.77814550e-01 4.54764258e-03 5.39276090e-03]
 [3.87441221e-04 6.71033348e-01 5.99391950e-04 3.46336615e-04]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [3.28960729e-03 1.04173531e-01 8.25276368e-01 1.06212999e-01]
 [8.25120350e-02 9.93593158e-01 1.74223660e-01 2.58295379e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
'''
print(f"Average Reward : {sum(rewards)/len(rewards)}:")
#Average Reward : 0.26666666666666666:

#we can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt

def get_average(values):
    return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

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

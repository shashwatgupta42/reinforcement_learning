# Reinforcement Learning
In this project, I use the Deep Q Learning algorithm to teach Q agents to play two versions of pong. 
1) Easy Pong - This is just simple single player pong. The Q agent has its paddle next to the right wall and the ball reflects back from the other three walls.
      - State - [ball x position, ball y position, ball direction, paddle postion]
      - Actions - [do nothing, move up, move down]
3) Hard Pong - In this version, the left, top, and bottom walls change randomly every time the ball hits the paddle. This makes the game much harder.
      - State - [ball x position, ball y position, ball direction, paddle postion, left wall position, top wall position, bottom wall position]
      - Actions - [do nothing, move up, move down]
  
<b> All the parts of this project - the deep Q learning algorithm, game environments, neural nets (with suitable backward pass) - have been written by me. No machine learning library has been used for anything. </b>

### The following algorithm has been used in this project
<img width="1271" alt="deep_q_algorithm" src="https://github.com/shashwatgupta42/reinforcement_learning/assets/142345559/49d05507-982e-4de5-afce-df8b5237c8b2">

## Content - 
- AI_plays_pong.py - This is the file responsible for running the games with the trained Q Agents. Both easy and hard pong games can be run by this file by making small changes in two lines at the top. Where the changes are needed is already written in this file.
  
- train.ipynb - The jupyter notebook for training the easy pong agent.

- hard_pong_train.ipynb - The jupyter notebook for training the hard pong agent.

- pong_env.py - The file responsible for generating the easy pong environment. Run the script to play easy pong yourself.

- hard_pong_env.py - The file responsible for generating the hard pong environment. Run the script to play hard pong yourself.

- RL_NN.py -
   - Handles all the functionality related to using neural networks as state-action value functions.
   - Generates, handles, and updates both the Q-Networks (original and target Q-networks).
   - Allows arbitrary depth and size.
   - Provides multiple activation functions.
   - Performs forward and backward pass
   - Performs update step on the original Q-network using ADAM and updates the target network using soft update.
   - Generates the training examples

- saved_param - folder containing the trained and untrained Q agents for hard and easy pong.
  
- img - folder containing some utility image

- demo - folder containing demo videos of both trained and untrained agents playing hard and easy pongs.

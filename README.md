# Car Racing AI Gym Environment

A stub notebook and conda env to start using Active Learning in the Car Racing environment of AI Gym. HAVE FUN!!!

# Active Learning

Active learning is the type of learning where the learner does not have access to training data directly. Instead the agents receives rewards based on actions taken and a current state of the environment:
1. Actions: is a set of options that the agent could take at every stage
2. State: is a set of possible configurations of the environment
3. Rewards: Negative or Positive number indicating if an action resulted in a good/bad outcome based on current state

# Prepare environment

A few options are available as follow.

## 1. Using conda and requirements.txt

```bash
  $ conda create --name aigym --file requirements.txt
  $ conda activate aigym
  $ pip install gym
  $ pip install box2d-py
  $ jupyter notebook
```

## 2. Using public miniconda (outside of ING VPN)

```bash
  $ conda create --file environment_no_vpn.yml
  $ conda activate gym
  $ jupyter notebook
```


## 3. Manual installation

* Create a python3 conda env ```conda create --name aigym python``` assuming python3 is the default version
* Activate the env ```conda activate aigym```
* Install jupyter notebook ```conda install jupyter```
* Install gym (the active learning library we are using) ```pip install gym```
* Install swig (don't ask why) ```conda install swig```
* Install box2d, A 2D Physics Engine for Games ```pip install box2d-py```
* Run ```jupyter notebook``` in this directory and run the notebook and you're done. Modify the notebook to improve the AI

# If you are doing this as part of a hackathon

* Please order pizza and beer/coke for the break
* Please don't shower
* Please don't straight copy a solution from the internet. But feel free to be inspired

# A summary of gym

* the ```env``` object is the main object
* ```env.observation_space``` and ```env.action_space``` returns the shape of the observation space and action space respectively (point, vector, matrix, ... etc). It is called ```Box``` in gym because it is usually bounded
* ```env.step``` a function that takes an action (with the same form specified by ```env.action_space``` as a python list) and returns a tuple ```(observation, reward, done, info)```
  * The observation is the state of the environment in the form specified by ```env.observation_space```
  * The reward is the active learning reward (go to the AI gym page about this environment to see how it is calculated)
  * Done is whether the current simulation has ended
  
# Episodes and stages

* Episode is a simulation which is a sequence of steps which are:
* A step consists of:
  * Take action
  * Observe reward and new state
  * Learn
  
In AI gym an episode ends either when a certain success rule is acheived or something goes completely wrong, otherwise it will keep going
  
# The goal

* The goal is to maximise the cummulative reward of simulations:
  * Calculate the cumulative reward for the first 2000 steps of each episode and keep the best one and report it at the end
  * Show us a simulating of your car

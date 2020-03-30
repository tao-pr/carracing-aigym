# Car Racing AI Gym Environment

A stub notebook and conda env to start using Active Learning in the Car Racing environment of AI Gym. HAVE FUN!!!

# Installation using conda and requirements.txt

```conda create --name aigym --file requirements.txt```
```jupyter notebook```

# Manual installation

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
  
# The goal

* The goal is to maximise the reward of simulations.

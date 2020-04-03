import os
import numpy as np
import joblib

from .encoder import *

class Agent:
  """
  Base reinforcement learning agent with basic interface
  """
  def __init__(self, learning_rate=0.25):
    self.learning_rate = learning_rate

    # Two-level dictionaries
    self.policy = dict() # [state => action => reward]
    self.state_machine = dict() # [state => state' => action]

    # Binding state and action encoder (np.array => str)
    self.encoder = StateActionEncoder() # TODO Switch to its child class

  def reset(self):
    """
    Reset all internal states
    """
    pass

  def learn(self, state, action, reward, next_state):
    """
    Learn that:
    - If we take an action (int) on the state (np.array)`
    - we will get back the reward (double value)
    - and register the next state (np.array)
    """
    pass

  def best_action(self, state):
    """
    Return the best action to take on the specified state 
    to maximise the possible reward
    """
    statehash = self.encoder.encode_state(state)

    if statehash not in self.state_machine:
      # Unrecognised state, return no recommended action
      return (-1, 0)
    best_action = -1
    best_reward = 0
    for next_statehash in self.state_machine[statehash]:
      a = self.state_machine[statehash][next_statehash]
      v = self.get_v(statehash, a)
      if v >= best_reward:
        best_reward = v
        best_action = a
    return (best_action, best_reward)

  def get_v(self, state, action):
    """
    Evaluate the reward value of `action` taken on the `state`
    """
    statehash = self.encoder.encode_state(state)
    actionhash = self.encoder.encode_action(action)
    if statehash not in self.policy:
      print("...new state")
      self.policy[statehash] = {actionhash: 0}
      return 0
    elif actionhash not in self.policy[statehash]:
      print("...known state, new action")
      self.policy[statehash][actionhash] = 0
      return 0
    else:
      print("...take action from experience")
      return self.policy[statehash][actionhash]

  def save(self, path):
    with open(path, "wb") as f:
      print("Saving the agent to {}".format(path))
      joblib.dump(self, f, compress=1)

  @staticmethod
  def load(path, default):
    if os.path.isfile(path):
      with open(path, "rb") as f:
        print("Agent loaded from {}".format(path))
        return joblib.load(f)
    else:
      print("No agent file to load, created a new one")
      return default


class TDAgent(Agent):
  """
  Temporal difference
  """
  def __init__(self, learning_rate=0.5, alpha=0.7):
    super().__init__(learning_rate)
    self.alpha = alpha
    self.encoder = PartialScreenStateActionEncoder()

  def learn(self, state, action, reward, next_state):
    statehash    = self.encoder.encode_state(state)
    newstatehash = self.encoder.encode_state(next_state)
    actionhash   = self.encoder.encode_action(action)
    
    old_v = self.get_v(state, action)
    new_v = self.get_v(next_state, action)

    # Update policy (weighted temporal difference)
    diff = self.learning_rate * (reward + self.alpha * new_v - old_v)
    self.policy[statehash][actionhash] = old_v + diff


class QAgent(Agent):
  """
  Simple Q-learning
  """
  pass

class PGAgent(Agent):
  """
  Policy Gradient
  """
  pass
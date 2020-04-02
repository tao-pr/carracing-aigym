import numpy as np
import joblib

class Agent:
  """
  Base reinforcement learning agent with basic interface
  """
  def __init__(self, learning_rate=0.25):
    self.alpha = learning_rate

    # Two-level dictionaries
    self.policy = dict() # [state => action => reward]
    self.state_machine = dict() # [state => state' => action]

  def reset(self):
    """
    Reset all internal states
    """
    pass

  def encode_state(self, state) -> str:
    """
    Encode `state` (np.array) as observed from the environment 
    into a hashable string
    """
    return ""

  def learn(self, state, action, reward, next_state):
    """
    Learn that:
    - If we take an action (int) on the state (np.array)`
    - we will get back the reward (double value)
    - and register the next state (np.array)
    """
    old_v = self.get_v(state, action)
    new_v = self.get_v(next_state, action)

    # Update policy
    self.policy[]

    # Update state machine
    # TAOTODO

  def best_action(self, state):
    return 0

  def get_v(self, state, action):
    """
    Evaluate the reward value of `action` taken on the `state`
    """
    statehash = self.encode_state(state)
    if statehash not in self.policy:
      self.policy[statehash] = {action: 0}
      return 0
    elif action not in self.policy[statehash]:
      self.policy[statehash][action] = 0
      return 0
    else:
      return self.policy[state][action]

  def save(self, path):
    with open(path, "wb") as f:
      print("Saving the agent to {}".format(path))
      joblib.dump(self, f, compress=1)

  @staticmethod
  def load(path):
    if os.path.isfile(path):
      with open(path, "rb") as f:
        return joblib.load(f)
import numpy as np

class StateActionEncoder:
  """
  Dummy state, action encoder.
  Please override!
  """

  def encode_state(self, s):
    return np.array2string(s, precision=0)

  def encode_action(self, s):
    return np.array2string(s)

  def decode_action(self, s):
    return np.fromstring(s)
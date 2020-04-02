import numpy as np
import cv2

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


class PixelateStateActionEncoder(StateActionEncoder):

  def __init__(self):
    self.n = 0

  def encode_state(self, s):
    """
    Encode 96x96x3 observation into 8x8
    """
    b8x8 = cv2.resize(s, (8,8))

    # DEBUG
    cv2.imwrite("debug/{}.png".format(self.n), b8x8)
    self.n = (self.n+1)%100

    for y in range(8):
      for x in range(8):
        b8x8[y,x] = b8x8[y,x]//32
    return np.array2string(b8x8, precision=0)


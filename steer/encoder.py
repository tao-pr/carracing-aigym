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

  def encode_color(self, c):
    b,g,r = c
    if min(b,g,r)==0: return 0 # Black
    elif g>=200 or g>=2*r: return 255 # Green
    elif abs(100-r)<30 and abs(100-g)<30 and abs(100-b)<30: return 128 # Gray tile
    else: return 64 # Otherwise
 

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


class PartialScreenStateActionEncoder(StateActionEncoder):

  def __init__(self):
    self.n = 0

  def encode_state(self, s):
    """
    Only encode bottom horizontal section and middle lane (line of sight)
    """

    frame = cv2.resize(s, (8,8))
    for y in range(frame.shape[0]):
      for x in range(frame.shape[1]):
        frame[y,x] = self.encode_color(frame[y,x])

    print(frame.shape)

    cv2.imwrite("debug/f-{}.png".format(self.n), frame)
    v = frame.flatten()

    #cv2.imwrite("debug/bottom.png", bottom)
    #cv2.imwrite("debug/middle.png", middle)

    # Encode colour
    # for y in range(bottom.shape[0]):
    #   for x in range(bottom.shape[1]):
    #     bottom[y,x] = self.encode_color(bottom[y,x])

    # for y in range(middle.shape[0]):
    #   for x in range(middle.shape[1]):
    #     middle[y,x] = self.encode_color(middle[y,x])

    #cv2.imwrite("debug/bottom-ec.png", bottom)
    #cv2.imwrite("debug/middle-ec.png", middle)

    code = np.array2string(v, precision=0)
    print(code)
    return code
    

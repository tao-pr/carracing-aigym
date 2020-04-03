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

    # frame = cv2.resize(s, (16,16))
    # for y in range(frame.shape[0]):
    #   for x in range(frame.shape[1]):
    #     frame[y,x] = self.encode_color(frame[y,x])
    frame = s[:80, :, :]

    # Box encode
    box = np.zeros_like(frame)
    box_size = 8
    for y in np.arange(0, frame.shape[0], box_size):
      for x in np.arange(0, frame.shape[1], box_size):
        b,g,r = frame[y,x]
        cv2.rectangle(
          box,
          (x,y), 
          (min(x+box_size, frame.shape[1]), min(y+box_size, frame.shape[0])),
          [int(i) for i in [b,g,r]], 
          -1)

    cv2.imwrite("debug/f-{}.png".format(self.n), box)
    self.n = self.n+1
    v = box.flatten()
    print("...State size : ", v.shape)
    code = np.array2string(v, precision=0)
    return code
    

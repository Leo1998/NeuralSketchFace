import os
import numpy as np
import cv2


def canny(image, sigma=0.0):
  #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  grayed = np.where(image < 20, 255, 0)

  lower = sigma*128 - 128
  upper = 255
  edges = cv2.Canny(image, lower, upper)

  return np.maximum(edges, grayed)


IMAGE_W = 160
IMAGE_H = 160
IN_DIR = 'faces'
OUT_DIR = 'parsed_data'

def parse_data():
  idx = 0
  X = []
  Y = []
  for root, subdirs, files in os.walk('faces'):
    for file in files:
      path = root + '/' + file
      
      if not (path.endswith('.jpg') or path.endswith('.png')):
        continue
      image = cv2.imread(path)
      if image is None:
        assert(False)

      print("parsing face {}".format(idx))

      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      image = cv2.resize(image, (IMAGE_W, IMAGE_H), interpolation= cv2.INTER_LINEAR)

      sketch_img = canny(image, 0.1)
      
      X.append(sketch_img)
      Y.append(image)

      idx+=1
  X = np.array(X).reshape(-1, IMAGE_W, IMAGE_H, 1)
  Y = np.array(Y).reshape(-1, IMAGE_W, IMAGE_H, 1)

  print(X.shape)
  print(Y.shape)

  np.save("X", X)
  np.save("Y", Y)



if __name__ == '__main__':
  parse_data()

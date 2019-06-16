import os
import numpy as np
import cv2


def canny(image, sigma=0.0):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  grayed = np.where(gray < 20, 255, 0)

  lower = sigma*128 - 128
  upper = 255
  edges = cv2.Canny(image, lower, upper)

  return np.maximum(edges, grayed)


IMAGE_W = 160
IMAGE_H = 160
IN_DIR = 'faces'
OUT_DIR = 'parsed_data'

def parse_data():
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

  idx = 0
  for root, subdirs, files in os.walk('faces'):
    for file in files:
      path = root + '/' + file
      
      if not (path.endswith('.jpg') or path.endswith('.png')):
        continue
      image = cv2.imread(path)
      if image is None:
        assert(False)

      image = cv2.resize(image, (IMAGE_W, IMAGE_H), interpolation= cv2.INTER_LINEAR)

      out_path = OUT_DIR + '/' + "img_{}x{}_{}.png".format(IMAGE_W, IMAGE_H, idx)
      
      print("Parsing image {} to output file {}".format(path, out_path))
      cv2.imwrite(out_path, image)

      sketch_img = canny(image, 0.1)
      out_path = OUT_DIR + '/' + "img_{}x{}_{}_sketch.png".format(IMAGE_W, IMAGE_H, idx)
      cv2.imwrite(out_path, sketch_img)
      
      idx+=1


if __name__ == '__main__':
  parse_data()

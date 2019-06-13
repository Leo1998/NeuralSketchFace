import os
import numpy as np
import cv2

IMAGE_W = 120
IMAGE_H = 120
IN_DIR = 'faces'
OUT_DIR = 'parsed_data'

def parse_data():
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
      idx+=1


if __name__ == '__main__':
  parse_data()

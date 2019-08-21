import numpy as np
import cv2

MODEL_FILENAME = "net-50E.model"
IMAGE_W = 160
IMAGE_H = 160

img = np.zeros((400, 400, 1), np.uint8)

import keras
from keras.models import Sequential, load_model

model = load_model(MODEL_FILENAME)



ix = 0
iy = 0
drawing = False

def draw_callback(event, x, y, flags, param):
  global drawing, ix, iy

  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    ix, iy = x, y
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing:
      cv2.line(img, (ix, iy), (x, y), (255,255,255), thickness=3)
      ix, iy = x, y
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False
    updateOutput()

def updateOutput():
  out_img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)

  out_img = np.expand_dims(out_img, axis=2)
  out_img = np.expand_dims(out_img, axis=0)

  out_img = model.predict(out_img)[0]

  out_img = cv2.resize(out_img, (400, 400), interpolation=cv2.INTER_LINEAR)

  cv2.imshow('image2', out_img)


cv2.namedWindow('image1')
cv2.setMouseCallback('image1', draw_callback)

while(1):
  cv2.imshow('image1', img) 
  k = cv2.waitKey(1) & 0xff
  if k == 27:
    break

cv2.destroyAllWindows()

# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="path to input image")
ap.add_argument("-p", "--pbtxt", default="kanji_output.pbtxt",
  help="path to Tensorflow 'deploy' pbtxt file")
ap.add_argument("-m", "--model", default="frozen_inference_graph.pb",
  help="path to Tensorflow pre-trained model .pb file")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
  help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# CLASSES = ["background", "word"]
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 4))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])
# net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph_v2_40000.pb", "output_v2.pbtxt")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then swap RGB it
image = cv2.imread("images/test4.png")
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
# loop over the detections
for i in np.arange(0, detections.shape[2]):
  # extract the confidence (i.e., probability) associated with the
  # prediction
  confidence = detections[0, 0, i, 2]

  # filter out weak detections by ensuring the `confidence` is
  # greater than the minimum confidence
  if confidence > args["confidence"]:
    # compute the (x, y)-coordinates of the bounding box for
    # the object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    # crop the detected kanji
    crop_img = image[startY:endY, startX:endX].copy()
    # save the cropped kanji image
    cv2.imwrite("{}.png".format(i), crop_img)

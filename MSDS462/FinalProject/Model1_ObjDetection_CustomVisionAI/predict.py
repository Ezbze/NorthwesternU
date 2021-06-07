# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import io
import re
import time
import picamera
from object_detection import ObjectDetection

from annotation import Annotator

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'
CAMERA_WIDTH = 680
CAMERA_HEIGHT = 450


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def annotate_objects(annotator, selected_boxes, selected_classes, selected_probs, labels):
  """Draws the bounding box and label for each object in the results."""
  for i in range(len(selected_boxes)):
    left   = round(float(selected_boxes[i][0]), 8)
    top    = round(float(selected_boxes[i][1]), 8)
    width  = round(float(selected_boxes[i][2]), 8)
    height = round(float(selected_boxes[i][3]), 8)

    xmin = int(left   * (CAMERA_WIDTH))
    xmax = int(width  * (CAMERA_WIDTH))
    ymin = int(top    * (CAMERA_HEIGHT))
    ymax = int(height * (CAMERA_HEIGHT))

    probability = round(float(selected_probs[i]), 8)
    tagName = labels[selected_classes[i]]
    
    ## Overlay the box, label, and score on the camera preview
    annotator.bounding_box([xmin, ymin, xmax, ymax])
    annotator.text([xmin, ymin],
                   '%s\n%.2f' % (tagName, probability))
        

def main():
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

    
    #predictions = od_model.predict_image(image)
    #print(predictions)
    
    with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            annotator = Annotator(camera)
            for _ in camera.capture_continuous(
                stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                #image = Image.open(stream).convert('RGB').resize(
                #    (60, 60), Image.ANTIALIAS)
                image = Image.open(stream).convert('RGB')
                start_time = time.monotonic()
                selected_boxes, selected_classes, selected_probs = od_model.predict_image(image)
                elapsed_ms = (time.monotonic() - start_time) * 1000
                #print(results)
                annotator.clear()
                annotate_objects(annotator, selected_boxes, selected_classes, selected_probs, labels)
                annotator.text([5, 0], '%.1fms' % (elapsed_ms))
                annotator.update()
                #print(results)
                stream.seek(0)
                stream.truncate()
        
        finally:
            camera.stop_preview()


if __name__ == '__main__':
        main()

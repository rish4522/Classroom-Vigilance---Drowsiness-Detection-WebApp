from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('/best.pt')

# Perform object detection on the input image
results = model.predict('D:/detectionProject/images/test.jpg', show=True)
cv2.waitKey(0)
# result = results[0]
# numOfDetections = len(result.boxes)
# print("Number of Detections: ", numOfDetections)
#
# box = result.boxes[0]
#
# print("Object type:", box.cls)
# print("Coordinates:", box.xyxy)
# print("Probability:", box.conf)
#
# print("Object type:",box.cls[0])
# print("Coordinates:",box.xyxy[0])
# print("Probability:",box.conf[0])
#
#
# cords = box.xyxy[0].tolist()
# class_id = box.cls[0].item()
# conf = box.conf[0].item()
# print("Object type:", class_id)
# print("Coordinates:", cords)
# print("Probability:", conf)
#
# cords = box.xyxy[0].tolist()
# cords = [round(x) for x in cords]
# class_id = result.names[box.cls[0].item()]
# conf = round(box.conf[0].item(), 2)
# print("Object type:", class_id)
# print("Coordinates:", cords)
# print("Probability:", conf)
#
#
# for box in result.boxes:
#   class_id = result.names[box.cls[0].item()]
#   cords = box.xyxy[0].tolist()
#   cords = [round(x) for x in cords]
#   conf = round(box.conf[0].item(), 2)
#   print("Object type:", class_id)
#   print("Coordinates:", cords)
#   print("Probability:", conf)
#   print("---")
#
# # Print a message indicating where the image is saved
# print("Image with detections saved as 'detections.jpg'")




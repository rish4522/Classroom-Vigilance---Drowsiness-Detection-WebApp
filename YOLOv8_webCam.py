import math
from ultralytics import YOLO
import cv2
import time
from datetime import datetime
global stop_flag
stop_flag = False


def video_detection(path_x, app, mysql, session_data):
   print(type(session_data)) # Add this line
   print(session_data) # Add this line
   with app.app_context():
       global stop_flag
       video_capture = path_x
       cap = cv2.VideoCapture(video_capture)
       frame_width = int(cap.get(3))
       frame_height = int(cap.get(4))

       model = YOLO('D:/detectionProject/best.pt')

       classNames = ['sleeping']
       num_detections = 0

       while not stop_flag:
           success, img = cap.read()
           results = model(img, stream=True)
           # num_detections = 0 # Initialize counter
           for r in results:
               boxes = r.boxes
               for box in boxes:
                  x1, y1, x2, y2 = box.xyxy[0]
                  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                  print(x1, y1, x2, y2)
                  cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                  conf = math.ceil((box.conf[0]*100))/100
                  cls = int(box.cls[0])
                  class_name = classNames[cls]
                  # detection_time = datetime.now()
                  label = f'{class_name}{conf}'
                  t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                  c2 = x1 + t_size[0], y1 - t_size[1] - 3
                  cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                  cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                  num_detections += 1 # Increment counter
           if num_detections > 0:
               session_id = str(session_data['session_id'])
               detection_time = datetime.now()
               class_name = str(classNames[cls])
               conf = float(math.ceil((box.conf[0] * 100)) / 100)
               num_detections = int(num_detections)

               # Insert counter value into database
               try:
                   cur = mysql.connection.cursor()
                   cur.execute('''INSERT INTO detections (session_id, detection_time, detected_class, confidence, number_of_detections)
                                       VALUES (%s, %s, %s, %s, %s)''',
                              (session_id, detection_time, class_name, conf, num_detections))
               except Exception as e:
                   print(f"Failed to insert data into database. Error: {e}")
               finally:
                   mysql.connection.commit()
                   cur.close()

           yield img
           num_detections = 0
           time.sleep(10) # Pause for 10 seconds
       cap.release()
       cv2.destroyAllWindows()


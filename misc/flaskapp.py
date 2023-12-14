from flask import Flask, Response,jsonify, request, session, render_template
import cv2

from YOLOv8_webCam import video_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'rish4522'

def generate_frames(path_x = ''):

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame/r/n'
                    b'Content-Type: image/jpeg/r/n/r/n' + frame + b'/r/n')

@app.route('/feed')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary = frame')


if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, Response, jsonify, request, session, render_template

from flask_wtf import FlaskForm
from flask_mysqldb import MySQL

from wtforms import FileField, SubmitField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import secrets
import cv2
from threading import Thread
from YOLOv8_webCam import video_detection

app = Flask(__name__)

app.config['SECRET KEY'] = 'Kusumpathi@121'
app.secret_key = '@alienforce'
app.config['UPLOAD_FOLDER'] = 'static/files'

# MySQL configurations
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'rish'
# app.config['MYSQL_DB'] = 'detectiondbase'

app.config['MYSQL_HOST'] = 'detectiondbase.cry4f3qhljwv.us-east-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'rishabh'
app.config['MYSQL_PASSWORD'] = 'alienforce'
app.config['MYSQL_DB'] = 'detectionDbase'

# Create a MySQL instance
mysql = MySQL(app)

session_id = secrets.token_hex(16)

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    # conf_slide = IntegerRangeField('Confidence: ', default=25, validators=[InputRequired()])
    submit = SubmitField("Run")

stop_flag = False

def generate_frames_web(path_x, session_data):
    yolo_output = video_detection(path_x, app, mysql, session_data)
    for detection_ in yolo_output:
        if stop_flag:
            break
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
    cur = mysql.connection.cursor()
    cur.execute('''SELECT * FROM detections''')
    data = cur.fetchall()
    cur.close()
    return render_template('index.html', data=data)

@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    global stop_flag
    stop_flag = False
    session_data = {'session_id': secrets.token_hex(16)}
    session['session_id'] = session_data['session_id']
    # session.clear()
    cur = mysql.connection.cursor()
    mysql.connection.commit()
    cur.close()
    return render_template('ui.html')

@app.route('/webapp')
def webapp():
   global stop_flag
   if stop_flag:
       stop_flag = False
   session_data = {'session_id': session.get('session_id')}
   return Response(generate_frames_web(path_x=0, session_data=session_data), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reports', methods=['GET', 'POST'])
def reports():
    global stop_flag
    stop_flag = True

    cur = mysql.connection.cursor()
    cur.execute('''SELECT * FROM detections ORDER BY detection_time DESC LIMIT 15''')
    data = cur.fetchall()
    cur.close()

    # Converting data into a list of dictionaries
    reports = [
        {'session_id': row[1], 'detection_time': row[2], 'detected_class': row[3], 'confidence': row[4],
         'number_of_detections': row[5]} for row in data]
    return render_template('reports.html', reports=reports)


if __name__ == '__main__':
    app.run(debug=True)

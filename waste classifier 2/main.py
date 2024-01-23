import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier
from flask import Flask, render_template, Response, session, redirect, url_for

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Function to load images from a folder
def load_images_from_folder(folder):
   images = []
   pathlist = os.listdir(folder)
   for path in pathlist:
       images.append(cv2.imread(os.path.join(folder, path), cv2.IMREAD_UNCHANGED))
   return images

# Load waste and bins images
imgWastelist = load_images_from_folder("Resources/Waste")
imgBinslist = load_images_from_folder("Resources/Bins")

# Load background image
imgBackground = cv2.imread("Resources/background.png")

# Load arrow image
imgArrow = cv2.imread("Resources/arrow.png", cv2.IMREAD_UNCHANGED)

# Load classifier model
Classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

# Mapping of class IDs to bin IDs
classDic = {0: None, 1: 0, 2: 0, 3: 3, 4: 3, 5: 1, 6: 1, 7: 2, 8: 2}

# Initialize variables
classIDBin = 0

# Function to generate frames
def generate_frames():
   cap = cv2.VideoCapture(0)
   while True:
       _, img = cap.read()
       imgResize = cv2.resize(img, (454, 340))
       imgBackground = cv2.imread("Resources/background.png")
       prediction = Classifier.getPrediction(img)
       classID = prediction[1]

       if classID != 0:
           imgBackground = cvzone.overlayPNG(imgBackground, imgWastelist[classID - 1], (909, 127))
           imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
           classIDBin = classDic[classID]

       imgBackground = cvzone.overlayPNG(imgBackground, imgBinslist[classIDBin], (895, 374))
       imgBackground[148:148 + 340, 159:159 + 454] = imgResize

       ret, jpeg = cv2.imencode('.jpg', imgBackground)
       frame = jpeg.tobytes()
       yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
   return render_template('welcome.html')

@app.route('/start')
def start():
   if 'running' not in session:
       session['running'] = True
   return render_template('start.html')

@app.route('/stop')
def stop():
   if 'running' in session:
       session.pop('running')
   return render_template('stop.html')

@app.route('/video_feed')
def video_feed():
   if 'running' not in session:
       return redirect(url_for('index'))
   return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
   app.run(debug=True)

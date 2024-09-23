from operator import ge
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from keras_preprocessing import image
from keras_preprocessing.image import img_to_array
from flask import Flask , request, render_template, Response, redirect, url_for, flash
from werkzeug.utils import secure_filename
import glob
import jyserver.Flask as jsf
from camera import VideoCamera

app = Flask(__name__)

#--------------------------------------------------Load Model-------------------------------------------------------------#

age_model = load_model("age_cnn.h5")
gender_model = load_model("gender_cnn.h5")

#--------------------------------------------------Dependencies-----------------------------------------------------------#

UPLOAD_FOLDER = 'static/uploads/'

gender_dict = {0:'Male', 1:'Female'}
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def giveFile():
    list_of_files = glob.glob('C:/Users/sanja/Desktop/detect_v2/static/uploads/*')
    filepath = max(list_of_files, key=os.path.getctime)
    return filepath

# ------------------------------------------------------Initial Routes-------------------------------------------------------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/openCam')
def openCam():
    return render_template('openCam.html')

# ------------------------------------------------------Image Prediction---------------------------------------------------#


@app.route('/predictfile',methods = ['GET','POST'])
def predictfile():
    list_of_files = glob.glob('C:/Users/sanja/Desktop/detect_v2/static/uploads/*')
    filepath = max(list_of_files, key=os.path.getctime)
    filepath = giveFile()
    print("Filepath of uploaded image" + filepath)
    
    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    frame = cv2.imread(filepath)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        print("Face Found!")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray=cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = roi_gray / 255.0
    
        # Age prediction
        age_predict = age_model.predict(roi_gray.reshape(1, 128, 128, 1))
        # Gender prediction
        gender_predict = gender_model.predict(roi_gray.reshape(1, 128, 128, 1))
        age = round(age_predict[0][0])
        print("Predicted age: "+ str(age))
        gender = gender_dict[round(gender_predict[0][0])]
        print("Predicted gender: "+ gender)
        # age_label_position = (x+h, y+h)
        # cv2.putText(frame, "Age="+str(age), age_label_position,
        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    text = str(age) + ', ' + gender
    flash("Predicted age and Gender is: " + text)
    return render_template('result.html',filename=storedFile)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/result', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        global storedFile
        storedFile = filename
        return render_template('result.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)






#--------------------------------------------------Video Prediction-----------------------------------------------------#

@app.route('/video')
def video():
    return Response(getframes(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


def getframes(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


if __name__ == "__main__":
    app.run(debug = False, threaded = False)

import cv2
from model import Model

class VideoCamera(object):
    def __init__(self):
        self.min=100
        self.max=0
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        vgg16 = Model()
        age_model = vgg16.load_model(vgg16.age_model_path, vgg16.age_caffemodel, vgg16.age_prototxt)
        gender_model = vgg16.load_model(vgg16.gender_model_path, vgg16.gender_caffemodel, vgg16.gender_prototxt)
        gender_dict = {0:'Female', 1:'Male'}
        success, frame_bgr = self.video.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = vgg16.detector(frame_rgb, 1)

        for d in faces:
            left = int(0.6 * d.left())     # + 40% margin
            top = int(0.6 * d.top())       # + 40% margin
            right = int(1.4 * d.right())   # + 40% margin
            bottom = int(1.4 * d.bottom()) # + 40% margin
            face_segm = frame_rgb[top:bottom, left:right]
            gender, gender_confidence = vgg16.predict(gender_model, face_segm, vgg16.input_height, vgg16.input_width)
            gender = gender_dict[round(gender)]
            age, age_confidence = vgg16.predict(age_model, face_segm, vgg16.input_height, vgg16.input_width)
            text = '{} ({:.2f}%), {} ({:.2f}%)'.format(gender, gender_confidence*100, age, age_confidence*100)
            cv2.putText(frame_bgr, text, (d.left(), d.top() - 20), vgg16.font, vgg16.fontScale, vgg16.fontColor, vgg16.lineType)
            cv2.rectangle(frame_bgr, (d.left(), d.top()), (d.right(), d.bottom()), vgg16.fontColor, 2)
            
        ret, jpeg = cv2.imencode('.jpg', frame_bgr)
        return jpeg.tobytes()
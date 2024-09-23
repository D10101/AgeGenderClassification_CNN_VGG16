import cv2
import dlib
from os.path import join

class Model(object):
    def __init__(self):
        self.input_height = 224
        self.input_width = 224
        self.age_model_path = './'
        self.age_caffemodel = 'dex_chalearn_iccv2015.caffemodel'
        self.age_prototxt = 'age.prototxt.txt'
        self.gender_model_path = './'
        self.gender_caffemodel = 'gender.caffemodel'
        self.gender_prototxt = 'gender.prototxt.txt'
        self.detector = dlib.get_frontal_face_detector()
        self.font, self.fontScale, self.fontColor, self.lineType = cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1
        
    def load_model(self, model_path, caffemodel, prototxt):
        caffemodel_path = join(model_path, caffemodel)
        prototxt_path = join(model_path, prototxt)
        model = cv2.dnn.readNet(prototxt_path, caffemodel_path)

        return model
    
    def predict(self, model, img, height, width):
        face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
        model.setInput(face_blob)
        predictions = model.forward()
        class_num = predictions[0].argmax()
        confidence = predictions[0][class_num]

        return class_num, confidence
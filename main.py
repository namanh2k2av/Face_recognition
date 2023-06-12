import cv2
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras.models import Model, load_model
from process import get_embedding

model_fn = load_model('model/facenet_keras.h5')
data = load('data/data_embedding.npz')
trainX, trainy = data['arr_0'], data['arr_1']
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
model_svc = SVC(kernel='linear', probability=True)
model_svc.fit(trainX, trainy)

def face_recog():
    camara=cv2.VideoCapture(0)
    camara.set(3, 800) ##chiều dài
    camara.set(4, 580)  ##chiều rộng

    def result(img):
        img_test = get_embedding(model_fn, cv2.resize(img, dsize=(160, 160)))
        samples = expand_dims(img_test, axis=0)
        yhat_class = model_svc.predict(samples)
        yhat_prob = model_svc.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        return predict_names[0], class_probability

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    camara = cv2.VideoCapture(0)
    camara.set(3, 800)
    camara.set(4, 580)
    camara.set(10, 150)
    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret, img = camara.read()
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = faceCascade.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 3)
            id, predict = result(img[y:y + h, x:x + w])
            if predict > 60:
                cv2.putText(img, f"Name: {id}", (10, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (225, 0, 0), 2)
                cv2.putText(img, f"Do chinh xac: {round(predict, 2)}", (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (225, 0, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Khong xac dinh", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("WindowFrame1", img)
        if cv2.waitKey(1) == ord('q'):
            break
    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_recog()
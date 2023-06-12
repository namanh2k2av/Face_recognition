import cv2
import os
import time
import shutil
from process import load_dataset, get_embedding, save_data
from keras.models import load_model


def create_Data():
    video = cv2.VideoCapture(0)
    # facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    nameID = str(input("Nhap ten cua ban: "))
    path = 'data/origin/' + nameID
    isExist = os.path.exists(path)

    if isExist:
        print("Ten da ton tai!")
        nameID = str(input("Nhap lai ten cua ban: "))
    else:
        os.makedirs(path)
        
    while True:
        ret, frame = video.read()
        # faces = facedetect.detectMultiScale(frame, 1.3, 5)
        count += 1
        name = 'data/origin/' + nameID +'/'+ str(count) + '.jpg'
        print("Dang tao Image..." + name)
        cv2.imwrite(name, frame)
        # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.imshow("WindowFrame", frame)
        time.sleep(0.2)
        k = cv2.waitKey(1)
        if k == ord('q') or count == 30:
            print('Da tao xong!')
            break
    video.release()
    cv2.destroyAllWindows()
    
def train_classifier():
    train_x, train_y = load_dataset('data/origin/')
    save_data('data/data_face.npz', train_x, train_y)
    modelfn = load_model('model/facenet_keras.h5')
    newTrainX = list()
    for face_pixels in train_x:
        embedding = get_embedding(modelfn, face_pixels)
        newTrainX.append(embedding)
    save_data('data/data_embedding.npz', newTrainX, train_y)
    directory = 'data/origin/'
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    shutil.move(directory+subdirectories[0], 'data/processed/')
    print("Traning thành công!")

create_Data()
train_classifier()
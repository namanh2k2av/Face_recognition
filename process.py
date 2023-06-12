from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
import os
import numpy as np

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if(len(results) > 0):
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
    else:
        face_array = []
    return face_array

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

def extract(img, required_size=(160, 160)):
    pixels = asarray(img)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if(len(results) > 0):
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
    else:
        face_array = []
    return face_array

def load_faces(directory):
    faces = list()
    i = 0
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        if len(face) > 0:
            faces.append(face)
        i+=1
        print("Picture", i)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        print("Subd: ", subdir)
        print("path: ", path)
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def save_data(filename, train_x, train_y):
    if os.path.exists(filename):
        data = np.load(filename)
        x = np.concatenate([data['arr_0'], train_x])
        y = np.concatenate([data['arr_1'], train_y])
        savez_compressed(filename, x, y)
    else:
        savez_compressed(filename, train_x, train_y)

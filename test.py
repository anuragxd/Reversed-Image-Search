import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from sklearn.neighbors import NearestNeighbors
import cv2

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = Sequential([
    model,
    GlobalMaxPooling2D()
])


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

img_path = 'images/1163.jpg'
img = image.load_img(img_path,target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric="euclidean")
neighbors.fit(feature_list)

dist , ind = neighbors.kneighbors([normalized_result])

print(ind)



for file in ind[1:6]:
    temp = cv2.imread(filenames[file][17:])
    cv2.imshow('output',cv2.resize(temp,(512,512)))

    cv2.waitKey(0)

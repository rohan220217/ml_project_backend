import tensorflow as tf
import numpy as np
import pandas as pd
from cv2 import cv2 

export_path='./cnn_eth'
loaded_model_gender=tf.keras.models.load_model(export_path)
img = cv2.imread('./images/0.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape(48, 48, 1)
img = np.array([img])

y_pred = loaded_model_gender.predict_classes(img)
print(y_pred)

# Image download

# age_gender_data = pd.read_csv("./age_gender.csv")
# age_gender_data.info()

# image_pixel = age_gender_data.pixels
# img_width = 48
# img_height = 48
# def str_to_npArr(x):
#     '''
#     Function to convert pixel data (string) into numpy_array of pixels
#     '''
#     x = x.reset_index(drop=True)
#     x = x.apply(lambda x: np.array(x.split(), dtype="float32")) #converting data to numpy array
#     return np.array([x[i].reshape(img_width, img_height,1) for i in range(x.shape[0])])

# Converting the string of pixels into image array for each of train, val and test set and normalization
# image_pixel = str_to_npArr(image_pixel)
# for n in range(10):
#     cv2.imwrite('./images/'+str(n) + '.png',image_pixel[n].reshape(48,48))

# print("Current: shape = {}, type = {}".format(image_pixel.shape, type(image_pixel)))


exit()
from flask import Flask
from flask_restful import Api, Resource, abort, reqparse
from flask_cors import CORS
import json
import freesasa
import werkzeug
from hashlib import md5
from time import localtime
import requests

app = Flask(__name__)
CORS(app)
api = Api(app)


#Handler for protein
class Protein(Resource):
	def get(self):
		protein = {}
		with open('./data/code_mapping.json', 'r') as protein_file:
			protein = json.load(protein_file)

		return protein


api.add_resource(Asaview, '/asaview')
api.add_resource(Protein, '/protein')

if _name_ == '__main__':
	app.run(debug=True)
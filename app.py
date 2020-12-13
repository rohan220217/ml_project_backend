import tensorflow as tf
import numpy as np
import pandas as pd
from cv2 import cv2 
import os


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


from flask import Flask, jsonify
from flask_ngrok import run_with_ngrok
from flask_restful import Api, Resource, abort, reqparse
from flask_cors import CORS
import json
import werkzeug
from hashlib import md5
from time import localtime
import requests

#Ethnicity Model
export_path='./cnn_eth'
loaded_model_eth=tf.keras.models.load_model(export_path)

#Gender Model
export_path='./cnn_gender'
loaded_model_gender=tf.keras.models.load_model(export_path)

#Age Model
export_path='./reg_age'
loaded_model_age=tf.keras.models.load_model(export_path)

#Novelty Model
export_path = './novel_40_non_aug'
loaded_model_novelty=tf.keras.models.load_model(export_path)
app = Flask(__name__, static_folder="images")
run_with_ngrok(app)
CORS(app)
api = Api(app)


class Filesa(Resource):
	def get(self):
		imgs = []
		for filename in os.listdir("images"):
			file_arr = filename.split("_")
			imgs.append({
				'index' : file_arr[0],
				'age' : file_arr[1],
				'gender' : file_arr[2],
				'ethnicity' : file_arr[3],
				'filename' : 'images/'+filename
				})
		return jsonify(imgs)

#Handler for protein
class Protein(Resource):

	def get(self, img_id):
		# img = cv2.imread('./images/0.png')
		img = cv2.imread('./images/'+img_id)
		print(img_id)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = img/255
		img = img.reshape(48, 48, 1)
		img = np.array([img])

		gender_pred = loaded_model_gender.predict(img)
		gender_pred = gender_pred[0][0]
		y_pred_eth = np.argmax(loaded_model_eth.predict(img), axis=-1)
		y_pred_age = loaded_model_age.predict(img)
		y_pred_novelty = loaded_model_novelty.predict(img)
		output_cnn = {
			'gender' : int(gender_pred >= 0.5),
			'eth' : int(y_pred_eth[0]),
			'age' : float(y_pred_age[0][0])
			}

		output_novelty = {
			'gender' : int(y_pred_novelty[0][0][0]),
			'eth' : int(np.argmax(y_pred_novelty[1])),
			'age' : float(y_pred_novelty[2][0][0])
			}
		
		return jsonify({ 'cnn' : output_cnn, 'cnn_novelty' : output_novelty})


api.add_resource(Protein, '/predict/<string:img_id>')
api.add_resource(Filesa, '/files')

if __name__ == '__main__':
	app.run()
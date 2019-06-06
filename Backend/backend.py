'''
nocode.ai - backend
-------------------
Bringing the power of scikit-learn, tensorflow and lime to the common man.

A GUI-based framework for creating, deploying and understanding machine and deep learning algorithms.
'''

# Standard libraries
import os
import json
import pickle
import sys
import importlib
import csv
import weakref
from os import remove
from pprint import pprint

# Libraries to handle data
import numpy
import pickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# Libraries for deep learning
import keras
import tensorflow

# Web Server libraries
from flask import Flask
from flask import request
from flask_cors import CORS
from flask import Response
from flask import send_file
from werkzeug.utils import secure_filename

# Libraries for device information
import GPUtil
import psutil
from tensorflow.python.client import device_lib


# Metric libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Interpretability library
import lime.lime_tabular as lt

# Libraries to generate pdfs
import pdfkit
from PyPDF2 import PdfFileMerger

# Custom functions and classes
from data_extraction import CSV, RawFile
from globals import ABSOLUTE_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, allowed_file, PREPROC, SUPER, UNSUPER, json_decoder, json_encoder, weakdict, str_isfloat
from notebook_handler import ACTIVE_NOTEBOOKS, create_notebook_global_table, get_notebook_data, notebook_global_table_exist, set_notebook_data
from usertable_handler import create_new_user_table, user_table_exists
from keras_callbacks import on_epoch_end_callback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# To interact with vue.js
CORS(app)

'''
Route functions used by login page
- Form that accepts username and password
'''

# a username is checked initially before submitting the form
# once form is submitted that username is guaranteed to exist, need to check if password matches

@app.route("/check_username_exists/<check_username_exists_json>", methods = ["GET"])
def check_username_exists(check_username_exists_json):
	try:
		check_username_exists_dict = json_decoder.decode(check_username_exists_json)
		
		# if new installation, create new user table

		if not (user_table_exists()):
			create_new_user_table()

		# check user table is username exists

		fileObject = open("USERTABLE", "rb")
		table = pickle.load(fileObject)

		if any(obj['username'] == check_username_exists_dict['username'] for obj in table):
			return json_encoder.encode({"message":"Success", "comment":"Username exists"})

		return json_encoder.encode({"message":"Success", "comment": "Username available"})

	except:
		return json_encoder.encode({"message":"Failure", "comment": "Other error"})

@app.route("/check_username_and_password_matches", methods = ["POST"])
def check_username_and_password_matches():
	try:

		# if new installation, create a new user table
		if not (user_table_exists()):
			create_new_user_table()

		# extract from POST request form data wrapped in json

		username = request.json['username']
		password = request.json['password']

		# open user table and check if username and password match.

		fileObject = open("USERTABLE", "rb")
		table = pickle.load(fileObject)

		if (any(username == obj['username'] and password == obj['password'] for obj in table)):
			return json_encoder.encode({"message":"Success", "comment":"Username and password match"})

		return json_encoder.encode({"message":"Success", "comment":"Username and password does not match"})
	except:
		return json_encoder.encode({"message":"Failure", "comment": "Other error"})

'''
Route functions used by register page
- Form to create a new user
'''

@app.route("/add_user", methods = ["POST"])
def add_user():
	try:
		user = request.json
		
		# if new installation, create a new user table

		if not (user_table_exists()):
			create_new_user_table()

		fileObject = open("USERTABLE", "rb")
		table = pickle.load(fileObject)
		fileObject.close()

		# newly created user does not have any notebooks

		obj = 	{
					"username": user['username'],
					"password": user['password'],
					"created_notebooks": [],
					"shared_notebooks": []
				}
		
		table.append(obj)

		# save updated user table

		fileObject = open("USERTABLE", "wb")
		pickle.dump(table, fileObject)
		fileObject.close()

		return json_encoder.encode({"message":"success"})
	except:
		return json_encoder.encode({"message":"failure"})

'''
Route functions used by dashboard
- Creates new notebooks
- Can display and load existing notebooks
'''

@app.route("/check_notebook_name_exists/<check_notebook_name_exists_json>", methods = ["GET"])
def check_notebook_name_exists(check_notebook_name_exists_json):
	check_notebook_name_exists_dict = json_decoder.decode(check_notebook_name_exists_json)

	# if new installation, create global notebooks table

	if not (notebook_global_table_exist()):
		create_notebook_global_table()

	# opens the global notebooks table, and checks whether notebook exists

	fileObject = open("NOTEBOOKS_DATA", "rb")
	table = pickle.load(fileObject)

	if any(obj['notebook_name'] == check_notebook_name_exists_dict['notebook_name'] for obj in table):
		return json_encoder.encode({"message":"Success", "comment":"Notebook name exists"})

	return json_encoder.encode({"message":"Success", "comment": "Notebook name available"})

@app.route("/get_user_notebooks/<get_user_notebooks_json>", methods = ["GET"])
def get_user_notebooks(get_user_notebooks_json):
	get_user_notebooks_dict = json_decoder.decode(get_user_notebooks_json)

	# opens user table and returns list of notebooks opened by that user

	table = pickle.load(open("USERTABLE", "rb"))

	for obj in table:
		if (obj['username'] == get_user_notebooks_dict['username']):
			return json_encoder.encode({"message":"Success", "notebook_names":obj['shared_notebooks'] + obj['created_notebooks']})

	return json_encoder.encode({"message":"Failure"})

@app.route("/add_notebook", methods = ["POST"])
def add_notebook():
	print("changed")
	try:
		# notebook is of type weakdict to create references of it
		notebook = weakdict(request.json)

		# creating a reference of notebook's data
		weakdict_notebook = weakref.proxy(notebook)

		fileObject = open("NOTEBOOK_"+notebook['notebook_name'], "wb")
		pickle.dump(notebook, fileObject)
		fileObject.close()

		# if new installation, create a global notebooks table

		if not (notebook_global_table_exist()):
			create_notebook_global_table()

		# open global notebooks table and add notebook configuration

		fileObject = open("NOTEBOOKS_DATA", "rb")
		table = pickle.load(fileObject)
		fileObject.close()

		table.append({
						"notebook_name": notebook['notebook_name'],
						"GPU_count": int(notebook['GPU_count']),
						"CPU_count": int(notebook['CPU_count']),
						"is_online": False
					})

		fileObject = open("NOTEBOOKS_DATA", "wb")
		pickle.dump(table, fileObject)
		fileObject.close()

		# open global user table and add this notebook to user's list of created notebooks

		table = pickle.load(open("USERTABLE", "rb"))
		for obj in table:
			if obj['username'] == notebook['username']:
				obj['created_notebooks'].append(notebook['notebook_name'])

		pickle.dump(table, open("USERTABLE", "wb"))

		return json_encoder.encode({"message":"Success", "comment": "Notebook created"})
	except:
		return json_encoder.encode({"message":"Failure", "comment": "Other error"})

@app.route("/get_devices/", methods = ["GET"])
def get_devices():
	# Currently works only on Linux
	# Fails on windows.

	# Check for unix
	if(os.name == 'posix'):
		
		# get GPU count if avaiable
		try:
			n_gpu = len(GPUtil.getGPUs())
		except:
			n_gpu = 0

		# get CPU count
		n_cpu = psutil.cpu_count()

	# For windows, use tensorflow to get details
	else:
		local_devices = device_lib.list_local_devices()
		n_gpu = len([x.name for x in local_devices if x.device_type == 'GPU'])
		n_cpu = psutil.cpu_count()

	# if new installation, create a global notebooks table

	if not (notebook_global_table_exist()):
		create_notebook_global_table()

	table = pickle.load(open("NOTEBOOKS_DATA", "rb"))

	# Send number of CPUs and GPUs currently available

	a_gpu = n_gpu - sum(obj["GPU_count"] for obj in table if obj['is_online'])
	a_cpu = n_cpu - sum(obj["CPU_count"] for obj in table if obj['is_online'])

	return json_encoder.encode({"message":"Success", "GPU_count": n_gpu, "CPU_count":n_cpu, "GPU_available": a_gpu, "CPU_available": a_cpu})

@app.route("/load_existing_notebook", methods = ["POST"])
def load_existing_notebook():
	# Send data to UI to load back existing notebook

	data = request.json
	notebook = get_notebook_data(data['notebook_name'])

	dct = {}
	for key in notebook:
		# need not send model and other array type of data
		if key not in {'model', 'x_raw', 'x_preprocessed', 'x_test', 'x_train', 'y_test', 'y_train', '_model','y_raw','confusion_matrix'}:
			dct[key] = notebook[key]

	return json_encoder.encode({"message": "Success", "notebook_data": dct})

'''
Route functions used by upload-data
- Can upload table in the form of CSV
- Can upload raw files in the form of npz and pkl
- Can use pre-existing datasets
'''

@app.route("/upload_table", methods = ["POST"])
def upload_table():
	data = request.form
	
	notebook = get_notebook_data(data['notebook_name'])

	# Check if file has been uploaded succesfully
	if 'file' not in request.files:
		return json_encoder.encode({"message":"Failure", "comment":"No file received"})
	
	file = request.files['file']
	
	# Check if there is a file
	if file.filename == '':
		return json_encoder.encode({"message":"Failure", "comment":"No file selected"})

	if(request.form['load_notebook_status'] == 'false'):

		# Check for file types
		if file and allowed_file(file.filename):

			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			uploaded_file = open(os.path.join(app.config['UPLOAD_FOLDER'], filename),"r")
			reader = csv.reader(uploaded_file)
			total_cols = len(next(reader))
			features_cols = []
			label_cols = []

			for col_num in range(0,total_cols-1):
				features_cols.append(col_num)

			label_cols.append(total_cols-1)

			# extract contents of the file to give X and Y data
			csvObject = CSV(os.path.join(app.config['UPLOAD_FOLDER'], filename), {'features':features_cols, 'labels':label_cols}, False)
			X, Y = csvObject.extract()

		# store extracted data into notebook
		notebook['x_raw'] = X
		notebook['y_raw'] = Y
		notebook['file_name'] = data['file_name']

		set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message":"Success", "comment":"Table loaded successfully"})

@app.route("/upload_raw", methods = ["POST"])
def upload_raw():

	# Check if file has been uploaded succesfully
	if 'file' not in request.files:
		return json_encoder.encode({"message":"Failure", "comment":"No file received"})

	file = request.files['file']

	# Check if there is a file
	if file.filename == '':
		return json_encoder.encode({"message":"Failure", "comment":"No file selected"})
		
	# Check for file types
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		
		# extract raw binary data
		rawObject = RawFile(os.path.join(app.config['UPLOAD_FOLDER'], filename), {'features':[0], 'labels':[1]}, False)
		X, Y = rawObject.extract()

		data = request.json
		notebook = get_notebook_data(data['notebook_name'])

		# store dataset in notebook
		notebook['x_raw'] = X
		notebook['y_raw'] = Y

		set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message":"Success", "comment":"Data loaded successfully"})

@app.route("/upload_predefined", methods = ["POST"])
def upload_predefined():

	# Use a pre existing dataset
		# Boston housing
		# CIFAR10
		# CIFAR100
		# Iris
		# Oxford17 flowers
		# MNIST

	data = request.json
	notebook = get_notebook_data(data['notebook_name'])

	# load dataset from datasets folder into notebook

	_x = numpy.load(open("datasets/"+data['dataset_name']+"/X", "rb"))

	notebook['x_raw'] = numpy.reshape(_x, newshape = (-1, numpy.prod(_x.shape[1:])))
	notebook['y_raw'] = numpy.load(open("datasets/"+data['dataset_name']+"/Y", "rb"))

	set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message":"Success", "comment":"Data loaded successfully"})

'''
@app.route("/preprocessing", methods = ["POST"])
def preprocessing():

	data = request.json

	notebook = get_notebook_data(data['notebook_name'])

	# function that preprocesses data using sklearn.preprocessing functions.
	def _preprocess(my_json, X):
		
		# import the required sklearn module
		module = importlib.import_module('sklearn.'+my_json['module'])
		_class = getattr(module, my_json['class'])
		
		# try to preprocess
		try:
			_X = _class(**my_json['hyperparameters']).fit_transform(X)
		except:
			_X = _class(**my_json['hyperparameters']).fit(X)
		
		return _X

	# get raw data from notebook
	X = notebook['x_raw'] if 'x_preprocessed' not in notebook else notebook['x_preprocessed']

	if (data['model_parameters']['module'] in PREPROC):
		_X = _preprocess(data['model_parameters'], X)

		# add the preprocessed data to the notebook
		notebook['x_preprocessed'] = _X
		notebook['preprocessing_applied'] = data['model_parameters']['class']
		notebook['has_columns'] = data['has_columns']
		notebook['uploaded_file_type'] = data['uploaded_file_type']

	set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message": "Success", "comment": "Preprocessor applied"})
'''

@app.route("/preprocessing", methods = ["POST"])
def preprocessing():

	data = request.json

	notebook = get_notebook_data(data['notebook_name'])

	def _preprocess(my_json, X):
		module = importlib.import_module('sklearn.'+my_json['module'])
		_class = getattr(module, my_json['class'])
		try:
			_X = _class(**my_json['hyperparameters']).fit_transform(X)
		except:
			_X = _class(**my_json['hyperparameters']).fit(X)
		return _X

	X = notebook['x_raw'] if 'x_preprocessed' not in notebook else notebook['x_preprocessed']

	if (data['model_parameters']['module'] in PREPROC):
		_X = _preprocess(data['model_parameters'], X)
		notebook['x_preprocessed'] = _X
		notebook['preprocessing_applied'] = data['model_parameters']['class']
		notebook['has_columns'] = data['has_columns']
		notebook['uploaded_file_type'] = data['uploaded_file_type']

	set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message": "Success", "comment": "Preprocessor applied"})



@app.route("/set_train_test_data", methods = ["POST"])
def set_train_test_data():

	data = request.json
	notebook = get_notebook_data(data['notebook_name'])

	# used preprocessed data if it exists, or else raw data
	X = notebook['x_raw'] if 'x_preprocessed' not in notebook else notebook['x_preprocessed']

	notebook['hyperparameters'] = {}
	notebook['hyperparameters']['test_size'] = data['test_size']

	# splits dataset into 4 parts and stores it in the notebook
	notebook['x_train'], notebook['x_test'], notebook['y_train'], notebook['y_test'] = train_test_split(X, notebook['y_raw'], test_size = data['test_size'])

	set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message": "Success", "comment": "Data set"})

'''
Route functions used by build-model
- Create neural network model
- Select machine learning algorithm
'''

@app.route("/create_sequential_model", methods = ["POST"])
def create_sequential_model():
	data = request.json

	notebook = get_notebook_data(data['notebook_name'])

	notebook['model_type'] = "NEURAL NETWORK"

	# boolean for server sent events notifier
	notebook['_epoch_done'] = False

	notebook['numLayers'] = data['numLayers']
	layers = data['layers']
	notebook['modelLayers'] = layers

	# get input shape for the first layer
	input_shape = notebook['x_preprocessed'].shape[1:] if 'x_preprocessed' in notebook else notebook['x_raw'].shape[1:] 

	# using keras sequential API
	model = keras.Sequential()

	# use eval to evaluate strings received by client (in JSON format) to create layers
	# add these layers to the model

	# special case to include input shape
	model.add(eval("keras.layers."+layers[1]['layerType']+"("+ ",".join([str(dct["name"])+"="+(str(dct["defaultValue"]) if str_isfloat(dct["defaultValue"]) else "'"+dct["defaultValue"]+"'") for dct in layers[1]['defaultOptions'] if dct["defaultValue"] not in {True, False, None}] + ["input_shape="+str(input_shape)]) +")"))
	
	for layer in layers[2:]:
		model.add(eval("keras.layers."+layer['layerType']+"("+ ",".join([str(dct["name"])+"="+(str(dct["defaultValue"]) if str_isfloat(dct["defaultValue"]) else "'"+dct["defaultValue"]+"'") for dct in layer['defaultOptions'] if dct["defaultValue"] not in {True, False, None}]) +")"))

	model.summary()

	# store model in json format
	notebook['model'] = model.to_json()
	set_notebook_data(notebook['notebook_name'])
	
	# clear the graph to avoid errors
	try:
		keras.backend.clear_session()
	except:
		pass

	return json_encoder.encode({"message": "Success", "comment": "Model Created!"})

'''
Route functions used by train-model
- Specify hyperparameters
- Train learning models
- Send events to display accuracy and loss history curves
'''

@app.route("/create_non_neural_network_model", methods = ["POST"])
def create_non_neural_network_model():

	data = request.json

	notebook = get_notebook_data(data['notebook_name'])

	notebook['hyperparameters'] = data['model_parameters']
	notebook['model_type'] = "NON NEURAL NETWORK"
	notebook['model_name'] = data['model_parameters']['class']

	# change the data types of parameters to required data types
	for key in data['model_parameters']['hyperparameters']:
		try:
			if(data['model_parameters']['hyperparameters'][key]=='None'):
				data['model_parameters']['hyperparameters'][key]=None
			elif(data['model_parameters']['hyperparameters'][key].isdigit()):
				data['model_parameters']['hyperparameters'][key] = int(data['model_parameters']['hyperparameters'][key])
			elif(str_isfloat(data['model_parameters']['hyperparameters'][key])):
				data['model_parameters']['hyperparameters'][key] = float(data['model_parameters']['hyperparameters'][key])
			elif(data['model_parameters']['hyperparameters'][key]=='True'):
				data['model_parameters']['hyperparameters'][key] = True
			elif(data['model_parameters']['hyperparameters'][key]=='False'):
				data['model_parameters']['hyperparameters'][key] = False
		except:
			pass

	# allocate CPUs if possible
	if 'n_jobs' in 	data['model_parameters']['hyperparameters']:
		data['model_parameters']['hyperparameters']['n_jobs'] = notebook['CPU_count']

	notebook['is_online'] = True

	def train_supervised(x_train, y_train, my_json):

		# import just the required class from specific module and train the model

		module = importlib.import_module('sklearn.'+my_json['module'])
		_class = getattr(module, my_json['class'])
		model = _class(**my_json['hyperparameters'])
		model.fit(x_train, y_train)
	
		# deallocate devices after training

		notebook['is_online'] = False

		return model

	def train_unsupervised(X, my_json):

		# import just the required class from specific module and train the model

		module = importlib.import_module('sklearn.'+my_json['module'])
		_class = getattr(module, my_json['class'])
		model = _class(**my_json['hyperparameters'])
		
		try:
			model.fit_transform(X)
		except:
			model.fit(X)
		
		# deallocate devices after training

		notebook['is_online'] = False

		return model 

	# train supervised and unsupervised algorithms separately

	# supervised algorithms require 1-D array of Y training samples which contain class labels
	if data['model_parameters']['module'] in SUPER:
		print("\n", numpy.array(list(map(numpy.argmax, notebook['y_train']))), "\n")
		y_train = notebook['y_train'] if len(notebook['y_train'].shape) <= 2 else numpy.array(list(map(numpy.argmax, notebook['y_train']))) 
		notebook['model'] = train_supervised(notebook['x_train'], y_train, data['model_parameters'])
	
	# unsupervised algorithms require on X training samples
	elif data['model_parameters']['module'] in UNSUPER:
		notebook['model'] = train_supervised(notebook['x_train'], data['model_parameters'])

	set_notebook_data(data['notebook_name'])

	return json_encoder.encode({"message": "Success", "comment": "Model trained"})

@app.route("/compile_sequential_model", methods = ["POST"])
def compile_sequential_model():
	# Compiles and Trains neural network	

	data = request.json

	notebook = get_notebook_data(data['notebook_name'])
	notebook['hyperparameters'] = data['hyperparameters']
	notebook["history"] = 	{
								"acc": [],
								"val_acc": [],
								"loss": [],
								"val_loss": []
							}

	# allocate specified device while creating notebook
							
	config = tensorflow.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = (notebook["GPU_count"] / len(GPUtil.getAvailable()))
	keras.backend.tensorflow_backend.set_session(tensorflow.Session(config = config))

	notebook['is_online'] = True

	# load created model
	model = keras.models.model_from_json(notebook['model'])
	
	# compile with client-sent hyperparamters
	model.compile(loss = data['hyperparameters']['loss'], optimizer = keras.optimizers.SGD(lr = float(data['hyperparameters']['learning_rate']), momentum = float(data['hyperparameters']['momentum']), nesterov = bool(data['hyperparameters']['nesterov'])), metrics = ['acc'])
	
	# Training starts
	model.fit(x = notebook['x_train'], y = notebook['y_train'], batch_size = 128, validation_data = (notebook['x_test'], notebook['y_test']), epochs = int(data['hyperparameters']['epochs']),  callbacks = [on_epoch_end_callback(notebook = notebook)])

	# save model separately as model weights could not be pickled	
	model.save("NOTEBOOK_" + data['notebook_name'] + "_neural_network_model.hdf5")

	notebook['model'] = model.to_json()
	set_notebook_data(data['notebook_name'])
	
	try:
		keras.backend.clear_session()
	except:
		pass

	return json_encoder.encode({"message": "Success", "comment": "Compiled model and trained"})

@app.route("/get_epoch_details/<notebook_name_json>", methods = ["GET"])
def get_epoch_details(notebook_name_json):
	
	notebook_name_dict = json_decoder.decode(notebook_name_json)
	notebook = get_notebook_data(notebook_name_dict['notebook_name'])

	# Yield the events of epoch ending
	def epoch_done_streamer():
		
		# Busy wait - can be improved
		while True:
			if notebook['_epoch_done']:
				
				# reset the epoch done notifier; continue to wait busily
				notebook['_epoch_done'] = False

				# send just a signal to reload the image created, therefore data = 0
				yield 'event: EPOCH_END\ndata: 0\n\n'
	
	# return Server sent events	
	return Response(epoch_done_streamer(), mimetype="text/event-stream")

'''
Route functions used results
- Displays accuracy, precision, recall, true positive, true negative, false positive, false negative
- Plots ROC and Precision-Recall Curves
'''

@app.route("/get_roc_curve/<notebook_name_json>", methods = ["GET"])
def get_roc_curve(notebook_name_json):
	notebook_name_dict = json_decoder.decode(notebook_name_json)

	notebook = get_notebook_data(notebook_name_dict['notebook_name'])

	# does predict on testing data

	# to invoke keras predict
	if notebook['model_type'] == "NEURAL NETWORK":
		model = keras.models.load_model("NOTEBOOK_" + notebook_name_dict['notebook_name'] + "_neural_network_model.hdf5")
		probs = model.predict(notebook['x_test'])
	
	# to invoke sklearn predict
	else:	
		probs = notebook['model'].predict_proba(notebook['x_test'])

	preds = probs[:, 1]
	fpr, tpr, threshold = roc_curve(notebook['y_test'], preds)
	roc_auc = auc(fpr, tpr)

	# Create ROC plot
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')

	# Save plot to be used by vue.js
	filename = "NOTEBOOK_" + notebook['notebook_name'] + "_roc_curve.jpg"
	plt.savefig("../UI/src/assets/" + filename)

	plt.clf()

	# Save file name in notebook
	notebook['roc_curve'] = filename

	set_notebook_data(notebook_name_dict['notebook_name'])

	try:
		keras.backend.clear_session()
	except:
		pass

	return json_encoder.encode({"message": "Success", "roc_curve": filename})

@app.route("/get_accuracy/<notebook_name_json>", methods = ["GET"])
def get_accuracy(notebook_name_json):
	notebook_name_dict = json_decoder.decode(notebook_name_json)

	notebook = get_notebook_data(notebook_name_dict['notebook_name'])

	# to invoke keras predict
	if notebook['model_type'] == "NEURAL NETWORK":
		model = keras.models.load_model("NOTEBOOK_" + notebook_name_dict['notebook_name'] + "_neural_network_model.hdf5")
		prediction = model.predict(notebook['x_test'])
		prediction = numpy.array(list(map(numpy.argmax, prediction)))
	
	# to invoke sklearn predict
	else:
		model = notebook['model']
		prediction = model.predict(notebook['x_test'])

	# de-"one hot"
	y_test = notebook['y_test']
	if len(y_test.shape) > 1:
		y_test = numpy.array(list(map(numpy.argmax, y_test)))
	
	accuracy = accuracy_score(notebook['y_test'], prediction)

	# save accuracy in notebook
	notebook['accuracy'] = accuracy

	set_notebook_data(notebook_name_dict['notebook_name'])
	
	try:
		keras.backend.clear_session()
	except:
		pass

	return json_encoder.encode({"message": "Success", "accuracy": str(accuracy)})

@app.route("/get_confusion_matrix/<notebook_name_json>", methods = ["GET"])
def get_confusion_matrix(notebook_name_json):
	notebook_name_dict = json_decoder.decode(notebook_name_json)

	notebook = get_notebook_data(notebook_name_dict['notebook_name'])

	# to invoke keras predict
	if notebook['model_type'] == "NEURAL NETWORK":
		model = keras.models.load_model("NOTEBOOK_" + notebook_name_dict['notebook_name'] + "_neural_network_model.hdf5")
		prediction = model.predict(notebook['x_test'])
		prediction = numpy.array(list(map(numpy.argmax, prediction)))
	
	# to invoke sklearn predict
	else:
		model = notebook['model']
		prediction = model.predict(notebook['x_test'])

	# de-"one hot"
	y_test = notebook['y_test']
	if len(y_test.shape) > 1:
		y_test = numpy.array(list(map(numpy.argmax, y_test)))

	matrix = confusion_matrix(notebook['y_test'], prediction).ravel()

	# save confusion matrix in notebook
	# did not consider for multiclass labels while displaying
	notebook['confusion_matrix'] = matrix
	notebook['true_negative'] = int(matrix[0])
	notebook['false_positive'] = int(matrix[1])
	notebook['false_negative'] = int(matrix[2])
	notebook['true_positive'] = int(matrix[3])

	set_notebook_data(notebook_name_dict['notebook_name'])
	
	try:
		keras.backend.clear_session()
	except:
		pass

	return json_encoder.encode({"message": "Success", "confusion_matrix": matrix.tolist()})

@app.route("/get_precision_recall_curve/<notebook_name_json>", methods = ["GET"])
def get_precision_recall_curve(notebook_name_json):
	notebook_name_dict = json_decoder.decode(notebook_name_json)

	notebook = get_notebook_data(notebook_name_dict['notebook_name'])


	# to invoke sklearn predict
	if notebook['model_type'] == "NEURAL NETWORK":
		model = keras.models.load_model("NOTEBOOK_" + notebook_name_dict['notebook_name'] + "_neural_network_model.hdf5")
		prediction = model.predict(notebook['x_test'])
		prediction = numpy.array(list(map(numpy.argmax, prediction)))
	
	# to invoke sklearn predict
	else:
		model = notebook['model']
		prediction = model.predict(notebook['x_test'])

	y_score = prediction

	# de-"one hot"
	y_test = notebook['y_test']
	if len(y_test.shape) > 1:
		y_test = numpy.array(list(map(numpy.argmax, y_test)))

	# Average precision curve for only binary class problems
	# can iterate for multi class
	y_test[y_test == y_test.min()] = 0
	y_test[y_test != 0] = 1

	average_precision = average_precision_score(y_test, y_score)

	notebook['average_precision_score'] = average_precision
	precision, recall, _ = precision_recall_curve(y_test, y_score)

	# create precision recall curve
	plt.step(recall, precision, color='b', alpha = 0.2, where = 'post')
	plt.fill_between(recall, precision, step = 'post', alpha = 0.2,color = 'b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

	# save data and filename in the notebook for reloading
	filename = "NOTEBOOK_" + notebook['notebook_name'] + "_precision_recall_curve.jpg"
	plt.savefig("../UI/src/assets/" + filename)
	plt.clf()
	notebook['precision_recall_curve'] = filename
	notebook['average_precision_score'] = average_precision
	notebook['recall'] = recall.mean()

	set_notebook_data(notebook_name_dict['notebook_name'])

	try:
		keras.backend.clear_session()
	except:
		pass

	return json_encoder.encode({"message": "Success", "precision_recall_curve": filename, "average_precision_score": average_precision, "recall": recall.mean()})


'''
Route functions used investigate-model
- Displays LIME generated plots
'''

@app.route("/investigate_model/<notebook_name_json>", methods = ["GET"])
def investigate_model(notebook_name_json):

	notebook_name_dict = json_decoder.decode(notebook_name_json)
	notebook = get_notebook_data(notebook_name_dict['notebook_name'])

	def predict_fn(instance):
		if notebook['model_type'] == "NEURAL NETWORK":
			return keras.models.model_from_json(notebook['model']).predict([instance])
		try:
			return notebook['model'].predict_proba(instance)
		except:
			return notebook['model'].predict(instance)

	def explain_instance_image_data(instance):

		newshape = numpy.prod(instance.shape)

		if notebook['model_type'] == "NEURAL NETWORK":
			model = keras.models.load_model("NOTEBOOK_" + notebook_name_dict['notebook_name'] + "_neural_network_model.hdf5")
			target = list(map(numpy.argmax, model.predict(numpy.reshape(instance, newshape = (1, *instance.shape)))[0]))[0]
		else:
			target = notebook['model'].predict(numpy.reshape(instance, newshape = newshape))

		plt.savefig("../UI/src/assets/" + "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance0.jpg")
		plt.clf()

		explainer = lt.LimeTabularExplainer(training_data=notebook['x_train'], feature_names=[str(i) for i in range(len(instance))] if 'column_names' not in notebook else notebook['column_names'])
		exp = explainer.explain_instance(instance, predict_fn, num_features=len(instance), num_samples=newshape, labels=(target,))
		exp.as_pyplot_figure(label=target).savefig("../UI/src/assets/" + "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance1.jpg", figsize=(50, 50))
		exp.save_to_file(file_path="../UI/src/assets/" + "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance.html")
		
		explaination_list = exp.as_map()[target]
		constructed_image = numpy.zeros(shape=len(explaination_list))

		for i, j in explaination_list:
			constructed_image[i] = 0.5 + j/2

		constructed_image = numpy.reshape(constructed_image, newshape = instance.shape)
		plt.savefig("../UI/src/assets/" + "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance2.jpg")
		plt.clf()
		notebook['explanation'] = "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance.html"
		set_notebook_data(notebook_name_dict['notebook_name'])
		
		try:
			keras.backend.clear_session()
		except:
			pass

		return json_encoder.encode({'instance': "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance0.jpg", 'explanation': "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance.html", 'constructed': "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance2.jpg"})

	def explain_instance_tabular_data(instance):
		newshape = numpy.prod(instance.shape)
		
		if notebook['model_type'] == "NEURAL NETWORK":
			model = keras.models.load_model("NOTEBOOK_" + notebook_name_dict['notebook_name'] + "_neural_network_model.hdf5")
			target = list(map(numpy.argmax, model.predict(numpy.reshape(instance, newshape = (1, *instance.shape)))[0]))[0]
		else:
			target = notebook['model'].predict([instance])[0]


		explainer = lt.LimeTabularExplainer(training_data=notebook['x_train'], feature_names=[str(i) for i in range(len(instance))])
		exp = explainer.explain_instance(instance, predict_fn, num_features=len(instance), num_samples=min(len(notebook['x_train']), 100), labels=(target,))
		exp.as_pyplot_figure(label=target).savefig("../UI/src/assets/" + "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance1.jpg", figsize=(50, 50))
		exp.save_to_file(file_path="../UI/src/assets/" + "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance.html")
		notebook['explanation'] = "NOTEBOOK_"+notebook['notebook_name']+"_investigate_model_instance.html"
		
		set_notebook_data(notebook_name_dict['notebook_name'])
		
		try:
			keras.backend.clear_session()
		except:
			pass

		return json_encoder.encode({'explanation': "NOTEBOOK_" + notebook['notebook_name'] + "_investigate_model_instance.html"})


	is_image = len(notebook['x_test'][0].shape) >= 2

	if is_image :
		return explain_instance_image_data(notebook['x_train'][numpy.random.randint(0,len(notebook['x_test']-1))])
	else:
		return explain_instance_tabular_data(notebook['x_train'][numpy.random.randint(0,len(notebook['x_test']-1))])

'''
To create PDFs
'''

@app.route("/export_pdf/<notebook_name_json>", methods = ["GET"])
def export_pdf(notebook_name_json):
	notebook_name_dict = json_decoder.decode(notebook_name_json)

	url = ["/upload-data", "/build-model", "/train-model", "/results", "/investigate-model"]
	merger = PdfFileMerger()
	
	# go over the 5 pages of the notebook and save them as pdfs using google-chrome invoked through shell.
	# will not work on windows
	for i in range(5):
		os.system("google-chrome --headless --disable-gpu --print-to-pdf=\"" +notebook_name_dict['notebook_name'] + 'out' + str(i) + '.pdf'+ "\" http://localhost:8080/notebook/" + notebook_name_dict['notebook_name'] + url[i])
	
	# merge the pdfs while deleting them
	for i in range(5):
		merger.append(open(notebook_name_dict['notebook_name'] + 'out' + str(i) + '.pdf', 'rb'))
		remove(notebook_name_dict['notebook_name'] + 'out' + str(i) + '.pdf')

	# save the merged pdf
	with open(notebook_name_dict['notebook_name'] + '_report.pdf', 'wb') as fp:
		merger.write(fp)

	# return PDF type data, to open on the browser
	return Response(open(notebook_name_dict['notebook_name'] + '_report.pdf', 'rb').read(), mimetype="application/pdf")	

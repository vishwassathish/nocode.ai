import keras
import matplotlib
import matplotlib.pyplot as plt
from notebook_handler import get_notebook_data, set_notebook_data

matplotlib.use('Agg')

# Callback Class to plot accuracy and loss history during neural network training.

class on_epoch_end_callback(keras.callbacks.Callback):
	def __init__(self, notebook):
		self.notebook = notebook
		return

	def on_epoch_end(self, epoch, logs={}):
		
		# Add history data to notebook that can be used during retraining.

		self.notebook['history']['acc'] += [logs['acc']]
		self.notebook['history']['loss'] += [logs['loss']]
		self.notebook['history']['val_acc'] += [logs['val_acc']]
		self.notebook['history']['val_loss'] += [logs['val_loss']]
		
		# plot accuracy history curve

		plt.plot(self.notebook['history']['acc'])
		plt.plot(self.notebook['history']['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		
		# save the plot into a folder vue.js can access it

		filename = "NOTEBOOK_" + self.notebook['notebook_name'] + "_accuracy_history_curve.jpg"
		plt.savefig("../UI/src/assets/" + filename)
		plt.clf()

		# update the notebook so that to plot when notebook reloads

		self.notebook['accuracy_history_curve'] = filename

		# plot loss history curve		

		plt.plot(self.notebook['history']['loss'])
		plt.plot(self.notebook['history']['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')

		# save the plot into a folder vue.js can access it
		
		filename = "NOTEBOOK_" + self.notebook['notebook_name'] + "_loss_history_curve.jpg"
		plt.savefig("../UI/src/assets/" + filename)
		plt.clf()

		# update the notebook so that to plot when notebook reloads

		self.notebook['loss_history_curve'] = filename

		set_notebook_data(self.notebook['notebook_name'])

		# boolean to notify external function for server sent events

		self.notebook['_epoch_done'] = True

		return 
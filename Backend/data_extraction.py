import numpy as np
import pandas as pd

class CSV:

	def __init__(self, filename, column_json, has_columns):

		self.filename = filename
		self.column_json = column_json
		self.has_columns = has_columns

	def extract(self):

		# For excel files with headings

		if self.filename[-5:] == '.xlsx' and self.has_columns :
			df = pd.read_excel(self.filename)

		# For excel files without headings

		elif self.filename[-5:] == '.xlsx' :
			df = pd.read_excel(self.filename, header=None)

		# For tab separated values with headings
		
		elif self.filename[-4:] == '.tsv' and self.has_columns :
			df = pd.read_csv(self.filename, sep='\t')

		# For tab separated values without headings
		
		elif self.filename[-4:] == '.tsv' :
			df = pd.read_csv(self.filename, sep='\t', header=None)

		# For comma separated values with headings

		elif self.filename[-4:] == '.csv' and self.has_columns:
			df = pd.read_csv(self.filename)

		# For comma separated values without headings

		elif self.filename[-4:] == '.csv' :
			df = pd.read_csv(self.filename, header=None)

		# Catch errors in data format

		else:
			print("Error reading file")
			return None, None

		return self._split(df)

	def _split(self, df):

		X = np.array([np.array(df[col]) for col in self.column_json['features']])
		Y = np.array([np.array(df[col]) for col in self.column_json['labels']])
		return np.transpose(X), np.transpose(Y)

class RawFile:

	def __init__(self, filename):
		self.filename = filename

	def extract():

		# For numpy array files

		if self.filename[-4:] == '.npz':
			data = np.load(self.filename)
			return data[0], data[1]

		# For pickled files

		elif self.filename[-4:] == '.pickle':
			data = pickle.load(self.filename)
			return data[0], data[1]
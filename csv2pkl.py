# import string
import csv
import pickle
import numpy as np


def read_csv(filename):
	with open(filename, 'r') as file:
		reader = csv.reader(file)
		return [line for line in reader] 
	pass

def numerica(data, start = 0, iset = set()):
	dim1 = len(data)
	dim2 = len(data[0])
	print(dim1, dim2)
	for j in range(dim2):
		converter = float if j not in iset else int
		for i in range(start, dim1):
			data[i][j] = converter(data[i][j])
	return data;

if __name__ == '__main__':
	txt = read_csv('./datasets/simple_multilabel.csv')
	data = numerica(txt, iset = {2})
	with open('./datasets/simple_multilabel.pkl', 'wb') as file:
		pickle.dump(data, file)
	
	pass
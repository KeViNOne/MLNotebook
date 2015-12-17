# import string
import pickle
import numpy as np

if __name__ == '__main__':
	filename = './datasets/simple_binarylabel'
	csv_ext = '.csv'
	pkl_ext = '.pkl'
	
	data = np.loadtxt(filename + csv_ext, delimiter=',')
	m, n = data.shape
	
	data_x = np.array(data[:,:-1], copy=True)
	data_y = np.array(data[:,-1].reshape((m,1)), copy=True)
	
	with open('./datasets/simple_binarylabel.pkl', 'wb') as file:
		pickle.dump((data_x, data_y), file)
	
	pass
import os
import gzip
import pickle
import numpy as np


def load(data_filename, zip = False):
	print('loading data', end=' ')
	
	# 处理文件名
	data_dir, data_file = os.path.split(data_filename)
	if data_dir == "" and not os.path.isfile(data_filename):
		# 如果仅为文件名，且该文件不在当前目录中
		data_filename = os.path.join(
			os.path.split(__file__)[0],
			"..",
			"datasets",
			data_filename
		)
	print('-', data_filename, end=' ')
	
	# 加载数据文件
	o = gzip.open if(zip)else open
	with o(data_filename, 'rb') as f:
		data = pickle.load(f, encoding='bytes')
		print('- success')
		return data
		
	print('- failed')
	pass

def pick(data, m, random = True):
	is_tuple = isinstance(data, tuple)
	if(is_tuple):
		if(random):
			rows = np.random.choice(data[0].shape[0], m, replace=False)
			return tuple(matrix[rows] for matrix in data)
		else:
			return tuple(matrix[:m] for matrix in data)
	else:
		if(random):
			rows = np.random.choice(data.shape[0], m, replace=False)
			return data[rows]
		else:
			return data[:m]
	pass

def split(data, s):
	is_tuple = isinstance(data, tuple)
	if isinstance(s, float):
		rows = data[0].shape[0] if(is_tuple)else data.shape[0]
		s = int(rows * s)
	if(is_tuple):
		ret = list()
		for matrix in data:
			ret.append(matrix[:s])
			ret.append(matrix[s:])
		return tuple(ret)
	else:
		return data[:s], data[s:]
	
	pass

import os
import pickle
import numpy as np


def load_data_array(data_filename):
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
	with open(data_filename, 'rb') as f:
		data = pickle.load(f, encoding='bytes')
		print('- success')
		return data
		
	print('- failed')
	pass
import os
import timeit

import pickle

import numpy as np
import theano as tn



class LinearRegression(object):

	"""线性回归

	"""

	def __init__(self, n_in, n_out):
		"""Initialize the parameters of the logistic regression

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""
		
		# 权重矩阵
		self.W = np.zeros((n_in, n_out), dtype=tn.config.floatX)
		# 偏置矩阵
		self.b = np.zeros((n_out,), dtype=tn.config.floatX)
		# 全部模型参数
		self.theta = [self.W, self.b]
		
		pass
	
	def compute(self, x):
		return np.dot(x, self.W) + self.b
	
	def error(self, y, z):
		return 0.5 * ((z - y) ** 2)
	
	def grad(self, y, z):
		return z - y
	
	def delta(self, g, x):
		return np.dot(g, x) / x.shape[0], np.mean(g)
	
	def loss(self, e):
		return np.mean(e)
	
	
		
def load_data(data_filename):
	print('loading data ...')
	
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
	print(data_filename)
	
	# 加载数据文件
	with open(data_filename, 'rb') as f:
		data = np.asarray(pickle.load(f, encoding='bytes'))
		print('loading data - success')
		return data
		
	print('loading data - failed')
	pass

def split_data(data, borrow=True):
	print('spliting data ...')

	data = np.asarray(data, dtype=tn.config.floatX)
	data_x = data[:, :-1]
	data_y = data[:, -1]
	m, n = data.shape
	
	print('spliting data - success')
	return (data_x, data_y, m, n - 1)

if __name__ == '__main__':
	data_file = 'simple_regression.pkl'
	learning_rate = 0.02
	epochs = 10000
	
	data = load_data(data_file)
	data_x, data_y, m, n = split_data(data)
	print(data_x.shape, data_y.shape, m, n)
	
	regression = LinearRegression(n_in = n, n_out = 1)
	
	z = regression.compute(data_x).ravel()
	e = regression.error(data_y, z)
	l = regression.loss(e)
	print(l)
	epoch = 0
	while(epoch < epochs):
		g = regression.grad(data_y, z)
		# print(g)
		d = regression.delta(g, data_x)
		print(d)
		regression.W -= learning_rate * d[0]
		regression.b -= learning_rate * d[1]
		
		z = regression.compute(data_x).ravel()
		e = regression.error(data_y, z)
		l = regression.loss(e)
		print(l)
		
		epoch += 1
	
	
	pass
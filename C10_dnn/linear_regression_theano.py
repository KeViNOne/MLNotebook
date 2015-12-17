import os
import timeit

import pickle

import numpy as np

import theano as tn
import theano.tensor as T



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
		self.W = tn.shared(
			value=np.zeros(
				(n_in, n_out),
				dtype=tn.config.floatX
			),
			name='W',
			borrow=True
		)
		# 偏置矩阵
		self.b = tn.shared(
			value=np.zeros(
				(n_out,),
				dtype=tn.config.floatX
			),
			name='b',
			borrow=True
		)
		# 全部模型参数
		
		pass
	
	def compute(self, x):
		return T.dot(x, self.W) + self.b
	
	def error(self, y, z):
		return 0.5 * T.power(z - y, 2)
	
	def grad(self, y, z):
		return z - y
	
	def delta(self, g, x):
		return T.dot(g.T, x), T.mean(g)
	
	def loss(self, e):
		return T.mean(e)
	
	
def load_data(data_filename):
	print('loading data ...')
	
	# 处理文件名
	data_dir, data_file = os.path.split(data_filename)
	print(data_dir, data_file, os.path.isfile(data_filename))
	if data_dir == "" and not os.path.isfile(data_filename):
		# 如果仅为文件名，且该文件不在当前目录中
		data_filename = os.path.join(
			os.path.split(__file__)[0],
			"..",
			"datasets",
			data_filename
		)
	
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
	data_x = tn.shared(data[:, :-1], borrow=borrow)
	data_y = tn.shared(data[:, -1], borrow=borrow)
	m, n = data.shape
	
	print('spliting data - success')
	return (data_x, data_y, m, n - 1)

if __name__ == '__main__':
	data_file = 'simple_regression.pkl'
	
	data = load_data(data_file)
	data_x, data_y, m, n = split_data(data)
	print(m, n)
	
	regression = LinearRegression(n_in = n, n_out = 1)
	
	z = regression.compute(data_x).ravel()
	print(z.eval())
	
	e = regression.error(data_y, z)
	print(e.eval())
	
	l = regression.loss(e)
	print(l.eval())
	
	g = regression.grad(data_y, z)
	print(g.eval())
	d = regression.delta(g, data_x)
	print(d[0].eval(), d[1].eval())
	
	pass
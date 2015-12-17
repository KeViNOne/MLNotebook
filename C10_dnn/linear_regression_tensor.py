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
		return T.dot(g, x) / x.shape[0], T.mean(g)
	
	def loss(self, e):
		return T.mean(e)
	

class LinearRegressionTrainer(object):
	def __init__(self, train_data, m, n, regression = None):
		self.X, self.Y = train_data
		self.m = m
		self.n_in = n
		self.n_out = 1
		
		self.regression = regression if regression != None else LinearRegression(self.n_in, self.n_out)
		
		pass
		
	def train(self, epochs = 1000, learning_rate = 0.1):
		z = regression.compute(data_x).ravel()
		e = regression.error(data_y, z)
		l = regression.loss(e)
		print('training start (loss: {0}):'.format(l.eval()))
		start_time = timeit.default_timer()
		epoch = 0
		while(epoch < epochs):
			g = regression.grad(data_y, z)
			d = regression.delta(g, data_x)
			regression.W -= learning_rate * d[0]
			regression.b -= learning_rate * d[1]
			
			z = regression.compute(data_x).ravel()
			e = regression.error(data_y, z)
			l = regression.loss(e)
			# print(l.eval())
			
			epoch += 1
			print('epoch:', epoch, end='\r')
		print('training finish (loss: {0}) took {1} seconds.'.format(l.eval(), timeit.default_timer() - start_time))
		
		pass
	

def load_data(data_filename):
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
		data = np.asarray(pickle.load(f, encoding='bytes'))
		print('- success')
		return data
		
	print('- failed')
	pass

def split_data(data, borrow=True):
	print('spliting data', end=' ')

	data = np.asarray(data, dtype=tn.config.floatX)
	data_x = tn.shared(data[:, :-1], borrow=borrow)
	data_y = tn.shared(data[:, -1], borrow=borrow)
	m, n = data.shape
	
	print('- success')
	return (data_x, data_y, m, n - 1)

if __name__ == '__main__':
	data_file = 'simple_regression.pkl'
	learning_rate = 0.02
	epochs = 300
	
	data = load_data(data_file)
	data_x, data_y, m, n = split_data(data)
	print(data_x.eval().shape, data_y.eval().shape, m, n)
	
	trainer = LinearRegressionTrainer((data_x, data_y), m, n)
	regression = trainer.regression
	trainer.train(epochs, learning_rate)
	
	pass
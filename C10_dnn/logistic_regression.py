import os
import timeit

import pickle

import numpy as np
import theano as tn

import dataset



class LogisticRegression(object):

	"""逻辑斯谛回归

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
		return 1 / (1 + np.exp(-(np.dot(x, self.W) + self.b)))
	
	def error(self, y, z):
		one = y * np.log(z)
		two = (1 - y) * np.log(1 - z)
		three = (one + two)
		three[np.isnan(three)] = 0
		# print(three)
		return -np.mean(three)
	
	def grad(self, y, z):
		return z - y
	
	def delta(self, g, x):
		return np.dot(g.T, x) / x.shape[0], np.mean(g)
	
	

class LogisticRegressionTrainer(object):
	def __init__(self, train_data, m, n, regression = None):
		self.X, self.Y = train_data
		self.m = m
		self.n_in = n
		self.n_out = 1
		
		self.regression = regression if regression != None else LinearRegression(self.n_in, self.n_out)
		
		pass
	
	def train(self, epochs = 1000, learning_rate = 0.1):
		x = self.X
		y = self.Y
		regression = self.regression
		
		print('training start:')
		start_time = timeit.default_timer()
		z = regression.compute(x)
		# e = regression.error(y, z)
		# print(e)
		epoch = 0
		while(epoch < epochs):
			g = regression.grad(y, z)
			d = regression.delta(g, x)
			regression.W -= learning_rate * d[0].T
			regression.b -= learning_rate * d[1]
			
			z = regression.compute(x)
			e = regression.error(y, z)
			
			epoch += 1
			print('epoch {0}, error {1}'.format(epoch, e), end='\r')
		print('\ntraining finish (error: {0}) took {1} seconds.'.format(e, timeit.default_timer() - start_time))
		
		pass
		

if __name__ == '__main__':
	data_file = 'simple_binarylabel.pkl'
	learning_rate = 0.001
	epochs = 10000
	
	# data = load_data(data_file)
	# data_x, data_y, m, n = split_data(data)
	

	# 加载数据文件
	data_x, data_y = dataset.load_data_array(data_file)
	m, n = data_x.shape
	print('data:', data_x.shape, data_y.shape, m, n)
	
	regression = LogisticRegression(n_in = n, n_out = 1)
	trainer = LogisticRegressionTrainer((data_x, data_y), m, n, regression=regression)
	
	trainer.train(epochs=epochs, learning_rate=learning_rate)
	
	pass
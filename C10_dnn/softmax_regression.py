import os
import timeit

import pickle

import numpy as np
import theano as tn

import dataset



class SoftmaxRegression(object):

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
		
		self.n_in = n_in
		self.n_out = n_out
		
		# 权重矩阵
		self.W = np.zeros((n_in, n_out), dtype=tn.config.floatX)
		# 偏置矩阵
		self.b = np.zeros((n_out,), dtype=tn.config.floatX)
		# 全部模型参数
		self.theta = [self.W, self.b]
		
		pass
	
	def compute(self, x):
		numer = np.exp(np.dot(x, self.W) + self.b)
		denom = np.sum(numer, 1)
		numer /= denom[:, np.newaxis]
		return numer
	
	def error(self, y, z):
		return -np.mean(np.log(z)[np.arange(z.shape[0]),y])
	
	def grad(self, y, z):
		z[np.arange(z.shape[0]),y] -= 1.0
		return z
	
	def delta(self, g, x):
		return np.dot(g.T, x) / x.shape[0], np.mean(g)
	
	

class SoftmaxRegressionTrainer(object):
	def __init__(self, train_data, m, n, k, regression = None):
		self.X, self.Y = train_data
		self.m = m
		self.n_in = n
		self.n_out = k
		
		self.regression = regression if regression != None else SoftmaxRegression(self.n_in, self.n_out)
		
		pass
	
	def train(self, epochs = 1000, learning_rate = 0.1):
		x = self.X
		y = self.Y
		regression = self.regression
		
		start_time = timeit.default_timer()
		z = regression.compute(x)
		# print(z)
		e = regression.error(y, z)
		print('training start  (error: {0})'.format(e))
		epoch = 0
		while(epoch < epochs):
			g = regression.grad(y, z)
			# print(g)
			d = regression.delta(g, x)
			# print(d)
			regression.W -= learning_rate * d[0].T
			regression.b -= learning_rate * d[1]
			
			z = regression.compute(x)
			e = regression.error(y, z)
			
			epoch += 1
			print('epoch {0}, error {1}'.format(epoch, e), end='\r')
		print('training finish (error: {0})'.format(e))
		print('{0} epochs took {1} seconds.'.format(epoch, timeit.default_timer() - start_time))
		
		pass
		
# 0.9050887373997508
if __name__ == '__main__':
	data_file = 'simple_multilabel.pkl'
	learning_rate = 0.0005
	epochs = 10000
	
	data_x, data_y = dataset.load_data_array(data_file)
	m, n = data_x.shape
	k = np.max(data_y) + 1
	print('data:', data_x.shape, data_y.shape, m, n, k)
	
	regression = SoftmaxRegression(n_in = n, n_out = k)
	trainer = SoftmaxRegressionTrainer((data_x, data_y), m, n, k, regression=regression)
	
	trainer.train(epochs=epochs, learning_rate=learning_rate)
	
	pass
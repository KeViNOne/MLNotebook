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
		
		self.setParam((np.zeros((n_in, n_out),dtype=tn.config.floatX), np.zeros((n_out,),dtype=tn.config.floatX)))
		
		pass
	
	def setParam(self, param):
		
		# 权重矩阵
		self.W = param[0]
		# 偏置矩阵
		self.b = param[1]
		# 全部模型参数
		self.param = [self.W, self.b]
		
		pass
	
	def compute(self, x):
		numer = np.exp(np.dot(x, self.W) + self.b)
		denom = np.sum(numer, 1)
		numer /= denom[:, np.newaxis]
		return numer
	
	def loss(self, y, z):
		return -np.mean(np.log(z)[np.arange(z.shape[0]),y])
	
	def grad(self, y, z):
		z[np.arange(z.shape[0]),y] -= 1.0
		return z
	
	def delta(self, g, x):
		return np.dot(g.T, x) / x.shape[0], np.mean(g)
	
	def error(self, y, z):
		l = np.argmax(z, axis=1)
		return np.nonzero(l - y)[0].shape[0] / y.shape[0]

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
		e = regression.error(y, z)
		print('training start  (error: {0:.4%})'.format(e))
		epoch = 0
		while(epoch < epochs):
			g = regression.grad(y, z)
			d = regression.delta(g, x)
			regression.W -= learning_rate * d[0].T
			regression.b -= learning_rate * d[1]
			
			z = regression.compute(x)
			e = regression.error(y, z)
			
			epoch += 1
			print('epoch {0}, error {1:.4%}'.format(epoch, e), end='\r')
		escape_time = timeit.default_timer() - start_time
		print('training finish (error: {0:.4%})'.format(e))
		if(escape_time > 300.):
			print('{0} epochs took {1:.2f} minutes.'.format(epoch, escape_time / 60.))
		else:
			print('{0} epochs took {1:.2f} seconds.'.format(epoch, escape_time))
		
		pass
		

if __name__ == '__main__':
	data_file = 'mnist.pkl.gz'
	learning_rate = 0.1
	epochs = 1000
	borrow = True
	
	data = dataset.load(data_file, True)
	train_set = data[0]
	m, n = train_set[0].shape
	k = np.max(train_set[1]) + 1
	print('data:', train_set[0].shape, train_set[1].shape, m, n, k)
	
	
	regression = SoftmaxRegression(n_in = n, n_out = k)
	
	trainer = SoftmaxRegressionTrainer(
		train_set, 
		m, n, k,
		regression = regression
	)
	
	del data
	del train_set
	
	trainer.train(
		epochs = epochs, 
		learning_rate = learning_rate
	)
	
	pass

if __name__ == '__debug__':
	data_file = 'mnist.pkl'
	learning_rate = 0.0004
	epochs = 10000
	borrow = True
	
	data = dataset.load(data_file)
	m, n = data[0].shape
	k = np.max(data[1]) + 1
	print('data:', data[0].shape, data[1].shape, m, n, k)
	
	train_x, train_y = data
	del data
	
	regression = SoftmaxRegression(n_in = n, n_out = k)
	
	trainer = SoftmaxRegressionTrainer(
		(train_x, train_y), 
		m, n, k,
		regression = regression
	)
	
	trainer.train(
		epochs = epochs, 
		learning_rate = learning_rate
	)
	
	pass

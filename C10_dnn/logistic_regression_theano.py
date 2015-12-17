
import os
import timeit

import dataset

import numpy as np

import theano as tn
import theano.tensor as T


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
		# return T.nnet.sigmoid(T.dot(x, self.W) + self.b)
		return 1 / (1 + T.exp(-(T.dot(x, self.W) + self.b)))
	
	def error(self, x, y):
		z = self.compute(x)
		one = y * T.log(z)
		two = (1 - y) * T.log(1 - z)
		three = one + two
		# three[T.isnan(three)] = 0
		return -T.mean(three)
		

class LogisticRegressionTrainer(object):
	def __init__(self, train_data, m, n, regression = None):
		self.X, self.Y = train_data
		self.m = m
		self.n_in = n
		self.n_out = 1
		
		self.regression = regression if regression != None else LinearRegression(self.n_in, self.n_out)
		
		pass
		
	def train(self, epochs = 1000, learning_rate = 0.1):
		regression = self.regression
		X = self.X
		Y = self.Y
		
		x = T.matrix('x')  # data, presented as rasterized images
		y = T.imatrix('y')  # labels, presented as 1D vector of [int] labels
		
		error = regression.error(x, y)
		g_W = T.grad(cost=error, wrt=regression.W)
		g_b = T.grad(cost=error, wrt=regression.b)
		
		updates = [(regression.W, regression.W - learning_rate * g_W),
					(regression.b, regression.b - learning_rate * g_b)]
		
		train_model = tn.function(
			inputs=[],
			outputs=error,
			updates=updates,
			givens={
				x: X,
				y: Y
			}
		)
		
		print('training start:')
		start_time = timeit.default_timer()
		epoch = 0
		while(epoch < epochs):
			avg_error = train_model()
			epoch += 1
			print('epoch {0}, error {1}'.format(epoch, avg_error), end='\r')
		print('\ntraining finish (error: {0}) took {1} seconds.'.format(regression.error(X, Y).eval(), timeit.default_timer() - start_time))
		
		pass
	

if __name__ == '__main__':
	data_file = 'simple_binarylabel.pkl'
	learning_rate = 0.001
	epochs = 10000
	borrow = True
	
	data = dataset.load_data_array(data_file)
	m, n = data[0].shape
	print('data:', data[0].shape, data[1].shape, m, n)
	
	train_x = tn.shared(data[0].astype(tn.config.floatX), borrow=borrow)
	train_y = tn.shared(data[1].astype(np.int32), borrow=borrow)
	data = None
	
	regression = LogisticRegression(n_in = n, n_out = 1)
	
	trainer = LogisticRegressionTrainer((train_x,train_y), m, n, regression=regression)
	trainer.train(epochs=epochs, learning_rate=learning_rate)
	
	pass
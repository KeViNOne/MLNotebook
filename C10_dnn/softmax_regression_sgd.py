
import os
import timeit

import dataset

import numpy as np

import theano as tn
import theano.tensor as T


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
		self.theta = [self.W, self.b]
		
		pass
	
	def compute(self, x):
		return T.nnet.softmax(T.dot(x, self.W) + self.b)
	
	def error(self, x, y):
		z = self.compute(x)
		return -T.mean(T.log(z)[T.arange(z.shape[0]),y])
		

class SoftmaxRegressionTrainer(object):
	def __init__(self, train_data, m, n, k, regression = None):
		self.X, self.Y = train_data
		self.m = m
		self.n_in = n
		self.n_out = k
		
		self.regression = regression if regression != None else SoftmaxRegression(self.n_in, self.n_out)
		
		pass
		
	def train(self, epochs = 1000, learning_rate = 0.1):
		regression = self.regression
		X = self.X
		Y = self.Y
		
		x = T.fmatrix('x')  # data, presented as rasterized images
		y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
		
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
		
		start_time = timeit.default_timer()
		e = train_model()
		print('training start  (error: {0})'.format(e))
		epoch = 0
		while(epoch < epochs):
			e = train_model()
			epoch += 1
			print('epoch {0}, error {1}'.format(epoch, e), end='\r')
		print('training finish (error: {0})'.format(regression.error(X, Y).eval()))
		print('{0} epochs took {1} seconds.'.format(epoch, timeit.default_timer() - start_time))
		
		pass
	

if __name__ == '__main__':
	data_file = 'simple_multilabel.pkl'
	learning_rate = 0.0006
	epochs = 10000
	borrow = True
	
	data = dataset.load_data_array(data_file)
	m, n = data[0].shape
	k = np.max(data[1]) + 1
	print('data:', data[0].shape, data[1].shape, m, n, k)
	
	train_x = tn.shared(data[0].astype(tn.config.floatX), borrow=borrow)
	train_y = tn.shared(data[1].astype(np.int32), borrow=borrow)
	data = None
	
	regression = SoftmaxRegression(n_in = n, n_out = k)
	
	trainer = SoftmaxRegressionTrainer((train_x,train_y), m, n, k, regression=regression)
	trainer.train(epochs=epochs, learning_rate=learning_rate)
	
	pass
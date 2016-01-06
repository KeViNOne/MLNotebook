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
		
		self.W_value = np.zeros((n_in, n_out),dtype=tn.config.floatX)
		self.b_value = np.zeros((n_out,),dtype=tn.config.floatX)
		
		self.setParam((self.W_value, self.b_value))
		pass
	
	def setParam(self, param):
		
		# 权重矩阵
		self.W = tn.shared(
			value=param[0],
			borrow=True
		)
		# 偏置矩阵
		self.b = tn.shared(
			value=param[1],
			borrow=True
		)
		# 全部模型参数
		self.param = [self.W, self.b]
		
		pass
	
	# def compute(self, x):
	# 	return T.nnet.softmax(T.dot(x, self.W) + self.b)
	
	def compute(self, x):
		numer = T.exp(T.dot(x, self.W) + self.b)
		return numer/numer.sum(1)[:,None]
	
	# def compute(self, x):
	# 	x = T.dot(x, self.W) + self.b
	# 	ret = x - x.max(axis=1)[:,None]
	# 	ret = T.exp(ret)
	# 	return ret/ret.sum(axis=1)[:,None]

	def loss(self, x, y):
		z = self.compute(x)
		return -T.mean(T.log(z[T.arange(z.shape[0]),y]))
		# return -T.mean(T.log(z)[T.arange(z.shape[0]),y])
		# return T.nnet.categorical_crossentropy(z, y).mean()
	
	def error(self, x, y):
		z = self.compute(x)
		l = T.argmax(z, axis=1)
		return 1 - T.mean(T.eq(l, y))

class SoftmaxRegressionTrainer(object):
	def __init__(self, train_data, m, n, k, regression = None):
		self.train_x = self.share_data(train_data[0], tn.config.floatX)
		self.train_y = self.share_data(train_data[1], np.int32)
		
		self.m = m
		self.n_in = n
		self.n_out = k
		
		self.regression = regression if(regression != None)else SoftmaxRegression(self.n_in, self.n_out)
		
		pass
	
	def train(self, epochs = 1000, learning_rate = 0.1):
		regression = self.regression
		
		x = T.fmatrix('x')  # data, presented as rasterized images
		y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
        
		loss = regression.loss(x, y)
		g_W = T.grad(cost=loss, wrt=regression.W)
		g_b = T.grad(cost=loss, wrt=regression.b)
		update = [(regression.W, regression.W - learning_rate * g_W),
					(regression.b, regression.b - learning_rate * g_b)]
		error = regression.error(x, y)
		
		iterate = tn.function(
			inputs=[],
			outputs=loss,
			updates=update,
			givens={
				x: self.train_x,
				y: self.train_y
			}
		)
		
		start_time = timeit.default_timer()
		epoch = 0
		print('training start  (error: {0:.4%})'.format(float(regression.error(self.train_x, self.train_y).eval())))
		while(epoch < epochs):
			e = iterate()
			epoch += 1
			# print('epoch {0}, error {1:.6f}'.format(epoch, float(e)), end='\r')
		escape_time = timeit.default_timer() - start_time
		print('training finish (error: {0:.4%})'.format(float(regression.error(self.train_x, self.train_y).eval())))
		if(escape_time > 300.):
			print('{0} epochs took {1:.2f} minutes.'.format(epoch, escape_time / 60.))
		else:
			print('{0} epochs took {1:.2f} seconds.'.format(epoch, escape_time))
		
		pass
	
	def share_data(self, data, dtype):
		if(data.dtype != np.dtype(dtype)):
			data = data.astype(dtype)
		return tn.shared(data, borrow=borrow)
	

if __name__ == '__run__':
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

if __name__ == '__main__':
	data_file = 'mnist.pkl'
	learning_rate = 0.01
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


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
			name='W',
			borrow=True
		)
		# 偏置矩阵
		self.b = tn.shared(
			value=param[1],
			name='b',
			borrow=True
		)
		# 全部模型参数
		self.param = [self.W, self.b]
		
		pass
	
	def compute(self, x):
		return T.nnet.softmax(T.dot(x, self.W) + self.b)
	
	def error(self, x, y):
		z = self.compute(x)
		return -T.mean(T.log(z)[T.arange(z.shape[0]),y])
	
	def precision(self, x, y):
		z = self.compute(x)
		l = T.argmax(z, axis=1)
		return T.mean(T.eq(l, y))

class SoftmaxRegressionTrainer(object):
	def __init__(self, train_data, m, n, k, regression = None):
		self.X, self.Y = train_data
		self.m = m
		self.n_in = n
		self.n_out = k
		
		self.regression = regression if regression != None else SoftmaxRegression(self.n_in, self.n_out)
		
		pass
		
	def train(self, epochs = 1000, learning_rate = 0.1, valid_ratio = False):
		regression = self.regression
		train_x, train_y, train_m, valid_x, valid_y, valid_m = self.split_data(valid_ratio)
		
		x = T.fmatrix('x')  # data, presented as rasterized images
		y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
		
		error = regression.error(x, y)
		g_W = T.grad(cost=error, wrt=regression.W)
		g_b = T.grad(cost=error, wrt=regression.b)
		
		update = [(regression.W, regression.W - learning_rate * g_W),
					(regression.b, regression.b - learning_rate * g_b)]
		
		iterate = tn.function(
			inputs=[],
			outputs=None,
			updates=update,
			givens={
				x: train_x,
				y: train_y
			}
		)
		
		
		start_time = timeit.default_timer()
		patience = 5
		up = 0
		epoch = 0
		frequency = 100
		best_validation = regression.error(valid_x, valid_y).eval()
		best_param = (regression.W.eval(), regression.b.eval())
		print('training start  (error: {0})'.format(best_validation))
		while(epoch < epochs and up < patience):
			iterate()
			epoch += 1
			if(epoch % frequency == 0):
				validation = regression.error(valid_x, valid_y).eval()
				print('epoch {0}, error {1}'.format(epoch, validation), end='\r')
				if(validation < best_validation):
					best_validation = validation
					best_param = (regression.W.eval(), regression.b.eval())
					up = 0
				else:
					up += 1
		regression.setParam(best_param)
		print('training finish (error: {0})'.format(best_validation))
		print('{0} epochs took {1} seconds.'.format(epoch, timeit.default_timer() - start_time))
		
		pass
	
		
	def split_data(self, valid_ratio):
		valid_m = int(self.m * valid_ratio)
		train_m = self.m - valid_m
		
		pick = np.random.choice(self.m, self.m, replace=False)
		rand_x = self.X[pick,:]
		rand_y = self.Y[pick,:]
		
		train_x = rand_x[0:train_m, :] 
		train_y = rand_y[0:train_m]
		valid_x = rand_x[train_m:, :] 
		valid_y = rand_y[train_m:]
		
		print(train_x.eval().shape, valid_x.eval().shape)
		
		return train_x, train_y, train_m, valid_x, valid_y, valid_m
	

if __name__ == '__main__':
	data_file = 'simple_multilabel.pkl'
	learning_rate = 0.0001
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
	trainer.train(epochs=epochs, learning_rate=learning_rate, valid_ratio=0.2)
	
	pass
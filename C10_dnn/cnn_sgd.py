import os
import time

import math
import numpy as np

import theano as tn
import theano.tensor as T
import theano.tensor.nnet as NN
from theano.tensor.signal import downsample

import dataset


class Conv2DLayer(object):

	def __init__(self, filter_shape, pool_shape = None, padding = True, varis = None, activation = T.tanh):
		assert len(filter_shape) == 4
		
		self.filter_shape = filter_shape
		self.pool_shape = pool_shape
		self.padding = padding
		self.activation = activation

		self.n_in = np.prod(filter_shape[1:])
		self.n_out = filter_shape[0] * filter_shape[2] * filter_shape[3]
		
		self.x = T.tensor4('x')
		
		self.initParams()
		if(varis):
			self.setVars(varis)
		
		pass
	
	def initParams(self):
		E = math.sqrt(6. / (self.n_in + self.n_out))
		rng = np.random.RandomState(22)
		W_value = np.asarray(
			rng.uniform(
				low = -E,
				high = E,
				size = self.filter_shape
			),
			dtype=tn.config.floatX
		)
		if self.activation == T.nnet.sigmoid:
			W_value *= 4
		b_value = np.zeros((self.filter_shape[0],), dtype=tn.config.floatX)
		self.setParams((W_value, b_value))
		pass
	
	def setParams(self, params):
		# 权重矩阵
		self.W = tn.shared(
			value = params[0],
			name = 'W',
			borrow = True
		)
		# 偏置矩阵
		self.b = tn.shared(
			value = params[1],
			name = 'b',
			borrow = True
		)
		# 全部模型参数
		self.params = [self.W, self.b]
		self.setValues()
		pass
	
	def setVars(self, varis = None):
		self.x = varis[0] if(vars and len(varis) > 0)else T.tensor4('x')
		self.setValues()
		pass
	
	def setValues(self):
		conv_out = NN.conv2d(
			input = self.x,
			filters = self.W,
			filter_shape = self.filter_shape,
			# image_shape = self.x.shape,
			border_mode = 'full' if(self.padding)else 'valid'
		)
		self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		pass
	

if __name__ == '__main__':
	data_file = 'mnist.pkl'
	learning_rate = 0.001
	epochs = 10000
	valid_ratio = 0.2
	batch_size = 100
	borrow = True
	
	data = dataset.load(data_file)
	m, n = data[0].shape
	k = np.max(data[1]) + 1
	s = int(m * (1 - valid_ratio))
	print('data:', data[0].shape, data[1].shape, m, n, k, s)
	
	train_x, valid_x, train_y, valid_y = dataset.split(dataset.pick(data, m, random = False), s)
	# del data
	
	x = np.ones((1,1,4,4), dtype=tn.config.floatX)
	print(x, x.shape)
	layer0 = Conv2DLayer(
		filter_shape = (2, 1, 2, 2),
		# pool_shape = (2, 2),
		varis = (x,)
	)
	output0 = layer0.output.eval()
	print(output0, output0.shape)
	layer1 = Conv2DLayer(
		filter_shape = (2, 2, 2, 2),
		# pool_shape = (2, 2),
		varis = (output0,)
	)
	output1 = layer1.output.eval()
	print(output1, output1.shape)
	# trainer = MLPTrainer(
	# 	(train_x, train_y), 
	# 	s, n, k, 100,
	# 	valid_data = (valid_x, valid_y)
	# )
	# 
	# trainer.train(
	# 	epochs = epochs, 
	# 	learning_rate = learning_rate, 
	# 	batch_size = batch_size
	# )
	
	pass

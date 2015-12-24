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

	def __init__(self, filter_shape, pool_shape = None, pool_step = None, padding = False, image_shape = None, varis = None, activation = T.tanh):
		assert len(filter_shape) == 4
		assert pool_shape == None or len(pool_shape) == 2
		assert image_shape == None or len(image_shape) == 4
		
		self.image_shape = image_shape
		self.filter_shape = filter_shape
		self.pool_shape = pool_shape
		self.pool_step = pool_step
		self.padding = padding
		self.activation = activation

		self.n_in = np.prod(filter_shape[1:])
		self.n_out = filter_shape[0] * np.prod(filter_shape[2:])
		
		self.x = T.tensor4('x')
		
		self.initParams()
		if(varis):
			self.setVars(varis)
		
		pass
	
	def initParams(self):
		E = math.sqrt(6. / (self.n_in + self.n_out))
		rng = np.random.RandomState(int(time.time()))
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
		self.x = varis[0] if(varis and len(varis) > 0)else T.tensor4('x')
		self.setValues()
		pass
	
	def setValues(self):
		conv_out = NN.conv2d(
            input = self.x,
            filters = self.W,
            filter_shape = self.filter_shape,
			image_shape = self.image_shape,
			border_mode = 'full' if(self.padding)else 'valid'
        )
		pooled_out = downsample.max_pool_2d(
			input = conv_out,
			ds = self.pool_shape,
			st = self.pool_step,
			ignore_border = (not self.padding)
		) if(self.pool_shape)else conv_out
		self.output = self.activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		pass
	

class SoftmaxClassifier(object):

	def __init__(self, n_in, n_out, vars = None):
		self.x = T.fmatrix('x')
		self.y = T.ivector('y')
		
		self.n_in = n_in
		self.n_out = n_out
		
		self.initParams()
		if(vars):
			self.setVars(vars)
			
		pass
	
	def initParams(self):
		W_value = np.zeros((self.n_in, self.n_out),dtype=tn.config.floatX)
		b_value = np.zeros((self.n_out,),dtype=tn.config.floatX)
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
	
	def setVars(self, vars):
		self.x = vars[0] if(vars and len(vars) > 0)else T.fmatrix('x')
		self.y = vars[1] if(vars and len(vars) > 1)else T.ivector('y')
		self.setValues()
		pass
	
	def setValues(self):
		self.output = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
		self.loss = -T.mean(T.log(self.output)[T.arange(self.output.shape[0]),self.y])
		self.error = 1 - T.mean(T.eq(T.argmax(self.output, axis=1), self.y))
		pass
	

class HiddenLayer(object):

	def __init__(self, n_in, n_out, vars = None, activation = T.tanh):
		self.x = T.fmatrix('x')
		
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
			
		self.initParams()
		if(vars):
			self.setVars(vars)
		
		pass
	
	def initParams(self):
		E = math.sqrt(6. / (self.n_in + self.n_out))
		rng = np.random.RandomState(22)
		W_value = np.asarray(
			rng.uniform(
				low = -E,
				high = E,
				size = (self.n_in, self.n_out)
			),
			dtype=tn.config.floatX
		)
		if self.activation == T.nnet.sigmoid:
			W_values *= 4
		b_value = np.zeros((self.n_out,), dtype=tn.config.floatX)
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
	
	def setVars(self, vars = None):
		self.x = vars[0] if(vars and len(vars) > 0)else T.fmatrix('x')
		self.setValues()
		pass
	
	def setValues(self):
		self.output = self.activation(T.dot(self.x, self.W) + self.b)
		pass
	

class MLP(object):

	def __init__(self, n_in, n_out, n_hidden = 300, vars = None, activation = T.tanh):
		self.x = T.fmatrix('x')
		self.y = T.ivector('y')
		
		self.n_in = n_in
		self.n_out = n_out
		self.n_hidden = n_hidden
		self.activation = activation
		
		if(vars):
			self.setVars(vars)
		else:
			self.setValues()
		
		pass
	
	def setVars(self, vars):
		self.x = vars[0] if(len(vars) > 0)else T.fmatrix('x')
		self.y = vars[1] if(len(vars) > 1)else T.ivector('y')
		self.setValues()
		pass
	
	def setValues(self):
		self.layer_hidden = HiddenLayer(self.n_in, self.n_hidden, vars = (self.x,), activation = self.activation)
		self.layer_output = SoftmaxClassifier(self.n_hidden, self.n_out, vars = (self.layer_hidden.output, self.y))
		
		self.params = self.layer_hidden.params + self.layer_output.params
		
		self.output = self.layer_output.output
		self.loss = self.layer_output.loss
		self.error = self.layer_output.error
		
		pass
	

class LeNet(object):

	def __init__(self, varis = None, activation = T.tanh):
		self.x = T.matrix('x')
		self.y = T.ivector('y')
		
		self.activation = activation
		self.n_in = (4, 4)
		self.n_out = 2
		self.n_hidden = 500
		
		if(varis):
			self.setVars(varis)
		else:
			self.setValues()
		
		pass
	
	def setVars(self, varis):
		self.x = varis[0] if(len(varis) > 0)else T.matrix('x')
		self.y = varis[1] if(len(varis) > 1)else T.ivector('y')
		self.setValues()
		pass
	
	def setValues(self):
		self.x = self.x.reshape((self.x.shape[0], 1, self.n_in[0], self.n_in[1]))
		self.layer0 = Conv2DLayer(
			varis = (self.x,),
			filter_shape = (2, 1, 2, 2),
			pool_shape = (2, 2),
			pool_step = (1, 1),
			padding = True
		)
		self.layer1 = Conv2DLayer(
			varis = (self.layer0.output,),
			filter_shape = (3, 2, 2, 2),
			pool_shape = (2, 2),
			pool_step = (1, 1),
			padding = True
		)
		self.layer_output = MLP(
			vars = (self.layer1.output.flatten(2), self.y),
			n_in = 48,
			n_out = 2,
			n_hidden = 100
		)
		
		self.params = self.layer0.params + self.layer1.params + self.layer_output.params
		
		self.output = self.layer_output.output
		self.loss = self.layer_output.loss
		self.error = self.layer_output.error
		
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
	
	x = np.ones((5,16), dtype=tn.config.floatX)
	print(x, x.shape)
	x = tn.shared(x, name='x', borrow=True)
	y = np.array([0, 1, 0, 1, 0], dtype=np.int32)
	print(y, y.shape)
	y = tn.shared(y, name='y', borrow=True)
	nn = LeNet(varis = (x, y))
	print(nn.output.eval().shape)
	
	
	
	# layer0 = Conv2DLayer(
	# 	varis = (x,),
	# 	filter_shape = (2, 1, 2, 2),
	# 	pool_shape = (2, 2),
	# 	pool_step = (1, 1),
	# 	padding = True
	# )
	# output0 = layer0.output
	# print(output0.eval(), output0.eval().shape)
	# layer1 = Conv2DLayer(
	# 	varis = (output0,),
	# 	filter_shape = (3, 2, 2, 2),
	# 	# pool_shape = (2, 2),
	# )
	# output1 = layer1.output.eval()
	# print(output1, output1.shape)
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

import os
import time
import timeit

import math
import numpy as np

import theano as tn
import theano.tensor as T

import dataset


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
	
	
class MLPTrainer(object):
	def __init__(self, train_data, m, n, k, h, valid_data = None):
		self.train_x = self.share_data(train_data[0], tn.config.floatX)
		self.train_y = self.share_data(train_data[1], np.int32)
		self.valid_x = self.share_data(valid_data[0], tn.config.floatX) if(valid_data != None)else self.train_x
		self.valid_y = self.share_data(valid_data[1], np.int32) if(valid_data != None)else self.train_y
		
		self.m = m
		self.n_in = n
		self.n_out = k
		self.n_hidden = h
		
		# self.x = T.fmatrix('x')  # data, presented as rasterized images
		# self.y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
		
		self.classifier = MLP(self.n_in, self.n_out, self.n_hidden)
		
		pass
	
	def train(self, epochs = 1000, learning_rate = 0.1, batch_size = None, valid_frequency = 100):
		classifier = self.classifier
		batch_size = int(batch_size) if(batch_size != None and batch_size >= 1 and batch_size < self.m) else self.m
		batchs = self.m // batch_size
		
		loss = classifier.loss
		gparams = [T.grad(loss, param) for param in classifier.params]
		updates = [
			(param, param - learning_rate * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
		
		i = T.iscalar('i')  # index to a batch
		train = tn.function(
			inputs=[i],
			outputs=None,
			updates=updates,
			givens={
				classifier.x: self.train_x[i * batch_size: (i + 1) * batch_size],
				classifier.y: self.train_y[i * batch_size: (i + 1) * batch_size]
			}
		)
		
		error = classifier.error
		validate = tn.function(
			inputs=[],
			outputs=error,
			givens={
				classifier.x: self.valid_x,
				classifier.y: self.valid_y
			}
		)
		
		start_time = timeit.default_timer()
		patience = 5
		up = 0
		epoch = 0
		frequency = int(valid_frequency) if(valid_frequency != None and valid_frequency >= 1) else 100
		best_validation = float(validate())
		# best_param = (classifier.W.eval(), classifier.b.eval())
		print('training start  (error: {0:.4%})'.format(best_validation))
		while(epoch < epochs and up < patience):
			for i in range(batchs):
				train(i)
			epoch += 1
			if(epoch % frequency == 0):
				validation = float(validate())
				print('epoch {0}, error {1:.4%}'.format(epoch, validation), end='\r')
				if(validation < best_validation):
					best_validation = validation
					# best_param = (classifier.W.eval(), classifier.b.eval())
					up = 0
				else:
					up += 1
		escape_time = timeit.default_timer() - start_time
		print('training finish (error: {0:.4%})'.format(best_validation))
		if(escape_time > 300.):
			print('{0} epochs took {1:.2f} minutes.'.format(epoch, escape_time / 60.))
		else:
			print('{0} epochs took {1:.2f} seconds.'.format(epoch, escape_time))
		
		# classifier.setParam(best_param)
		
		pass
	
	def share_data(self, data, dtype):
		if(data.dtype != np.dtype(dtype)):
			data = data.astype(dtype)
		return tn.shared(data, borrow=borrow)
	

if __name__ == '__run__':
	data_file = 'mnist.pkl.gz'
	learning_rate = 0.005
	epochs = 10000
	batch_size = 500
	borrow = True
	
	data = dataset.load(data_file, True)
	train_set, valid_set, test_set = data
	m, n = train_set[0].shape
	k = np.max(train_set[1]) + 1
	print('data:', train_set[0].shape, train_set[1].shape, m, n, k)
	
	classifier = Softmaxclassifier(n_in = n, n_out = k)
	
	trainer = SoftmaxclassifierTrainer(
		train_set, 
		m, n, k,
		valid_data = valid_set,
		classifier = classifier
	)
	
	del(data)
	del(train_set)
	del(valid_set)
	del(test_set)
	
	trainer.train(
		epochs = epochs, 
		learning_rate = learning_rate, 
		batch_size = batch_size
	)
	
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
	del data
	
	trainer = MLPTrainer(
		(train_x, train_y), 
		s, n, k, 100,
		valid_data = (valid_x, valid_y)
	)
	
	trainer.train(
		epochs = epochs, 
		learning_rate = learning_rate, 
		batch_size = batch_size
	)
	
	pass

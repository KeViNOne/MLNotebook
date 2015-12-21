import os
import time
import timeit

import math
import numpy as np

import theano as tn
import theano.tensor as T

import dataset


class SoftmaxLayer(object):

	def __init__(self, n_in, n_out):
		self.n_in = n_in
		self.n_out = n_out
		
		W_value = np.zeros((n_in, n_out),dtype=tn.config.floatX)
		b_value = np.zeros((n_out,),dtype=tn.config.floatX)
		
		self.setParams((W_value, b_value))
		
		pass
	
	def setParams(self, param):
		
		# 权重矩阵
		self.W = tn.shared(
			value = param[0],
			name = 'W',
			borrow = True
		)
		# 偏置矩阵
		self.b = tn.shared(
			value = param[1],
			name = 'b',
			borrow = True
		)
		# 全部模型参数
		self.params = [self.W, self.b]
		
		pass
	
	def output(self, x):
		return T.nnet.softmax(T.dot(x, self.W) + self.b)
	
	def loss(self, x, y):
		z = self.output(x)
		return -T.mean(T.log(z)[T.arange(z.shape[0]),y])
	
	def error(self, x, y):
		z = self.output(x)
		l = T.argmax(z, axis=1)
		return 1 - T.mean(T.eq(l, y))


class HiddenLayer(object):

	def __init__(self, n_in, n_out, activation = T.tanh):
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		
		E = math.sqrt(6. / (n_in + n_out))
		rng = np.random.RandomState(22)
		
		W_value = np.asarray(
			rng.uniform(
				low = -E,
				high = E,
				size = (n_in, n_out)
			),
			dtype=tn.config.floatX
		)
		if activation == T.nnet.sigmoid:
			W_values *= 4
		b_value = np.zeros((n_out,), dtype=tn.config.floatX)
			
		self.setParams((W_value, b_value))
		
		pass
	
	def setParams(self, param):
		
		# 权重矩阵
		self.W = tn.shared(
			value = param[0],
			name = 'W',
			borrow = True
		)
		# 偏置矩阵
		self.b = tn.shared(
			value = param[1],
			name = 'b',
			borrow = True
		)
		# 全部模型参数
		self.params = [self.W, self.b]
		
		pass
	
	def output(self, x):
		return self.activation(T.dot(x, self.W) + self.b)

class MLP(object):

	def __init__(self, n_in, n_out, n_hidden = 300, activation = T.tanh):
		self.n_in = n_in
		self.n_out = n_out
		self.n_hidden = n_hidden
		
		self.layer_hidden = HiddenLayer(n_in, n_hidden, activation)
		self.layer_output = SoftmaxLayer(n_hidden, n_out)
		
		self.params = self.layer_hidden.params + self.layer_output.params
		
		pass
	
	def output(self, x):
		return self.layer_output.output(self.layer_hidden.output(x))

	def loss(self, x, y):
		z = self.output(x)
		return -T.mean(T.log(z)[T.arange(z.shape[0]),y])
	
	def error(self, x, y):
		z = self.output(x)
		l = T.argmax(z, axis=1)
		return 1 - T.mean(T.eq(l, y))
	
class MLPTrainer(object):
	def __init__(self, train_data, m, n, k, h, valid_data = None, classifier = None):
		self.train_x = self.share_data(train_data[0], tn.config.floatX)
		self.train_y = self.share_data(train_data[1], np.int32)
		self.valid_x = self.share_data(valid_data[0], tn.config.floatX) if(valid_data != None)else self.train_x
		self.valid_y = self.share_data(valid_data[1], np.int32) if(valid_data != None)else self.train_y
		
		self.m = m
		self.n_in = n
		self.n_out = k
		self.n_hidden = h
		
		self.classifier = classifier if(classifier != None)else MLP(self.n_in, self.n_out, self.n_hidden)
		
		pass
	
	def train(self, epochs = 1000, learning_rate = 0.1, batch_size = None, valid_frequency = 100):
		classifier = self.classifier
		batch_size = int(batch_size) if(batch_size != None and batch_size >= 1 and batch_size < self.m) else self.m
		batchs = self.m // batch_size
		
		x = T.fmatrix('x')  # data, presented as rasterized images
		y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
		i = T.iscalar('i')  # index to a batch
		
		loss = classifier.loss(x, y)
		gparams = [T.grad(loss, param) for param in classifier.params]
		updates = [
			(param, param - learning_rate * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
		
		train = tn.function(
			inputs=[i],
			outputs=None,
			updates=updates,
			givens={
				x: self.train_x[i * batch_size: (i + 1) * batch_size],
				y: self.train_y[i * batch_size: (i + 1) * batch_size]
			}
		)
		
		error = classifier.error(x, y)
		validate = tn.function(
			inputs=[],
			outputs=error,
			givens={
				x: self.valid_x,
				y: self.valid_y
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
	
	classifier = MLP(n_in = n, n_out = k, n_hidden = 100)
	
	trainer = MLPTrainer(
		(train_x, train_y), 
		s, classifier.n_in, classifier.n_out, classifier.n_hidden,
		valid_data = (valid_x, valid_y),
		classifier = classifier
	)
	
	trainer.train(
		epochs = epochs, 
		learning_rate = learning_rate, 
		batch_size = batch_size
	)
	
	pass

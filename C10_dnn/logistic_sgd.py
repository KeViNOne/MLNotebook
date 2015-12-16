import pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano as tn
import theano.tensor as T


class LogisticRegression(object):
	"""Multi-class Logistic Regression Class

	The logistic regression is fully described by a weight matrix :math:`W`
	and bias vector :math:`b`. Classification is done by projecting data
	points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""

	def __init__(self, input, n_in, n_out):
		""" Initialize the parameters of the logistic regression

		:type input: tn.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""
		# start-snippet-1
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		# 权重矩阵
		self.W = tn.shared(
			value=np.zeros(
				(n_in, n_out),
				dtype=tn.config.floatX
			),
			name='W',
			borrow=True
		)
		# initialize the biases b as a vector of n_out 0s
		# 偏置矩阵
		self.b = tn.shared(
			value=np.zeros(
				(n_out,),
				dtype=tn.config.floatX
			),
			name='b',
			borrow=True
		)

		# symbolic expression for computing the matrix of class-membership
		# probabilities
		# Where:
		# W is a matrix where column-k represent the separation hyperplane for
		# class-k
		# x is a matrix where row-j  represents input training sample-j
		# b is a vector where element-k represent the free parameter of
		# hyperplane-k
		# 由输入计算得到输出(预测概率矩阵)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		# 由输出得到每个样本概率值最高的对应列(标签)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1

		# parameters of the model
		# 模型参数(权重,偏置)
		self.params = [self.W, self.b]

		# keep track of model input
		# 输入数据
		self.input = input

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		.. math::

			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
				\log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
			\ell (\theta=\{W,b\}, \mathcal{D})

		:type y: tn.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label

		Note: we use the mean instead of the sum so that
			  the learning rate is less dependent on the batch size
		"""
		# start-snippet-2
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		# end-snippet-2

	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
		over the total number of examples of the minibatch ; zero one
		loss over the size of the minibatch

		:type y: tn.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label
		"""

		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

def load_data(dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''
	data_dir, data_file = os.path.split(dataset)
	print(data_dir, data_file)
	if data_dir == "" and not os.path.isfile(dataset):
		# 如果仅为文件名，且该文件不在当前目录中
		new_path = os.path.join(
			os.path.split(__file__)[0],
			"..",
			"datasets",
			dataset
		)
	
	# Load the dataset
	with open(dataset, 'rb') as f:
		data = np.asarray(pickle.load(f, encoding='bytes'))
		return data
		# print(pickle.load(f, encoding='bytes'))
	pass

def split_data(data, borrow=True):
	""" Function that loads the dataset into shared variables

	The reason we store our dataset in shared variables is to allow
	tn to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""
	train_x = tn.shared(np.asarray(data[:,0:2],
											dtype=tn.config.floatX),
								borrow=borrow)
	train_y = tn.shared(np.asarray(data[:,2],
											dtype='int32'),
								borrow=borrow)
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return train_x, train_y

def training(data_x, data_y, learning_rate=0.13, epochs=100, batch_size=100):
	print('... building the model')

	# generate symbolic variables for input (x and y represent a
	# minibatch)
	x = T.matrix('x')  # data, presented as rasterized images
	y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

	classifier = LogisticRegression(input=x, n_in=data_x.get_value(borrow=True).shape[1], n_out=10)

	# the cost we minimize during training is the negative log likelihood of
	# the model in symbolic format
	cost = classifier.negative_log_likelihood(y)

	# compute the gradient of cost with respect to theta = (W,b)
	g_W = T.grad(cost=cost, wrt=classifier.W)
	g_b = T.grad(cost=cost, wrt=classifier.b)

	# start-snippet-3
	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs.
	updates = [(classifier.W, classifier.W - learning_rate * g_W),
			   (classifier.b, classifier.b - learning_rate * g_b)]

	# compiling a Theano function `train_model` that returns the cost, but in
	# the same time updates the parameter of the model based on the rules
	# defined in `updates`
	train_model = tn.function(
		inputs=[],
		outputs=cost,
		updates=updates,
		givens={
			x: data_x,
			y: data_y
		}
	)
	
	print('... training the model')
	
	best_validation_loss = np.inf
	test_score = 0.
	start_time = timeit.default_timer()
	
	done_looping = False
	epoch = 0
	while (epoch < epochs) and (not done_looping):
		avg_cost = train_model()
		epoch = epoch + 1
		print(epoch, '-', avg_cost)
	end_time = timeit.default_timer()
	print(end_time - start_time)
	
	pass
	
if __name__ == '__main__':
	dataset='./datasets/simple_multilabel.pkl'
	learning_rate=0.13
	epochs=1000
	batch_size=10

	data = load_data(dataset)
	train_set_x, train_set_y = split_data(data)
	training(train_set_x, train_set_y)
	
	# print(train_set_x.eval(), train_set_y.eval())
	
#	 valid_set_x, valid_set_y = datasets[1]
#	 test_set_x, test_set_y = datasets[2]
# 
#	 # compute number of minibatches for training, validation and testing
#	 n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
#	 n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
#	 n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
# 
#	 print(n_train_batches, n_valid_batches, n_test_batches)
	
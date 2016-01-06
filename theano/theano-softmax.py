# -*- coding: utf-8 -*-

import numpy as np
import theano as tn
import theano.tensor as T


x = T.matrix('x')
y = T.ivector('y')

X = np.asarray([
		[1, 0, 1, 0, 0, 0, 0, 1],
		[0, 1, 0, 1, 0, 0, 1, 0],
		[1,0,0,0,0,1,1,0],
		[0,0,1,1,1,1,1,0]
	],
	dtype=T.config.floatX
)
Y = np.asarray([1,1,0,0], dtype=np.int32)

cost_f = 0.
grad_f = 0.

def softmax(x):
	numer = T.exp(x)
	return numer/numer.sum(1)[:,None]

##########################################################
W = tn.shared(np.zeros((8, 2)).astype(T.config.floatX))
b = tn.shared(np.zeros((2,)).astype(T.config.floatX)) 
params = [W, b]

pred = T.nnet.softmax(T.dot(x, W) + b)
cost = T.nnet.categorical_crossentropy(pred, y).mean()

gparams = T.grad(cost, params)
updates = [
			(param, param - 0.1 * gparam)
			for param, gparam in zip(params, gparams)
		]
grad = gparams[0][0][0]
f = tn.function(
	inputs = [],
	outputs = [cost, grad],
	updates = updates,
	givens = {
		x: X,
		y: Y
	}
)
for i in range(100):
	cost_f, grad_f = f()
print(float(cost_f), float(grad_f))

##########################################################
W = tn.shared(np.zeros((8, 2)).astype(T.config.floatX))
b = tn.shared(np.zeros((2,)).astype(T.config.floatX)) 
params = [W, b]

pred = T.nnet.softmax(T.dot(x, W) + b)
cost = -T.mean(T.log(pred)[T.arange(pred.shape[0]),y])

gparams = T.grad(cost, params)
updates = [
			(param, param - 0.1 * gparam)
			for param, gparam in zip(params, gparams)
		]
grad = gparams[0][0][0]
f = tn.function(
	inputs = [],
	outputs = [cost, grad],
	updates = updates,
	givens = {
		x: X,
		y: Y
	}
)
for i in range(100):
	cost_f, grad_f = f()
print(float(cost_f), float(grad_f))

##########################################################
W = tn.shared(np.zeros((8, 2)).astype(T.config.floatX))
b = tn.shared(np.zeros((2,)).astype(T.config.floatX)) 
params = [W, b]

pred = T.nnet.softmax(T.dot(x, W) + b)
cost = -T.mean(T.log(pred[T.arange(pred.shape[0]),y]))


gparams = T.grad(cost, params)
updates = [
			(param, param - 0.1 * gparam)
			for param, gparam in zip(params, gparams)
		]
grad = gparams[0][0][0]
f = tn.function(
	inputs = [],
	outputs = [cost, grad],
	updates = updates,
	givens = {
		x: X,
		y: Y
	}
)
for i in range(100):
	cost_f, grad_f = f()
print(float(cost_f), float(grad_f))

##########################################################
W = tn.shared(np.zeros((8, 2)).astype(T.config.floatX))
b = tn.shared(np.zeros((2,)).astype(T.config.floatX)) 
params = [W, b]

pred = softmax(T.dot(x, W) + b)
cost = T.nnet.categorical_crossentropy(pred, y).mean()

gparams = T.grad(cost, params)
updates = [
			(param, param - 0.1 * gparam)
			for param, gparam in zip(params, gparams)
		]
grad = gparams[0][0][0]
f = tn.function(
	inputs = [],
	outputs = [cost, grad],
	updates = updates,
	givens = {
		x: X,
		y: Y
	}
)
for i in range(100):
	cost_f, grad_f = f()
print(float(cost_f), float(grad_f))

##########################################################
W = tn.shared(np.zeros((8, 2)).astype(T.config.floatX))
b = tn.shared(np.zeros((2,)).astype(T.config.floatX)) 
params = [W, b]

pred = softmax(T.dot(x, W) + b)
cost = -T.mean(T.log(pred)[T.arange(pred.shape[0]),y])

gparams = T.grad(cost, params)
updates = [
			(param, param - 0.1 * gparam)
			for param, gparam in zip(params, gparams)
		]
grad = gparams[0][0][0]
f = tn.function(
	inputs = [],
	outputs = [cost, grad],
	updates = updates,
	givens = {
		x: X,
		y: Y
	}
)
for i in range(100):
	cost_f, grad_f = f()
print(float(cost_f), float(grad_f))

##########################################################
W = tn.shared(np.zeros((8, 2)).astype(T.config.floatX))
b = tn.shared(np.zeros((2,)).astype(T.config.floatX)) 
params = [W, b]

pred = softmax(T.dot(x, W) + b)
cost = -T.mean(T.log(pred[T.arange(pred.shape[0]),y]))

gparams = T.grad(cost, params)
updates = [
			(param, param - 0.1 * gparam)
			for param, gparam in zip(params, gparams)
		]
grad = gparams[0][0][0]
f = tn.function(
	inputs = [],
	outputs = [cost, grad],
	updates = updates,
	givens = {
		x: X,
		y: Y
	}
)
for i in range(100):
	cost_f, grad_f = f()
print(float(cost_f), float(grad_f))

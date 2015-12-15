# -*- coding: utf-8 -*-

import numpy as np
import theano as tn
import theano.tensor as T

import time
t_ = time.time()


# 数据矩阵
A = np.mat([
	[5,5,3,0,5,5],
	[5,0,4,0,4,4],
	[0,3,0,5,4,5],
	[5,4,3,3,5,5]	
])

# Theano循环语句
X = T.matrix('X')
results, updates = tn.scan(lambda x_k : T.sqrt((x_k ** 2).sum()), sequences = [X])
compute_norm_lines = tn.function(inputs = [X], outputs = [results])

x = np.diag(np.arange(1, 6, dtype = tn.config.floatX), 1)
print(x)

# Theano计算结果
print('Theano:', compute_norm_lines(x)[0])

# Numpy计算结果
print('Numpy:', np.sqrt((x ** 2).sum(1)))

print('Finished in ' + str(time.time() - t_) + ' seconds.')
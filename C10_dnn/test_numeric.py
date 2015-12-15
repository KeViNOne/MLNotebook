import time
import math
import numpy as np
import theano as tn
import theano.tensor as T

# 随机数生成器
rng = np.random.RandomState(round(time.time()))

# 定义向量
B = np.array(
            rng.uniform(low=-1, high=1, size=(2,)),
            dtype='float32'
			)
print(type(B), B)

# 定义矩阵
W = np.mat(
            rng.uniform(low=-1, high=1, size=(2,4)),
            dtype='float32'
			)
print(type(W), W)

# 定义泛型张量
B = np.asarray(
			[0.30439222,-0.26683933,0.41701338],
            dtype=tn.config.floatX
			)
print(type(B), B)	# 1维张量
W = np.asarray(
            [[1,2,3],[4,5,6],[7,8,9]],
            dtype=tn.config.floatX
			)
print(type(W), W)	# 2维张量

print(W.trace())					# 迹
print(np.linalg.inv(W))				# 逆
print(np.linalg.matrix_rank(W))		# 秩

print(W + 10)
print(W * 10)
print(W ** 2)

print(W * B)						# (外积)叉乘
print(np.multiply(W, B))			# (外积)叉乘
print(np.dot(W, B))					# (内积)点乘

# 共享内存张量
B = tn.shared(B, borrow=True)
W = tn.shared(W, borrow=True)
X = tn.shared(np.asarray([1,0,1], dtype=tn.config.floatX), borrow=True)
print(type(B), type(W), type(X))

print((T.dot(X, W) + B).eval())
print(T.log(X).eval())
print(T.mean(X).eval())
print(T.eq(X, X * 2).eval())
print(T.neq(X, X * 2).eval())

x = T.dscalar('x')
y = x ** 2 + 10
f = tn.function([x], T.grad(y, x))
print(f(100))

Y = T.nnet.softmax(T.dot(X, W) + B)
print(Y.eval())
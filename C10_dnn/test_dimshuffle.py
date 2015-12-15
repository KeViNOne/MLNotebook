import time
import math
import numpy as np
import theano as tn
import theano.tensor as T

b = tn.shared( 
		np.arange(1,4,dtype = tn.config.floatX),
		borrow = True
	)
print(b.eval())

print(b.dimshuffle(0).eval())
print(b.dimshuffle(0, 'x').eval())
print(b.dimshuffle(0, 'x', 'x').eval())

W = tn.shared( 
		np.arange(1,10,dtype = tn.config.floatX).reshape(3,3),
		borrow = True
	)
print(W.eval())

print(W.dimshuffle(0, 1).eval())
print(W.dimshuffle(1, 0).eval())
print(W.dimshuffle(0, 1, 'x').eval())
print(W.dimshuffle(0, 'x', 1).eval())
print(W.dimshuffle('x', 0, 1).eval())
（1）theano主要支持符号矩阵表达式

（2）theano与numpy中都有broadcasting：numpy中是动态的，而theano需要在这之前就知道是哪维需要被广播。针对不同类型的数据给出如下的一张表，基本类型包括scalar、vector、row、col、matrix、tensor3、tensor4，然后有整形int对应的8、16、32、64位分别为b、w、i、l；float类型对应的32、64位为f、d；complex类型对应的64、128位为c、z。

Constructor	dtype	ndim	shape	broadcastable
bscalar		int8		0	()	()
bvector		int8		1	(?,)	(False,)
brow		int8		2	(1,?)	(True, False)
bcol		int8		2	(?,1)	(False, True)
bmatrix		int8		2	(?,?)	(False, False)
btensor3	int8		3	(?,?,?)	(False, False, False)
btensor4	int8		4	(?,?,?,?)	(False, False, False, False)
wscalar		int16		0	()	()
wvector		int16		1	(?,)	(False,)
wrow		int16		2	(1,?)	(True, False)
wcol		int16		2	(?,1)	(False, True)
wmatrix		int16		2	(?,?)	(False, False)
wtensor3	int16		3	(?,?,?)	(False, False, False)
wtensor4	int16		4	(?,?,?,?)	(False, False, False, False)
iscalar		int32		0	()	()
ivector		int32		1	(?,)	(False,)
irow		int32		2	(1,?)	(True, False)
icol		int32		2	(?,1)	(False, True)
imatrix		int32		2	(?,?)	(False, False)
itensor3	int32		3	(?,?,?)	(False, False, False)
itensor4	int32		4	(?,?,?,?)	(False, False, False, False)
lscalar		int64		0	()	()
lvector		int64		1	(?,)	(False,)
lrow		int64		2	(1,?)	(True, False)
lcol		int64		2	(?,1)	(False, True)
lmatrix		int64		2	(?,?)	(False, False)
ltensor3	int64		3	(?,?,?)	(False, False, False)
ltensor4	int64		4	(?,?,?,?)	(False, False, False, False)
dscalar		float64		0	()	()
dvector		float64		1	(?,)	(False,)
drow		float64		2	(1,?)	(True, False)
dcol		float64		2	(?,1)	(False, True)
dmatrix		float64		2	(?,?)	(False, False)
dtensor3	float64		3	(?,?,?)	(False, False, False)
dtensor4	float64		4	(?,?,?,?)	(False, False, False, False)
fscalar		float32		0	()	()
fvector		float32		1	(?,)	(False,)
frow		float32		2	(1,?)	(True, False)
fcol		float32		2	(?,1)	(False, True)
fmatrix		float32		2	(?,?)	(False, False)
ftensor3	float32		3	(?,?,?)	(False, False, False)
ftensor4	float32		4	(?,?,?,?)	(False, False, False, False)
cscalar		complex64	0	()	()
cvector		complex64	1	(?,)	(False,)
crow		complex64	2	(1,?)	(True, False)
ccol		complex64	2	(?,1)	(False, True)
cmatrix		complex64	2	(?,?)	(False, False)
ctensor3	complex64	3	(?,?,?)	(False, False, False)
ctensor4	complex64	4	(?,?,?,?)	(False, False, False, False)
zscalar		complex128	0	()	()
zvector		complex128	1	(?,)	(False,)
zrow		complex128	2	(1,?)	(True, False)
zcol		complex128	2	(?,1)	(False, True)
zmatrix		complex128	2	(?,?)	(False, False)
ztensor3	complex128	3	(?,?,?)	(False, False, False)
ztensor4	complex128	4	(?,?,?,?)	(False, False, False, False)



(DL) xuqian@severA914:~$ python Experiments/MLNotebook/C10_dnn/softmax_regression.py
Using gpu device 0: Tesla K20c
loading data - /home/xuqian/Experiments/MLNotebook/C10_dnn/../datasets/mnist.pkl.gz - success
data: (50000, 784) (50000,) 50000 784 10
training start  (error: 90.1360%)
training finish (error: 9.8280%)
1000 epochs took 246.38 seconds.

(DL) xuqian@severA914:~$ python Experiments/MLNotebook/C10_dnn/softmax_regression_theano.py
Using gpu device 0: Tesla K20c
loading data - /home/xuqian/Experiments/MLNotebook/C10_dnn/../datasets/mnist.pkl.gz - success
data: (50000, 784) (50000,) 50000 784 10
training start  (error: 90.1360%)
training finish (error: 9.7360%)
1000 epochs took 24.42 seconds.

(DL) xuqian@severA914:~$ python Experiments/MLNotebook/C10_dnn/softmax_regression_esi.py
Using gpu device 0: Tesla K20c
loading data - /home/xuqian/Experiments/MLNotebook/C10_dnn/../datasets/mnist.pkl.gz - success
data: (50000, 784) (50000,) 50000 784 10
training start  (error: 90.0900%)
training finish (error: 8.2300%)
4000 epochs took 70.34 seconds.

(DL) xuqian@severA914:~$ python Experiments/MLNotebook/C10_dnn/softmax_regression_sgd.py
Using gpu device 0: Tesla K20c
loading data - /home/xuqian/Experiments/MLNotebook/C10_dnn/../datasets/mnist.pkl.gz - success
data: (50000, 784) (50000,) 50000 784 10
training start  (error: 90.0900%)
training finish (error: 7.1500%)
3500 epochs took 5.75 minutes.
(DL) xuqian@severA914:~$ python2 Experiments/
DeepLearningTutorials/ MLNotebook/

(DL) xuqian@severA914:~$ python2 Experiments/DeepLearningTutorials/code/logistic_sgd.py
Using gpu device 0: Tesla K20c
... loading data
... building the model
... training the model
epoch 1, minibatch 50/50, validation error 16.390000 %
     epoch 1, minibatch 50/50, test error of best model 17.280000 %
...
epoch 100, minibatch 50/50, validation error 8.100000 %
     epoch 100, minibatch 50/50, test error of best model 8.170000 %
Optimization complete with best validation score of 8.100000 %,with test performance 8.170000 %
The code run for 101 epochs, with 9.802333 epochs/sec
The code for file logistic_sgd.py ran for 10.3s
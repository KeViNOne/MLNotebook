# -*- coding: utf-8 -*-

import time
t_ = time.time()

# 导入科学计算包
import numpy as np

# 将数组转化为数学矩阵
myVector = np.mat([1,2,3])
print(myVector)
myMatrix = np.mat([[1,2,3],[4,5,6],[7,8,9]])
print(myMatrix)

# 矩阵初始化
myZero = np.zeros([3,5])	# 3x5的全0矩阵
print(myZero)
myOne = np.ones([3,5])		# 3x5的全1矩阵
print(myOne)
myRand = np.random.rand(3,5)	# 3x5的0~1之间的随机数矩阵
print(myRand)
myEye = np.eye(3)	# 3x3的单位矩阵
print(myEye)

# 矩阵标量四则运算
a = 10
b = 5
c = 2
print(a * myMatrix + b)


# 矩阵标量函数运算
print(np.sum(myMatrix))	# 矩阵各元素求和
print(np.multiply(myMatrix, myMatrix))	# 矩阵对应元素相乘
print(np.power(myMatrix, c))	# 矩阵各元素n次幂

# 矩阵转置(返回副本)
print(myVector.T)
print(myVector.transpose())

# 矩阵加减运算(行列数必须相同)
print(myOne + myRand)
print(myOne - myRand)

# 矩阵布尔运算(行列数必须相同)
print(myMatrix == myMatrix.T)
print(myMatrix < myMatrix.T)

# 矩阵点乘运算
print(myMatrix * myVector.T)

# 其他操作
print(myMatrix.copy())	# 生成副本
print(np.shape(myRand));	# 矩阵的行列数(或各维)
print(myMatrix[0]);	# 按行切片
print(myMatrix[:,0]); # 按列切片

# 线性代数库(numpy.linalg)
print(np.linalg.det(myMatrix))	# 矩阵行列式
print(np.linalg.inv(myMatrix))	# 矩阵的逆
print(np.linalg.matrix_rank(myMatrix))	# 矩阵的秩
print(np.linalg.solve(myMatrix,myVector.T))	# 可逆矩阵求解

# 矩阵特征值向量与特征向量矩阵
evals, evecs = np.linalg.eig(myMatrix)	# 库函数计算
print(evals, evecs)
sigma = evals * np.eye(evals.shape[0])	# 特征值向量转化为Sigma对角矩阵
print(sigma)
print(evecs * sigma * np.linalg.inv(evecs))

print(myMatrix.ndim)
print(myMatrix.shape)
print(myMatrix.trace())
print(np.hstack((myMatrix,myMatrix)))
print(np.vstack((myMatrix,myMatrix)))
#print(np.dstack((myMatrix,myMatrix)))


print('Finished in ' + str(time.time() - t_) + ' seconds.')
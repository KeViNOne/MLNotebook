# -*- coding: utf-8 -*-

import time
t_ = time.time()

# 导入科学计算包
import numpy as np

# 数据矩阵
A = np.mat([
	[5,5,3,0,5,5],
	[5,0,4,0,4,4],
	[0,3,0,5,4,5],
	[5,4,3,3,5,5]	
])

# 奇异值(SVD)分解

# 库函数分解
U1, S1, VT1 = np.linalg.svd(A)
print(S1)

# 手工分解
U2 = A * A.T
lamda, hU = np.linalg.eig(U2)
VT2 = A.T * A
eV, hVT = np.linalg.eig(VT2)
hV = hVT.T
S2 = np.sqrt(lamda)
print(S2)

# 转换为Sigma矩阵
r = S1.shape[0]
Sigma1 = np.zeros(A.shape)
Sigma1[:r,:r] = np.diag(S1)
print(Sigma1)
print(U1 * Sigma1 * VT1)	# 还原原矩阵

print('Finished in ' + str(time.time() - t_) + ' seconds.')
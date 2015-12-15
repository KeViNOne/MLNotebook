# -*- coding: utf-8 -*-

import time
t_ = time.time()

# 导入科学计算包
import numpy as np

# 将数组转化为数学矩阵
vector0 = np.mat([8,1,6])
vector1 = np.mat([1,2,3])
vector2 = np.mat([4,5,6])

# 空间向量的范数
print(np.sum(vector0 != 0))	# L0范数
print(np.sum(np.abs(vector0)))	# L1范数
print(np.linalg.norm(vector0, np.inf))	# L1范数
print(np.sqrt(np.sum(np.power(vector0, 2))))	# L2范数
print(np.linalg.norm(vector0))	# L2范数
print(np.max(np.abs(vector0)))	# Linf范数
print(np.min(np.abs(vector0)))	# L-inf范数

# 闵可夫斯基距离(Minkowsiki Distance)
print(np.sum((vector1 - vector2) != 0))		# 汉明(Hamming)距离 (p=0) (L0范数) (等长的编辑距离)
print(np.sum(np.abs(vector1 - vector2)))	# 曼哈顿(Manhattan)距离 (p=1) (L1范数)
print(np.sqrt((vector1 - vector2) * (vector1 - vector2).T)[0,0])	# 欧氏(Euclidean)距离 (p=2) (L2范数)
print(np.max(np.abs(vector1 - vector2)))	# 切比雪夫(Chebyshev)距离 (p=inf) (Linf范数)

# 余弦相似度(Cosine)
print((vector1 * vector2.T) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

# 汉明距离(Hamming)




print('Finished in ' + str(time.time() - t_) + ' seconds.')
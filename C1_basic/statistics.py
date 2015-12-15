# -*- coding: utf-8 -*-

import time
t_ = time.time()

# 导入科学计算包
import numpy as np

featuremat = np.mat([
	[88.5,96.8,104.1,111.3,117.7,124.0,130.0,135.4,140.2,145.3,151.9,159.5,165.9,169.8,171.6,172.3,172.7],
	[12.54,14.65,16.64,18.98,21.26,24.06,27.33,30.46,33.74,37.69,42.49,48.08,53.37,57.08,59.35,60.68,61.40]	
])

# 计算期望(均值）
m1 = np.mean(featuremat[0])
m2 = np.mean(featuremat[1])
print(m1, m2)

# 计算标准差
d1 = np.std(featuremat[0])
d2 = np.std(featuremat[1])
print(d1, d2)

# 计算方差
v1 = np.var(featuremat[0])
v2 = np.var(featuremat[1])
print(v1, v2)

# 计算协方差矩阵
cov = np.cov(featuremat)
print(cov)

# 相关性
corref = np.mean(np.multiply(featuremat[0] - m1, featuremat[1] - m2)) / (d1 * d2)	# 相关系数(Correlation Coefficient)
cordis = 1 - corref		# 相关距离(Correlation Distance)
print(corref, cordis)
print(np.corrcoef(featuremat))		# 相关系数矩阵

# 马氏距离(Mahalanobis Distance)
covinv = np.linalg.inv(cov)
delta = featuremat.T[0] - featuremat.T[1]
print(np.sqrt(delta * covinv * delta.T))

print('Finished in ' + str(time.time() - t_) + ' seconds.')
import numpy as np

v = np.array([1,2,3,4])
print(v)

# create a range
x = np.arange(0, 10, 1) # arguments: start, stop, step    
print(x)

# using linspace, both end points ARE included
y = np.linspace(0, 10, 4)  # arguments: start, stop, totalnum  
print(y)

#return base^x
z = np.logspace(0, 3, 3, base=10)
print(z)

x, y = np.mgrid[0:5, 0:5] # similar to meshgrid in MATLAB,x右扩展，y下扩展
print(x)
print(y)


#random data
from numpy import random

# uniform random numbers in [0,1]
r = random.rand(5,5)
print(r)

# standard normal distributed random numbers
rn = random.randn(5,5)
print(rn)

# diag
# a diagonal matrix
d = np.diag([1,2,3,4])
print(d)
# 提取对角线
d2 = np.diag(d)
print(d2)

# diagonal with offset from the main diagonal
d_offset = np.diag([1,2,3,4],k =1)
print(d_offset)

#zeros 与ones
z = np.zeros((3,4))
print(z)

one = np.ones((3,4))
print(one)

#文件和创建数组
#wget http://labfile.oss.aliyuncs.com/courses/348/stockholm_td_adj.dat
#用 numpy.genfromtxt 函数读取CSV文件
data = np.genfromtxt(r'C:\Users\liufeng\Desktop\stockholm_td_adj.dat')
print(data.shape)

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(14,4))
ax.plot(data[:,0]+data[:,1]/12.0+data[:,2]/365,data[:,5] )
ax.axis('tight')
ax.set_title('temperatures in stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature(C)')
fig

#使用 numpy.savetxt 我们可以将 Numpy 数组保存到csv文件中:
M = np.random.rand(3,4)
np.savetxt(r'C:\Users\liufeng\Desktop\random-matrix.csv',M,fmt = '%.5f')

#Numpy 原生文件类型
np.save(r'C:\Users\liufeng\Desktop\random-matrix.npy',M)
M1 = np.load(r'C:\Users\liufeng\Desktop\random-matrix.npy')
print(M,M1)


# 高级索引
xA = np.array([[n+m*10 for n in range(5)] for m in range(5)])
row_indices = [1,2,3]
r = xA[row_indices]
print(r)

#使用索引掩码
B = np.array([n for n in range(5)])
row_mask = np.array([True, False, True, False, False])
r = B[row_mask]
print(r)

#使用比较操作符生成掩码:
x = np.arange(0, 10, 0.5)
mask = (5 < x) * (x < 7.5)
r = x[mask]
#print(x,mask,r)

#使用 where 函数能将索引掩码转换成索引位置
print(np.where(mask))   #返回为true 的索引

#take 函数与高级索引（fancy indexing）用法相似
v2 = np.arange(-3,3)
row_indices = [1, 3, 5]
r = v2.take(row_indices)
print(r)
#但可以作用于list
r2 = np.take([-3, -2, -1,  0,  1,  2], row_indices)
print(r2)

which = [1, 0, 1, 0]
choices = [[-2,-2,-2,-2], [5,5,5,5]]
r = np.choose(which, choices)
print(r)


#当我们在array间进行*乘时，它的默认行为是 element-wise(逐项乘) 的
A = np.ones((3,4))
r = A*A
print(r)

v1 = np.arange(0, 5) #.reshape(5,1)
A = np.arange(1,26).reshape(5,5)
r = A*v1
r2 = v1*A
print(r)
print(r2)

# 使用 dot 函数进行 矩阵－矩阵，矩阵－向量，数量积乘法
A = np.ones((3,3))
v1 = np.arange(1,4)
r = np.dot(A,A)
r2 = np.dot(A,v1)
r3 = np.dot(v1,A)
r4 = np.dot(v1,v1) # 生成标量
v2 = np.arange(1,4).reshape(1,3)
r5 = np.dot(v2.reshape(3,1),v2)  #生成矩阵形式array
print(r)
print(r2)
print(r3)
print(r4)
print(r5)

#将数组对象映射到 matrix 类型。
v = np.random.rand(3,4)
M = np.matrix(v)
print(type(v))
print(type(M))
# 矩阵乘法
v = np.arange(1,6)
M = np.matrix(v)
ml = M.T*M  #我们也可以使用 transpose 函数完成同样的事情。
print(ml)

# 共轭操作
C = np.matrix([[1+1j, 2+2j], [3j, 4j]])
r = np.conjugate(C)
print(r)

#共轭转置
C = np.matrix([[1+1j, 2+2j], [3j, 4j]])
r = C.H
print(r)

#real 与 imag 能够分别得到复数的实部与虚部
C = np.matrix([[1+1j, 2+2j], [3j, 4j]])
re = np.real(C)
img = np.imag(C)
print(re)
print(img)

# angle 与 abs 可以分别得到幅角和绝对值
C = np.matrix([[1+1j, 2+2j], [3j, 4j]])
ag = np.angle(C)
abss = np.abs(C)
print(ag)
print(abss)

### 矩阵计算

import scipy as sp
C = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
r = sp.linalg.inv(C)  #矩阵求逆
print(r)

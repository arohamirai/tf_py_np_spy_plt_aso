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
data = np.genfromtxt(r'C:\Users\lenovo\Desktop\stockholm_td_adj.dat')
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
r2 = C.I  #矩阵求逆
print(r)
print(r2)

#行列式
C = np.matrix([[1,2,3],[4,5,6],[5,8,9]])
r = sp.linalg.det(C)
print(r)

# 数据处理
data = np.genfromtxt(r'C:\Users\lenovo\Desktop\stockholm_td_adj.dat')
print(data.shape)
np.mean(data[:,3])
np.std(data[:,3])
np.var(data[:,3])
data[:,3].min()
data[:,3].max()

d = np.arange(0, 10)
sum(d)
np.prod(d+1) #阶乘
np.cumsum(d) #累加
np.diff(d)  #累减
np.cumprod(d+1)

# 叠加与重复数组
a = np.array([[1, 2,3], [3, 4, 5]])
r= np.repeat(a, 2,axis = 1)
r2 = np.tile(a, 4)

b = np.array([[5, 6, 6]])
r3 = np.concatenate((a, b), axis=0)

# 深拷贝
import copy as cp
A = np.random.rand(3,4)
B = cp.deepcopy(A) # 实验证明cp.copy()也是深拷贝，但对python内置内型却只是浅拷贝

# 遍历数组
v = np.array([1,2,3,4])
for element in v:
    print(element)
    
M = np.array([[1,2], [3,4]])
for row in M:
    print("row", row)
    for element in row:
        print(element)
        
#矢量化函数
a = np.array([-3,-2,-1,0,1,2,3])
b  = np.vectorize(a)
type(b)

# 更改numpy 数据类型
M = np.array([[1,4],[9,16]])
M.dtype
M2 = M.astype(float)
M2.dtype
M3 = M.astype(bool)
M3.dtype

###########################
###########################
###########################
###########################
######### SciPy 科学计算库


###########################
###########################
###########################
###########################
######### matplotlib
# 类matlab函数，不推荐使用
from pylab import *
#使用 qt 作为图形后端：
#%matplotlib qt
import numpy as np
x = np.linspace(0,5,10)
y = x**2
figure()

plot(x,y,'r')
xlabel('x')
ylabel('y')
title('title')
show()


# 创建子图
x = np.linspace(0,5,10)
y = x**2
subplot(1,2,1)
plot(x, y, 'r--')
subplot(1,2,2)
plot(y, x, 'g*-');

#推荐 matplotlib 面向对象 API
import matplotlib.pyplot as plt

x = np.linspace(0,5,10)
y = x**2
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width,height (range 0 to 1)

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')
fig

# 多图
x = np.linspace(0,5,10)
y = x**2
fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
#自动调整标签的位置
fig.tight_layout()
fig

# figsize,dpi
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,3),dpi=1000)
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
fig.tight_layout()
fig

#Matplotlib 可以生成多种格式的高质量图像，
#包括PNG，JPG，EPS，SVG，PGF 和 PDF。
#如果是科学论文的话，我建议尽量使用pdf格式。 
#(pdflatex 编译的 LaTeX 文档使用 includegraphics 命令就能包含 PDF 文件)。 
#一些情况下，PGF也是一个很好的选择。
fig.savefig(r"C:\Users\lenovo\Desktop\filename.pdf",dpi=1000)


#图例
x = np.linspace(0,5,10)
fig = plt.figure(figsize=(8,4),dpi = 300)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, x**2, label="curve1") #添加label
ax.plot(x, x**3, label="curve2")

ax.legend(loc=4)#调整图例位置，0：let matplotlib decide the optimal
                 # 1：upper right corner
                 # 2：upper left corner
                 # 3：lower left corner
                 # 4：lower right corner
fig

#同时包含了标题，轴标，与图例的用法：
x = np.linspace(0,5,10)
fig, ax = plt.subplots()
ax.plot(x, x**2, label="y = x**2")
ax.plot(x, x**3, label="y = x**3")
ax.legend(loc=3); # upper left corner
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title');
fig


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight')
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60])       #坐标范围
axes[2].set_xlim([2, 5])
axes[2].set_title("custom axes range");
fig

#对数刻度  axes.set_yscale("log")
x = np.linspace(0,5,10)
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(x, x**2, x, np.exp(x))
axes[0].set_title("Normal scale")
axes[1].plot(x, x**2, x, np.exp(x))
axes[1].set_yscale("log") #对数刻度
axes[1].set_title("Logarithmic scale (y)");
fig


alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p)*np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T
fig, ax = plt.subplots()

p = ax.pcolor(X/(2*pi), Y/(2*pi), Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)

fig


fig, ax = plt.subplots()
cnt = ax.contour(Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
fig

# 3D 图
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure(figsize=(14,6))
# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)

#绘制线框
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)

#绘制投影轮廓
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir='z', offset=-pi, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-pi, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=3*pi, cmap=cm.coolwarm)
ax.set_xlim3d(-pi, 2*pi);
ax.set_ylim3d(0, 3*pi);
ax.set_zlim3d(-pi, 2*pi);

#改变视图角度
#view_init 可以改变视图角度，读取两个参数: elevation 与 azimuth 角度
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
ax.view_init(30, 45)
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
ax.view_init(70, 30)
fig.tight_layout()

















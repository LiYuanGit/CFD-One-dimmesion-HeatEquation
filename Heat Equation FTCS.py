# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:31:02 2022

@author: LIYuan
"""

#导入必要的函数库
import matplotlib.pyplot as plt      # 绘图库
import numpy as np                   # 数组处理库
# 第一步：*******************定义变量*****************************
xl = -1  #左边界
xr = 1   #右边界
tf = 1   # 计算时间
dx = 0.025
dt = 0.0025
pi = np.pi
afa = 1/(pi*pi)
beta = afa*dt/(dx*dx)

nx = int((xr-xl)/dx)+1  #nx是整数所以用int()函数强制转换为整数
nt = int((tf-0)/dt)+1  #nt是整数所以用int()函数强制转换为整数 
#******************数组的定义********************************
x = np.linspace(xl, xr, nx)       # 数组x的定义
t = np.linspace(0, tf, nt)            # 数组t的定义
un = np.zeros((nt,nx))             # 数组un的定义
""" 注意：un数组大小为(nt,nx)，Python默认数组索引是从0开始，所以
    un（nt,nx）数组越界。 
"""
#注意：un数组大小为(nt,nx)，Python默认数组索引是从0开始，
# 所以\n    un（nt,nx）数组越界。
#******************初始条件**********************************
for i in range(nx):
    un[0,i] = -np.sin(pi*x[i])

# 第二步：************计算un(j,i)****************************
for j in range(1,nt):  # 注意j的起始值为1，因为j=0为初始条件，已给出
    for i in range(1,nx-1):  # 注意求的是内部un，边界由边界条件给出
        un[j,i] = un[j-1,i]+beta*(un[j-1,i+1]-2*un[j-1,i]+un[j-1,i-1])
    un[j,0] = 0 #左边界条件，数值为0
    un[j,nx-1] = 0 #右边界条件，数值为0

#--------------------------------------------------------------------
#-----------绘图代码，比较简单，可以在Matplolib官网下载模板直接修改------
#------------------------------仅供同学们参考-------------------------

# 1.准备数据
t1 = 1              # 计算时间 t=1时u函数的精确解
ue = np.array(x)
ue = -np.exp(-t1)*np.sin(pi*x) 

# 2.创建画布
plt.figure(figsize=(18,7),dpi=100)

# 3.绘图
# 3.1 标题
ax1 = plt.subplot(121)
ax1 = plt.title("Solution field",fontsize=24)
ax1 = plt.xlabel('$x$',fontsize = 24)
ax1 = plt.ylabel('$u$',fontsize = 24)
ax1 = plt.tick_params(labelsize=20)
ax1 =plt.rcParams['xtick.direction']='in'
ax1 =plt.rcParams['ytick.direction']='in'
plt.plot(x, ue, color = "blue",linewidth = 4)
plt.plot(x, un[nt-1,:], color = "red",\
         linestyle='--',linewidth = 4)
plt.legend(['Exact Solution',"FTCS solution"],fontsize=18)

ax2 = plt.subplot(122)
ax2 = plt.title("Discretization error",fontsize=24)
ax2 = plt.ticklabel_format(style='sci', \
                    scilimits=(-1,2), axis='y')
ax2 = plt.xlabel('$x$',fontsize = 24)
ax2 = plt.ylabel('$err$',fontsize = 24)
ax2 = plt.tick_params(labelsize=20)
ax2 = plt.rcParams['xtick.direction']='in'
ax2 = plt.rcParams['ytick.direction']='in'
err = np.abs(ue-un[nt-1,:])
plt.plot(x, err, color = "green",marker = "o",\
         ms = 8, mec = 'k',linewidth = 4)
plt.show()
    
#************************绘制云图**************************
#***********数值解云图
# 准备数据
ue = np.zeros((nt,nx)) # 数组ue的定义,即精确解数组
for i in range(nt):
    for j in range(nx):
        ue[i,j] = -np.exp(-t[i])*np.sin(pi*x[j])

X,T = np.meshgrid(x, t)    #形成网格数据
levels = np.arange(-1, 1, 0.15)
plt.figure(figsize=(12,12),dpi=300)

plt.subplot(2,1,1)              # 将图形分割为2行1列，在第一个窗口绘图
plt.contourf(X, T, un, levels, cmap= 'copper' )
C = plt.contour(X, T, un, levels, colors=('k'), linewidths=(2))
plt.clabel(C,inline=True,fontsize=10)
plt.title('FTCS solution')

plt.subplot(2,1,2)  # 将图形分割为2行1列，在第二个窗口绘图
plt.contourf(X, T, ue, levels, cmap= 'rainbow' )
C = plt.contour(X, T, ue, levels, colors=('b'), linewidths=(2))
plt.clabel(C,inline=True,fontsize=10)
plt.title('Exact solution')

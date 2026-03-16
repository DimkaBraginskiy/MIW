import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('dane5.txt')
# print(a)
# print(a.shape)

x = a[:,[0]]
# print(x)
y = a[:,[1]]
# print(y)

c = np.hstack([x*x*x, x*x, x, np.ones(x.shape)])  # c is phy (symbol)
#MINIMIZED square ERROR:
v = np.linalg.inv(c.T@c)@c.T @ y  # ordinary least squares solution

#MINIMIZED cubic ERROR:
e = y -(v[0]*x*x*x + v[1]*x*x + v[2]*x + v[3])
print('Model 3 - polinomial DEGREE 3')
print((e.T@e)/len(e))

print('Model 2')
print(c)

# print(v)

c1 = np.hstack([1/x, np.ones(x.shape)])
v1 = np.linalg.pinv(c1) @ y # pseudoinverse
#print(c1)
#print(v1)

c2 = np.hstack([x, np.ones(x.shape)])
v2 = np.linalg.pinv(c2) @ y
print('Model 1: linear')
e2 = y -(v2[0]*x + v2[1])
print((e2.T@e2)/len(e2))


plt.plot(x, y, 'ro')   # just points from original data set
# plt.plot(x,v[0]*x*x*x + v[1]*x*x + v[2]*x + v[3],) #cubic polynomial
# plt.plot(x,v1[0]/x + v1[1])  #inverse
plt.plot(x,v2[0]*x + v2[1])  #linear
plt.show()
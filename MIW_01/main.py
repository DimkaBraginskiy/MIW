import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

a = np.loadtxt('dane5.txt')
# print(a)
# print(a.shape)

x = a[:,[0]]
# print(x)
y = a[:,[1]]
# print(y)

#LEVEL 3 POLYNOMIAL

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
# Building the matrix
cTrain = np.hstack([x_train**3, x_train**2, x_train, np.ones(x_train.shape)])
#Model training
vTrain = np.linalg.pinv(cTrain) @ y_train
# e = y_train - y_pred. Now calculating predicted to find e (error):
#Predictions
yPredTrain = vTrain[0]*x_train**3 + vTrain[1]*x_train**2 + vTrain[2]*x_train + vTrain[3]
yPredTest = vTrain[0]*x_test**3 + vTrain[1]*x_test**2 + vTrain[2]*x_test + vTrain[3]

eTrain = y_train - yPredTrain
eTest = y_test - yPredTest

print('Model 3 - polynomial homework')
print("Train : ", (eTrain.T@eTrain)/len(eTrain))
print("Test : ", (eTest.T@eTest)/len(eTest))


#LEVEL 4 POLYNOMIAL
c4Train = np.hstack([x_train**4, x_train**3, x_train**2, x_train, np.ones(x_train.shape)])
v4Train = np.linalg.pinv(c4Train) @ y_train
y4PredTrain = v4Train[0]*x_train**4 + v4Train[1]*x_train**3 + v4Train[2]*x_train**2 + v4Train[3]*x_train + v4Train[4]
y4PredTest = v4Train[0]*x_test**4 + v4Train[1]*x_test**3 + v4Train[2]*x_test**2 + v4Train[3]*x_test + v4Train[4]

e4Train = y_train - y4PredTrain
e4Test = y_test - y4PredTest

print('Model 4 - polynomial homework')
print("Train : ", (e4Train.T@e4Train)/len(e4Train))
print("Test : ", (e4Test.T@e4Test)/len(e4Test))





c = np.hstack([x*x*x, x*x, x, np.ones(x.shape)])
v = np.linalg.inv(c.T@c)@c.T @ y
e = y -(v[0]*x*x*x + v[1]*x*x + v[2]*x + v[3])
print('Model 3 - polynomial')
print((e.T@e)/len(e))


c1 = np.hstack([1/x, np.ones(x.shape)])
v1 = np.linalg.pinv(c1) @ y # pseudoinverse
print('Model 2')
print(c)
# print(v)
#print(c1)
#print(v1)


c2 = np.hstack([x, np.ones(x.shape)])
v2 = np.linalg.pinv(c2) @ y
e2 = y -(v2[0]*x + v2[1])
print('Model 1: linear')
print((e2.T@e2)/len(e2))



plt.scatter(x_train, y_train, color = 'green', label = 'training data')
plt.scatter(x_test, y_test, color = 'red', label = 'tesing data')

x_line = np.linspace(min(x), max(x), 100)
y_line = vTrain[0]*x_line**3 + vTrain[1]*x_line**2 + vTrain[2]*x_line + vTrain[3]

y_line4 = v4Train[0]*x_line**4 + v4Train[1]*x_line**3 + v4Train[2]*x_line**2 + v4Train[3]*x_line + v4Train[4]

plt.plot(x_line, y_line, color='orange', linewidth=2, label='3d degree polynomial approximation')
plt.plot(x_line, y_line4, color='cyan', linewidth=2, label='4th degree polynomial approximation')

plt.legend()
plt.show()





# plt.plot(x,v[0]*x*x*x + v[1]*x*x + v[2]*x + v[3],)
# plt.plot(x,v1[0]/x + v1[1])
# plt.plot(x,v2[0]*x + v2[1])
# plt.show()
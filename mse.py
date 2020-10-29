from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import xlrd
X = pd.read_excel(r"C:\Users\ugand\Desktop\Book1.xlsx", header = None)
Y= pd.read_excel(r"C:\Users\ugand\Documents\output.xlsx", header = None)
test_X = pd.read_excel(r"C:\Users\ugand\Downloads\test_feature_matrix.xlsx", header = None)
test_Y= pd.read_excel(r"C:\Users\ugand\Downloads\test_output.xlsx", header = None)
X= (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()
test_X= (test_X - test_X.mean())/test_X.std()
test_Y = (test_Y - test_Y.mean())/test_Y.std()

X = X.iloc[:,0:2].values
test_X = test_X.iloc[:,0:2].values
ones = np.ones([X.shape[0],1])

Y = Y.iloc[:,0:1].values
test_Y = test_Y.iloc[:,0:1].values
np.random.seed(19680801)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#xs = X[:, 0]
#ys = X[:, 1]
#zs = Y
#ax.scatter(xs, ys, zs)

#ax.set_xlabel('X1')
#ax.set_ylabel('X2')
#ax.set_zlabel('Y')

#plt.show()

learning_rate = 0.04
max_iteration = 300

theta = np.zeros((X.shape[1] + 1, 1))
s_theta = np.zeros((X.shape[1], 1))
mb_theta = np.zeros((X.shape[1], 1))

def h (theta, X) :
  tempX = np.ones((X.shape[0], X.shape[1] + 1))
  tempX[:,1:] = X
  return np.matmul(tempX, theta)

def loss (theta, X, Y) :
  return np.average(np.square(Y - h(theta, X))) / 2

def gradient (theta, X, Y) :
  tempX = np.ones((X.shape[0], X.shape[1] + 1))
  tempX[:,1:] = X
  d_theta = - np.average((Y - h(theta, X)) * tempX, axis= 0)
  d_theta = d_theta.reshape((d_theta.shape[0], 1))
  return d_theta

def gradient_descent (theta, X, Y, learning_rate, max_iteration, gap) :
  cost = np.zeros(max_iteration)
  
  for i in range(max_iteration) :
    d_theta = gradient (theta, X, Y)
    theta = theta - learning_rate * d_theta
    cost[i] = loss(theta, X, Y)
    if i % gap == 0 :
      print ('iteration : ', i, ' loss : ', loss(theta, X, Y)) 
  return theta, cost

theta, cost = gradient_descent (theta, X, Y, learning_rate, max_iteration, 100)

print (theta)


ys = test_X[:, 0]
xs = test_X[:, 1]
zs = test_Y[:, 0]

z= 1.13636085e-15+ (4.30685543e-02*ys) + (2.58972260e-01*xs)- zs
z1= (z*zs.std()) + z.mean()

print(min(z1))


    


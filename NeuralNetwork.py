# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:56:19 2018

@author: admin
"""

import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from scipy import optimize

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import time

#define neuralNetwork function
def neuralNetwork(input_layer_size,hidden_layer_size,out_put_layer):
    data_img = spio.loadmat("data_digits.mat")#import data
    X = data_img['X']
    y = data_img['y']
    
    '''scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)'''  
    
    m,n = X.shape
    """digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    m,n = X.shape
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)"""
    
    ##　Randomize Display a few row of numbers
    rand_indices = [t for t in [np.random.randint(x-x, m) for x in range(100)]]  # generate 100 (0-m)
    display_data(X[rand_indices,:])     # Display 100 numbers
    #nn_params = np.vstack((Theta1.reshape(-1,1),Theta2.reshape(-1,1)))
    
    Lambda = 1
    
    initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size); 
    initial_Theta2 = randInitializeWeights(hidden_layer_size,out_put_layer)
    
    initial_nn_params = np.vstack((initial_Theta1.reshape(-1,1),initial_Theta2.reshape(-1,1)))  #get theta    
    #np.savetxt("testTheta.csv",initial_nn_params,delimiter=",")
    start = time.time()
    result = optimize.fmin_cg(nnCostFunction, initial_nn_params, fprime=nnGradient, args=(input_layer_size,hidden_layer_size,out_put_layer,X,y,Lambda), maxiter=100)
    print (u'Run time：',time.time()-start)
    print (result)
    '''visulized Theta1'''
    length = result.shape[0]
    Theta1 = result[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = result[hidden_layer_size*(input_layer_size+1):length].reshape(out_put_layer,hidden_layer_size+1)    
    display_data(Theta1[:,1:length])
    display_data(Theta2[:,1:length])
    '''predict'''
    p = predict(Theta1,Theta2,X)
    print (u"Accuracy：%f%%"%np.mean(np.float64(p == y.reshape(-1,1))*100))    
    res = np.hstack((p,y.reshape(-1,1)))
    np.savetxt("predict.csv", res, delimiter=',')
                 
                 
def display_data(imgData):
    sum = 0
    '''
    Display 100 numbers 
    1. init a 2D Array
    2. adjust row data to imag matrix putting into array
    3. display
    '''
    m,n = imgData.shape
    width = np.int32(np.round(np.sqrt(n)))
    height = np.int32(n/width);
    rows_count = np.int32(np.floor(np.sqrt(m)))
    cols_count = np.int32(np.ceil(m/rows_count))
    pad = 1
    display_array = -np.ones((pad+rows_count*(height+pad),pad+cols_count*(width+pad)))
    for i in range(rows_count):
        for j in range(cols_count):
            if sum >= m: #exceed row
                break;
            display_array[pad+i*(height+pad):pad+i*(height+pad)+height,pad+j*(width+pad):pad+j*(width+pad)+width] = imgData[sum,:].reshape(height,width,order="F")
            sum += 1
        if sum >= m:  #exceed row
            break;
            
    plt.imshow(display_array,cmap='gray')   #grey image
    plt.axis('off')
    plt.show()

# CostFunction
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    length = nn_params.shape[0] # theta-length
    # get theta1&theta2 back
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,hidden_layer_size+1)
    
    # np.savetxt("Theta1.csv",Theta1,delimiter=',')
    
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))      # y for 0-9，mapping 0/1
    # Mapping y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # Note reshape(1,-1)for giving values
     
    '''Regurization from 1 get rid of theta1 and theta2's first col'''    
    Theta1_colCount = Theta1.shape[1]    
    Theta1_x = Theta1[:,1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]    
    Theta2_x = Theta2[:,1:Theta2_colCount]
    # Regurization of theta^2
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1)))),np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1))))
    
    '''Feedforward,add one col 1 of bias'''
    a1 = np.hstack((np.ones((m,1)),X))      
    z2 = np.dot(a1,np.transpose(Theta1))    
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h  = sigmoid(z3)    
    '''cost function'''    
    J = -(np.dot(np.transpose(class_y.reshape(-1,1)),np.log(h.reshape(-1,1)))+np.dot(np.transpose(1-class_y.reshape(-1,1)),np.log(1-h.reshape(-1,1)))-Lambda*term/2)/m   
    #temp1 = (h.reshape(-1,1)-class_y.reshape(-1,1))
    #temp2 = (temp1**2).sum()
    #J = 1/(2*m)*temp2
    return np.ravel(J)
# 梯度
def nnGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1).copy()   # 这里使用copy函数，否则下面修改Theta的值，nn_params也会一起修改
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,hidden_layer_size+1).copy()
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))      # 数据的y对应0-9，需要映射为0/1的关系    
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # 注意reshape(1,-1)才可以赋值
     
    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta1_colCount = Theta1.shape[1]    
    Theta1_x = Theta1[:,1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]    
    Theta2_x = Theta2[:,1:Theta2_colCount]
    
    Theta1_grad = np.zeros((Theta1.shape))  #第一层到第二层的权重
    Theta2_grad = np.zeros((Theta2.shape))  #第二层到第三层的权重
      
   
    '''正向传播，每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1,np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h  = sigmoid(z3)
    
    
    '''反向传播，delta为误差，'''
    delta3 = np.zeros((m,num_labels))
    delta2 = np.zeros((m,hidden_layer_size))
    for i in range(m):
        #delta3[i,:] = (h[i,:]-class_y[i,:])*sigmoidGradient(z3[i,:])  # 均方误差的误差率
        delta3[i,:] = h[i,:]-class_y[i,:]                              # 交叉熵误差率
        Theta2_grad = Theta2_grad+np.dot(np.transpose(delta3[i,:].reshape(1,-1)),a2[i,:].reshape(1,-1))
        delta2[i,:] = np.dot(delta3[i,:].reshape(1,-1),Theta2_x)*sigmoidGradient(z2[i,:])
        Theta1_grad = Theta1_grad+np.dot(np.transpose(delta2[i,:].reshape(1,-1)),a1[i,:].reshape(1,-1))
    
    Theta1[:,0] = 0
    Theta2[:,0] = 0          
    '''梯度'''
    grad = (np.vstack((Theta1_grad.reshape(-1,1),Theta2_grad.reshape(-1,1)))+Lambda*np.vstack((Theta1.reshape(-1,1),Theta2.reshape(-1,1))))/m
    return np.ravel(grad)

# S-function    
def sigmoid(z):
    h = np.zeros((len(z),1))    # init，to be consistent with length of z
    
    h = 1.0/(1.0+np.exp(-z))
    return h

# S-gradient
def sigmoidGradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g

# random Initialize Weights theta
def randInitializeWeights(L_in,L_out):
    W = np.zeros((L_out,1+L_in))    # theta's weight
    epsilon_init = (6.0/(L_out+L_in))**0.5
    W = np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init # np.random.rand(L_out,1+L_in) -> L_out*(1+L_in) random matrix
    return W
# 初始化调试的theta权重
def debugInitializeWeights(fan_in,fan_out):
    W = np.zeros((fan_out,fan_in+1))
    x = np.arange(1,fan_out*(fan_in+1)+1)
    W = np.sin(x).reshape(W.shape)/10
    return W

# 预测
def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    #p = np.zeros((m,1))
    '''正向传播，预测结果'''
    X = np.hstack((np.ones((m,1)),X))
    h1 = sigmoid(np.dot(X,np.transpose(Theta1)))
    h1 = np.hstack((np.ones((m,1)),h1))
    h2 = sigmoid(np.dot(h1,np.transpose(Theta2)))
    
    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    #np.savetxt("h2.csv",h2,delimiter=',')
    p = np.array(np.where(h2[0,:] == np.max(h2, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(h2[i,:] == np.max(h2, axis=1)[i]))
        p = np.vstack((p,t))
    return p    

if __name__ == "__main__":
    #checkGradient()
    neuralNetwork(400, 25, 10)
#001
#003
#004
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd_data = pd.read_csv(r'C:\Users\54015\data2.csv',engine='python')
datas = pd_data.as_matrix()

data_x = datas[:,0:2]
data_y = datas[:,2:]

#特征缩放
x1_mean=datas[:,0:1].mean()
x2_mean=datas[:,1:2].mean()
x1_max=datas[:,0:1].max()
x2_max=datas[:,1:2].max()
y_max=datas[:,2:].max()
data_x[:,0:1] = (data_x[:,0:1]-x1_mean)/x1_max
data_x[:,1:2] = (data_x[:,1:2]-x2_mean)/x2_max

data_y = (data_y/y_max).T[0]#除于最大数

def sigmoid(x): #激活函数
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x): #求导
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred): #损失函数
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:

  def __init__(self):
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # 截距项，Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x): #前向传播
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1
  
  def feedforward2(self, x):  #用于画图 
    h1 = sigmoid(self.w1 * x + self.w2 * data_x[:,1:2].mean() + self.b1)
    h2 = sigmoid(self.w3 * x + self.w4 * data_x[:,1:2].mean() + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1
  
  def train(self, data, all_y_trues): #反馈训练
    learn_rate = 0.1
    epochs = 200
    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        '''
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        
        delta = -2*(y_true - o1)
        delta_h1 = self.w5 * delta * h1*(1-h1)
        delta_h2 = self.w6 * delta * h2*(1-h2)
        
        self.w6 -=learn_rate * delta * h2
        self.w5 -=learn_rate * delta * h1
        self.w4 -=learn_rate * delta_h2 * x[1]
        self.w3 -=learn_rate * delta_h1 * x[1]
        self.w2 -=learn_rate * delta_h2 * x[0]
        self.w1 -=learn_rate * delta_h1 * x[0]
        self.b3 -=learn_rate * delta
        self.b2 -=learn_rate * delta_h2
        self.b1 -=learn_rate * delta_h1
      if epoch % 100 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))
        '''
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

##########################更新参数#######################################
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
      if epoch % 100 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))        
     
network = OurNeuralNetwork()
network.train(data_x, data_y)

#画图
plt.scatter(data_x[:,0:1],data_y)
x1=np.linspace(0,300,300)
y1=[]
for i in x1:
  y1+=[network.feedforward([(i-x1_mean)/x1_max,(12-x2_mean)/x2_max])]
plt.plot((x1-x1_mean)/x1_max,y1)
plt.show()

import numpy as np
def sigmoid(x):
    # sigmoid函数
    return 1/(1+np.exp(-x))
def deriv_sigmoid(x):
    fx=sigmoid(x)
    return fx*(1-fx)
def mse_loss(y_true,y_pred):
    ## 均方误差损失
    return ((y_true-y_pred)**2).mean()

# 一个小的神经网络预测得到一个身高和体重的人的性别
class OurNeuronNetWork_1:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
    def feedforward(self, x):
        # 定义前向传播
    # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    def train(self, data, all_y_trues):
        # 定义训练及优化过程
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
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

        # --- Update weights and biases
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

      # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
data = np.array([
[-2,-1],
[25,6],
[17,4],
[-15,-6]]
)
all_y_trues = np.array([
1,
0,
0,
1
])
network_1=OurNeuronNetWork_1()
network_1.train(data,all_y_trues)
# 做出对男女的预测结果
emily = np.array([-7,-3])
frank = np.array([20,2])
print("emily:%.3f"%network_1.feedforward(emily))
print("frank:%.3f"%network_1.feedforward(frank))


# 定义一个简单的神经元
class neuron:
    def __init__(self,weights,bias):
        self.weights=weights
        self.bias=bias
    def feedforward(self,inputs):
        total=np.dot(self.weights,inputs)+self.bias
        return sigmoid(total)
weights=np.array([0,1])
bias=4
n=neuron(weights,bias)

x=np.array([2,3])
print(n.feedforward(x))
class OurNeuronNetWork:
    # 利用神经元组成一个简单的二级神经网络
    def __init__(self):
        weights=np.array([0,1])
        bias=0

        self.h1=neuron(weights,bias)
        self.h2=neuron(weights,bias)
        self.o1=neuron(weights,bias)

    def feedforward(self,x):
        out_h1=self.h1.feedforward(x)
        out_h2=self.h2.feedforward(x)

        out_o1=self.o1.feedforward([out_h1,out_h2])

        return out_o1
x=np.array([2,3])
network=OurNeuronNetWork()
print(network.feedforward(x))

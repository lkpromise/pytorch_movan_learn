import numpy as np
def sigmmoid(x):
    return 1/(1+np.exp(-x))
class neuron:
    def __init__(self,weights,bias):
        self.weights=weights
        self.bias=bias
    def feedforward(self,inputs):
        total=np.dot(self.weights,inputs)+self.bias
        return sigmmoid(total)
weights=np.array([0,1])
bias=4
n=neuron(weights,bias)

x=np.array([2,3])
print(n.feedforward(x))
class OurNeuronNetWork:
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

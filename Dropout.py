import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def dropout_layer(X,dropout):
    assert 0<=dropout<=1
    if dropout==0:
        return X
    if dropout==1:
        return torch.zeros_like(X)
    mask=(torch.randn(X.shape)>dropout).float()
    return mask*X/(1-dropout)


num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256

dropout1,dropout2=0.2,0.5

class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training=is_training
        self.lin1=nn.Linear(num_inputs,num_hiddens1)
        self.lin2=nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3=nn.Linear(num_hiddens2,num_outputs)
        self.relu=nn.ReLU()

    def forward(self,X):
        H1=self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training:
            H1=dropout_layer(H1,dropout1)
        H2=self.relu(self.lin2(H1))
        if self.training:
            H2=dropout_layer(H2,dropout2)
        out=self.lin3(H2)
        return out

net=Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)

num_epochs,lr,batch_size=10,0.5,256

train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

loss=nn.CrossEntropyLoss()

trainer=torch.optim.SGD(net.parameters(),lr=lr)

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=d2l.Accumulator(3)
    for X,Y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,Y)  # 现在l是标量
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()  # 可以直接对标量调用backward()
            updater.step()
            metric.add(float(l)*len(Y),d2l.accuracy(y_hat,Y),Y.numel())
        else:
            l.backward()  # 可以直接对标量调用backward()
            updater(X.shape[0])
            metric.add(float(l)*len(Y),d2l.accuracy(y_hat,Y),Y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    plt.show()

train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
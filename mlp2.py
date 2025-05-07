import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)


batch_size,lr,num_epochs=256,0.1,10
loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=lr)

train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)


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
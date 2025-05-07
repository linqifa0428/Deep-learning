import torch
from torch import nn
from torch.nn import functional as F

#保存和加载张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)
#存储一个张量列表
y = torch.zeros(4)
torch.save([x, y], 'xy-file')

xy = torch.load('xy-file')
print(xy[0])
print(xy[1])

#保存一个字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')

mydict2 = torch.load('mydict')
print(mydict2)

#加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

#将模型参数存储为一个文件
torch.save(net.state_dict(), 'mlp.params')

#加载模型参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))

#验证
Y_clone = clone(X)
print(Y_clone == Y)
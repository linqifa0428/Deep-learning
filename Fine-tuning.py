import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['hotdog']=(d2l.DATA_URL+'hotdog.zip',
                       'fba48693a9957c1e75378f9f4a181798d0d55d71')

data_dir=d2l.download_extract('hotdog')

train_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'))
test_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'))

hotdogs=[train_imgs[i][0] for i in range(8)]
not_hotdogs=[train_imgs[-i-1][0] for i in range(8)]
d2l.show_images(hotdogs+not_hotdogs,2,8,scale=1.4)

normalize=torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_augs=torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])


pretrained_net=torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc)

finetune_net=torchvision.models.resnet18(pretrained=True)
finetune_net.fc=nn.Linear(finetune_net.fc.in_features,2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def train_fine_tuning(net,lr,batch_size=128,num_epochs=5,param_group=True):
    train_iter=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir,'train'),transform=train_augs),
        batch_size=batch_size,shuffle=True)
    test_iter=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir,'test'),transform=test_augs),
        batch_size=batch_size,shuffle=False)
    
    devices=d2l.try_all_gpus()
    if not devices:  # 如果没有找到GPU，则使用CPU
        devices = [torch.device('cpu')]
    loss=nn.CrossEntropyLoss(reduction='none')
    net=nn.DataParallel(net,device_ids=devices).to(devices[0])
    if param_group:
        params_1x=[param for name,param in net.named_parameters() if name not in ['fc.weight','fc.bias']]
        trainer=torch.optim.SGD([{'params':params_1x},
                                {'params':net.fc.parameters(),'lr':lr*10}],
                                lr=0.01,weight_decay=0.001)
    else:
        trainer=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=0.001)
    d2l.train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices)
    
train_fine_tuning(finetune_net,5e-5)
#以下是模型比较，所有模型参数初始化为随机值
# scratch_net=torchvision.models.resnet18()
# scratch_net.fc=nn.Linear(scratch_net.fc.in_features,2)
# train_fine_tuning(scratch_net,5e-4,param_group=False)
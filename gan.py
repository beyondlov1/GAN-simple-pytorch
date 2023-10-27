#  参考 https://blog.csdn.net/weixin_43849763/article/details/104370660
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics
import time
import torchvision
from torch.autograd import Variable

np.random.seed(100)
torch.manual_seed(100)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


datadir = "/media/beyond/70f23ead-fa6d-4628-acf7-c82133c03245/home/beyond/Documents/ml/data/dataset/MNIST"
batch_size = 128
epochs = 100
image_shape = (28,28,1)
log_dir = "./logs"
noize_dim = 100


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

traindataloader = DataLoader(
    torchvision.datasets.MNIST(datadir, train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1,), (0.5,))
                               ])),batch_size=batch_size, shuffle=True)


print(next(iter(traindataloader)))


class Generator(nn.Module):
    def __init__(self, in_features, image_shape):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        def block(in_features, out_features, norm = True):
            layers = []
            layers.append(nn.Linear(in_features, out_features))
            if norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *block(in_features, 128, norm = False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, np.prod(image_shape)),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_shape), 512),
            nn.BatchNorm1d(512),  # 这个很重要, 没有这个很难收敛
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.model(x)
        return out
    
def get_noise(batch_size, n_noise, device):
    return Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, n_noise)))).to(device)



gen = Generator(noize_dim, image_shape).to(device)
disc = Discriminator(image_shape).to(device)


# 权重初始化，默认xavier
def init_network(model, method='xavier', excludes=['embedding', "bns"]):
    for name, w in model.named_parameters():
        excluded = False
        for exclude in excludes: #排除embedding层
            if exclude in name:
                excluded = True
                break
        if excluded:
            continue
        if len(w.shape) < 2:
            w = w.unsqueeze(0)

        #权重按指定方式初始化 默认xavier
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
        #偏置初始化为常数0
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass

init_network(gen)
init_network(disc)
        
criterion = nn.BCELoss()
optimizer_G = optim.Adam(gen.parameters(), lr=0.0003)
optimizer_D = optim.Adam(disc.parameters(), lr=0.0003)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)

writer = SummaryWriter(log_dir=log_dir + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

total_batch = 0
for i in range(epochs):
    for j, (data, _) in enumerate(traindataloader):
        real  = Variable(torch.ones((data.size(0), 1))).to(device)
        fake =  Variable(torch.zeros(data.size(0), 1)).to(device)

        noise = get_noise(data.size(0), noize_dim, device)
        gen_data = gen(noise)
        loss_g = criterion(disc(gen_data), real)
        
        optimizer_G.zero_grad()
        loss_g.backward()
        optimizer_G.step()

        loss_d = criterion(disc(data.view(data.size(0), -1)), real) + criterion(disc(gen_data.detach()), fake)
        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        if total_batch % 20 == 0:
            writer.add_scalar("Gloss/train", loss_g.item(), total_batch) 
            writer.add_scalar("Dloss/train", loss_d.item(), total_batch)

        print(f"epoch {i} batch {j} loss_g {loss_g} loss_d {loss_d}")

        if j % 100 == 0:
            print(f"epoch {i} batch {j} loss_g {loss_g} loss_d {loss_d}")
            img_fake = gen_data.view(gen_data.size(0), 1, 28, 28)
            torchvision.utils.save_image(img_fake, f"images/gan_fake_{i}_{j}.png")
            img = data.view(data.size(0), 1, 28, 28)
            torchvision.utils.save_image(img, f"images/gan_real_{i}_{j}.png")

        total_batch += 1
        
    scheduler_G.step()
    scheduler_D.step()





        
        
import torch.nn as nn
import torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset

from utils.TrafficDataset import TrafficDataSet

class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()
        self.num_class = num_class
        resnet = models.resnet18(True)
        feature_size = 107520

        resnet = nn.Sequential(*list(resnet.children())[:-1])

        # resnet.fc = nn.LeakyReLU(0.1)

        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = resnet
        self.fc = nn.Linear(feature_size, num_class)
        # print(self.resnet)

    def forward(self, input):
        feature = []
        # input = input[0]
        print(input.size())
        for i in range(3):
            x = self.resnet(input[:,i,:,:,:])
            x = x.view(x.size(0), -1)
            feature.append(x)
        feature = torch.cat(feature,1)
        print(feature.size())
        out = self.fc(feature)
        return out



if __name__ == "__main__":
    dataset = TrafficDataSet("../data/1208_train", 500, 400)
    dataloader = DataLoader(dataset, 2, num_workers=2, shuffle=True)
    datait = iter(dataloader)

    x, y = next(datait)
    x = x[:,:,0:3,:,:]
    # print(x.)
    model = Model(2)
    model.forward(x.type("torch.FloatTensor"))
    # model = Model(2)
    loss = nn.CrossEntropyLoss()
    torch.optim.adam(model.parameters(), )
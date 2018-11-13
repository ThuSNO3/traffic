
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
        self.fc1 = nn.Linear(feature_size, 1024)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, num_class)

        # print(self.resnet)

    def forward(self, input):
        feature = []
        # input = input[0]
        # print(input.size())
        for i in range(3):
            x = self.resnet(input[:,i,:,:,:])
            x = x.view(x.size(0), -1)
            feature.append(x)
        feature = torch.cat(feature,1)
        # print(feature.size())
        out = self.fc1(feature)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

train_path = "../data/1208_train"
test_path = "../data/1208_test"
save_path = "../model/"
epochs = 100
lr = 0.0001
batch_size = 20
numclass = 2
use_gpu = True

def train(train_data, test_data, model):
    lossfunc = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # print("training samples:", len(train_data))
    for epoch in range(epochs):
        batch = 0
        for x, y in train_data:

            optimzer.zero_grad()
            x = x[:, :, 0:3, :, :]
            x = x.type("torch.FloatTensor")
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            out = model.forward(x)
            loss = lossfunc(out, y)
            loss.backward()
            optimzer.step()
            print("loss:", loss.item())
            batch += 1
            # if batch > 2:
            #     break

        # test
        print()
        print("epoch test:", epoch)

        with torch.no_grad():
            loss = 0
            total = 0
            correct = 0
            for x, y in test_data:
                x = x[:, :, 0:3, :, :]
                x = x.type("torch.FloatTensor")
                if use_gpu:
                    x = x.cuda()
                    y = y.cuda()
                out = model.forward(x)
                loss += lossfunc(out, y).item()
                total += y.size(0)
                _, predicted = torch.max(out, 1)
                correct += (predicted == y).sum()

            print("total test samples:", total)
            print("loss:", loss / total)
            print("accuracy={}%".format(100 * correct / float(total)))
            model_path = os.path.join(save_path, "model.{}.pth".format(epoch))
            torch.save(model.state_dict(), model_path)
            print("saving model to {}".format(model_path))


if __name__ == "__main__":
    train_data = TrafficDataSet(train_path, 500, 400)
    train_loader = DataLoader(train_data, batch_size, num_workers=0, shuffle=True)
    test_data = TrafficDataSet(test_path, 500, 400)
    test_loader = DataLoader(test_data, batch_size, num_workers=0, shuffle=True)
    # datait = iter(dataloader)

    # for x,y in train_loader:
    #     print(x.shape, y.shape)


    print("Train data:", len(train_data))
    print("Test data:", len(test_data))
    print("Build Model......")
    model = Model(numclass)

    if use_gpu:
        model = model.cuda()

    print("Begin training......")
    train(train_loader, test_loader, model)
    #
    # x, y = next(datait)
    # x = x[:,:,0:3,:,:]
    # # print(x.)
    # model = Model(2)
    # model.forward(x.type("torch.FloatTensor"))
    # # model = Model(2)
    # loss = nn.CrossEntropyLoss()
    # # torch.optim.adam(model.parameters(), )
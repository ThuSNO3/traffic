
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import numpy as np


LabelColor = {""}

class TrafficDataSet(Dataset):
    def __init__(self, path, width, height):
        self.path = path
        self.width = width
        self.heigth = height
        self.shape = (height, width)

        self.imagepath = os.path.join(path, "image")
        self.jsonpath = os.path.join(path, "json")
        names = set()
        for file in os.listdir(self.imagepath):
            items = file.split("_")
            key = "_".join(items[:-1])
            names.add(key)
        self.names = list(names)


    def load_label(self, file, shape):
        fd = open(os.path.join(self.jsonpath, file))
        jsonObj = json.load(fd)
        items = jsonObj["shapes"]
        mask = np.zeros(shape, dtype=np.uint8)
        for item in items:
            points = np.array(item["points"])
            points = points[np.newaxis, :]
            # print(b.shape, points.shape)
            cv2.fillPoly(mask, points, 255)
        return mask

    def load_data(self, name):
        array = []
        img_width=0
        img_height=0
        for i in range(1,4):
            file = "{}_{}.jpg".format(name,str(i).zfill(2))
            image = cv2.imread(os.path.join(self.imagepath, file))
            print(file)
            img_height, img_width, _ = image.shape
            image = cv2.resize(image, self.shape)
            array.append(image)

        array = np.stack(array)
        items = name.split("_")
        mask = self.load_label(items[2]+".json",(img_height,img_width))
        mask = cv2.resize(mask, self.shape)
        mask = mask[:,:,np.newaxis]
        mask = np.tile(mask,(3,1,1,1))
        array = np.concatenate((array, mask), -1)
        label = int(items[0])
        return array, label

    def __getitem__(self, index):
        file = self.names[index]
        x, y = self.load_data(file)
        return x, y

    def __len__(self):
        return len(self.names)


if __name__ == "__main__":
    dataset = TrafficDataSet("../../data/1208/", 1000, 800)
    dataloader = DataLoader(dataset, 2, shuffle=True)
    datait = iter(dataloader)
    print(len(datait))
    x, y = next(datait)
    print(x.shape, y.shape)

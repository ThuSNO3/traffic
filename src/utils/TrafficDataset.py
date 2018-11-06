
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import numpy as np


class TrafficDataSet(Dataset):
    def __init__(self, path, width, height):
        self.path = path
        self.width = width
        self.heigth = height
        self.shape = (width, height, 3)

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

    def load_image(self, name):
        array = []

        for i in range(1,4):
            file = "{}_{}.jpg".format(name,str(i).zfill(2))
            image = cv2.imread(os.path.join(self.imagepath,  file))
            shape = image.shape
            image = cv2.resize(image, self.shape)
            array.append(image)


    def __getitem__(self, index):
        file = self.names[index]

    def __len__(self):
        return len(self.names)


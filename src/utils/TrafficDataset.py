
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


labelColor = {'leftline': 10, 'rightline':30, 'straitline':50,
              'rightstraitline':70, 'leftstraitline':90,
               'whiteline':110, 'stopline':130,
              'leftstraitlight':150, 'rightstraitlight':170,
               'leftlight':190, 'straitlight':210,'rightlight':230,}

class TrafficDataSet(Dataset):
    def __init__(self, path, width, height):
        self.path = path
        self.width = width
        self.heigth = height
        self.shape = (width, height)

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
            label = item["label"].lower()
            points = points[np.newaxis, :]
            # print(b.shape, points.shape)
            color = labelColor.get(label)
            cv2.fillPoly(mask, points, color)
        return mask

    def load_data(self, name):
        array = []
        img_width=0
        img_height=0
        for i in range(1,4):
            file = "{}_{}.jpg".format(name,str(i).zfill(2))
            filePath = os.path.join(self.imagepath, file)
            image = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
            # print(file)
            img_height, img_width, _ = image.shape
            image = cv2.resize(image, self.shape)
            # print(self.shape, image.shape)
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
        x = np.transpose(x, (0, 3, 1, 2)) / 255.0
        # print(x.shape)
        return x.astype(np.float), y

    def __len__(self):
        return len(self.names)


if __name__ == "__main__":
    dataset = TrafficDataSet("../../data/1208_test", 1000, 800)
    dataloader = DataLoader(dataset, 1, num_workers=2, shuffle=True)
    datait = iter(dataloader)
    # count = 0
    # for batch, id in datait:
    #     count += 1
    #     print(batch.shape, count)
    x, y = next(datait)
    image = x[0,0,:,:,0:3]
    label = x[0,0,:,:,3]

    # plt.imshow(image,origin=(0,0))
    plt.imshow(label)

    plt.show()
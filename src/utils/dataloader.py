import os
from PIL import Image
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt


def file_normal(path):
    imagepath = os.path.join(path, path, "image")
    jsonpath = os.path.join(path, path, "json")

    for file in os.listdir(imagepath):
        items = file.split("_")
        number = int(items[-1].split(".")[0])

        if number<1 or number>3:
            print(file)

        nname = "{}_{}.jpg".format("_".join(items[:-1]), str(number).zfill(2))
        if nname != file:
            print(file, nname)
            os.rename(os.path.join(imagepath, file), os.path.join(imagepath, nname))

def check(path):

    imagepath = os.path.join(path,path,"image")
    jsonpath = os.path.join(path,path,"json")
    jsonfile = os.listdir(jsonpath)
    #
    # for file in jsonfile:
    #     fd = open(os.path.join(jsonpath, file))
    #     jsonObj = json.load(fd)
    #     shapes = jsonObj["shapes"]
    #     for shape in shapes:
    #         print(shape["label"])


    count = 0
    for image in os.listdir(imagepath):
        items = image.split("_")
        if items[2]+".json" not in jsonfile:
            print(dir, image)
            count += 1
    print(count)

    count = 0
    maps = {}
    for image in os.listdir(imagepath):
        items = image.split("_")
        key = "_".join(items[:-1])
        if key not in maps:
            maps[key] = 0
        maps[key] += 1
    for key,value in maps.items():
        if value != 3:
            print(key)
            count += 1
    print(count)


def load_label(file, shape):
    fd = open(file)
    jsonObj = json.load(fd)
    items = jsonObj["shapes"]
    mask = np.zeros(shape, dtype=np.uint8)
    for item in items:
        points = np.array(item["points"])
        points = points[np.newaxis,:]
        # print(b.shape, points.shape)
        cv2.fillPoly(mask, points, 255)
    cv2.imwrite("test.jpg",mask)

def load_image(file):
    image = cv2.imread(file)
    print(image.shape)
    image = cv2.resize(image,(500, 800))


if __name__ == "__main__":
    # file_normal("E:/workstation/traffic-event/data/1208")
    load_label("E:/workstation/traffic-event/data/1208/json/1ghe225.json", (1080, 1920))
    # load_image("E:/workstation/traffic-event/data/1208/image/0_12080_1ghe225_AA8872_20181031175409_01.jpg")
    # check("E:/workstation/traffic-event/data/1208")
    # load("E:/workstation/traffic-event/data/1208")
import os
from PIL import Image
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt


def file_normal(path):
    imagepath = os.path.join(path, "image")
    jsonpath = os.path.join(path, "json")

    for file in os.listdir(imagepath):
        items = file.split("_")
        number = int(items[-1].split(".")[0])

        if number<1 or number>3:
            print(file)

        nname = "{}_{}.jpg".format("_".join(items[:-1]), str(number).zfill(2))
        if nname != file:
            print(file, nname)
            os.rename(os.path.join(imagepath, file), os.path.join(imagepath, nname))

def delete(src, dest):
    imagepath = os.path.join(src, "image")
    jsonpath = os.path.join(src, "json")

    count = 0
    for file in os.listdir(imagepath):
        dest_file = os.path.join(dest,"image",file)
        print(dest_file)
        if os.path.exists(dest_file):
            count += 1
            os.remove(dest_file)

    print(count)

    for file in os.listdir(jsonpath):
        dest_file = os.path.join(dest,"json",file)
        if os.path.exists(dest_file):
            os.remove(dest_file)


def check(path):

    imagepath = os.path.join(path,"image")
    jsonpath = os.path.join(path,"json")
    jsonfile = os.listdir(jsonpath)

    # markset = set()
    # for file in jsonfile:
    #     fd = open(os.path.join(jsonpath, file))
    #     jsonObj = json.load(fd)
    #     shapes = jsonObj["shapes"]
    #     for shape in shapes:
    #         markset.add(shape["label"].lower())
    #         # print(shape["label"])
    #
    # print(markset)


    count = 0
    for image in os.listdir(imagepath):
        items = image.split("_")
        if items[2]+".json" not in jsonfile:
            print(image)
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
    return mask[:,:,np.newaxis]

def load_image(file):
    image = cv2.imread(file)
    print(image.shape)
    array = np.tile(image,(3,1,1,1))
    mask = load_label("../../data/1208/json/5201002012761797.json", (1080, 1920))
    mask = np.tile(mask, (3,1,1,1))
    print(array.shape, mask.shape)
    result = np.concatenate((array,mask),-1)
    print(result.shape)
    mask = result[0,:,:,3]
    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    # file_normal("E:/workstation/traffic-event/data/1_14")
    # load_label("E:/workstation/traffic-event/data/1208/json/1ghe225.json", (1080, 1920))
    # load_image("../../data/1208/image/1_12081_5201002012761797_贵JRB135_20180930170214_01.jpg")
    # check("E:/workstation/traffic-event/data/1208-split/1_14")
    check("../../data/image18")
    # file_normal("../../data/1208-split/1_04")
    # delete("../../data/1208-split/1_04", "../../data/1208")
    # load("E:/workstation/traffic-event/data/1208")
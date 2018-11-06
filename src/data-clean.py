
import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_seg(filename):
    img = cv2.imread(filename, 0)
    print(img.shape)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    dx = np.abs(sobelx)
    dy = np.abs(sobely)
    dx = (np.sum(abs(dx), axis=0))
    dy = (np.sum(abs(dy), axis=1))
    xi = np.where(dx == np.amax(dx))[0][0]
    yi = np.where(dy == np.amax(dy))[0][0]
    print(yi, xi)

    image = img[0:yi,0:xi]

    plt.subplot(2,2,1) , plt.imshow(img , cmap="gray")
    plt.title("Original") , plt.xticks([]) , plt.yticks([])
    plt.subplot(2,2,2), plt.imshow(image , cmap="gray")
    plt.title("Laplacian") , plt.xticks([])  ,plt.yticks([])
    plt.show()

if __name__ == "__main__":
    image_seg("../data/53.jpg")

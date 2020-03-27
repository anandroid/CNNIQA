from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image



def orb(img_path):
    img = np.array(PIL.Image.open(img_path).convert('L'))

    descriptor_extractor = ORB(n_keypoints=200)

    descriptor_extractor.detect_and_extract(img)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors




    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    ax[0].imshow(np.array(img))
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    plt.show()

    return keypoints1.shape[0]
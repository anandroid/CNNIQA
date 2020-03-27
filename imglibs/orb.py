from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image



def orb(img_path):
    image = PIL.Image.open(img_path).convert('L')
    img = np.array(image)
    img2 = tf.rotate(img, 180)


    descriptor_extractor = ORB(n_keypoints=200)

    descriptor_extractor.detect_and_extract(img)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors


    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)


    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    ax[0].imshow(np.array(img))
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    plt.show()

    image.close()

    return matches12.shape[0]
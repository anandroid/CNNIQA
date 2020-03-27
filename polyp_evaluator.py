"""
Test Demo
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
 Date: 2018/5/26
"""

import os

import torch
from PIL import Image
from sklearn import svm

from CNNIQAnet import CNNIQAnet
from IQADataset import NonOverlappingCropPatches

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    # model = nn.DataParallel(model)

    checkpoint = torch.load('checkpoints/CNNIQA-POLYP-OLD')

    model.load_state_dict(checkpoint)

    clear_dir = "data/ldq/polyp/0-clear/";

    blur_dir = "data/ldq/polyp/1-blurry/"

    clear_images = os.listdir(clear_dir)

    blur_images = os.listdir(blur_dir)

    model.eval()

    missScore = 0

    with torch.no_grad():
        clear_scores = []

        for image in clear_images:
            path = clear_dir + image
            im = Image.open(path).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(image + ":" + str(score))
            clear_scores.append(score)

        print("################")

        blur_scores = []
        for image in blur_images:
            path = blur_dir + image
            im = Image.open(path).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(image + ":" + str(score))
            blur_scores[score]

        SVM_TRAIN_RATIO = 0.6

        x, y = [], []

        for i in range(int(len(clear_scores) * SVM_TRAIN_RATIO)):
            x.append(clear_scores[i])
            y.append(0)

        for i in range(int(len(blur_scores) * SVM_TRAIN_RATIO)):
            x.append(blur_scores[i])
            y.append(1)

        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(x, y)

        misPredicts = 0
        for i in range(int(len(clear_scores) * SVM_TRAIN_RATIO), len(clear_scores)):
            x = clear_scores[i]
            y = clf.predict(x)
            true_value = "Clear "
            pred_value = "Clear"
            if y == 1:
                pred_value = "Blurry"
                misPredicts = misPredicts + 1

            print("Clear | score :" + str(x) + pred_value)

        for i in range(int(len(blur_scores) * SVM_TRAIN_RATIO), len(blur_scores)):
            x = blur_scores[i]
            y = clf.predict(x)
            true_value = "Blurry "
            pred_value = "Blurry"
            if y == 0:
                pred_value = "Clear"
                misPredicts = misPredicts + 1

            print("Blurry | score :" + str(x) + pred_value)

        print("Total :" + str(len(blur_scores) - int(len(blur_scores) * SVM_TRAIN_RATIO)) + " Mispredicted :" +
              str(misPredicts))

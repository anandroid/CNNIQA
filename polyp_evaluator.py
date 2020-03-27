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
from imglibs.orb import orb
from imglibs.blob_detection import  blobdetection
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    # model = nn.DataParallel(model)

    checkpoint = torch.load('checkpoints/CNNIQA-POLYP-RUNNING')

    model.load_state_dict(checkpoint)

    clear_dir = "data/ldq/polyp/0-clear/";

    blur_dir = "data/ldq/polyp/1-blurry/"

    clear_images = os.listdir(clear_dir)

    blur_images = os.listdir(blur_dir)

    model.eval()

    missScore = 0

    with torch.no_grad():
        clear_scores = []
        clear_orb_scores=[]
        clear_blob_scores=[]

        for image in clear_images:
            path = clear_dir + image
            im = Image.open(path).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(image + ":" + str(score))
            clear_scores.append(score)
            clear_orb_scores.append(orb(path))
            clear_blob_scores.append(blobdetection(path))
            im.close()

        print("################")

        blur_scores = []
        blur_orb_scores = []
        blur_blob_scores=[]
        for image in blur_images:
            path = blur_dir + image
            im = Image.open(path).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(image + ":" + str(score))
            blur_scores.append(score)
            blur_orb_scores.append(orb(path))
            blur_blob_scores.append(blobdetection(path))
            im.close()

        SVM_TRAIN_RATIO = 0.6

        X, y = [], []

        for i in range(int(len(clear_scores) * SVM_TRAIN_RATIO)):
            X.append([clear_scores[i],clear_orb_scores[i]])
            y.append(0)

        for i in range(int(len(blur_scores) * SVM_TRAIN_RATIO)):
            X.append([blur_scores[i],blur_orb_scores[i]])
            y.append(1)

        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(X, y)

        Y_true=[]
        Y_pred=[]

        misPredicts = 0
        for i in range(int(len(clear_scores) * SVM_TRAIN_RATIO), len(clear_scores)):
            x = [[clear_scores[i],clear_blob_scores[i]]]
            y = clf.predict(x)
            Y_true.append(0)
            Y_pred.append(y)
            true_value = "Clear "
            pred_value = "Clear"
            if y[0] == 1:
                pred_value = "Blurry"
                misPredicts = misPredicts + 1

            print("Clear | score :" + str(x) + pred_value)

        for i in range(int(len(blur_scores) * SVM_TRAIN_RATIO), len(blur_scores)):
            x = [[blur_scores[i],blur_blob_scores[i]]]
            y = clf.predict(x)
            Y_true.append(1)
            Y_pred.append(y)
            true_value = "Blurry "
            pred_value = "Blurry"
            if y[0] == 0:
                pred_value = "Clear"
                misPredicts = misPredicts + 1

            print("Blurry | score :" + str(x) + pred_value)

        total = (len(blur_scores) - int(len(blur_scores) * SVM_TRAIN_RATIO))+\
        (len(clear_scores) - int(len(clear_scores) * SVM_TRAIN_RATIO))

        print("Total :" + str(total) + " Mispredicted :" +
              str(misPredicts))

        print("Accuracy :"+str(accuracy_score(Y_true, Y_pred)))
        roc_auc_score(Y_true, Y_pred)



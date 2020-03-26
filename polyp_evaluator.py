"""
Test Demo
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
 Date: 2018/5/26
"""


import torch
from PIL import Image
from IQADataset import NonOverlappingCropPatches
from CNNIQAnet import CNNIQAnet
import os




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    checkpoint = torch.load('checkpoints/CNNIQA-POLYP-OLD')

    model.load_state_dict(checkpoint)


    clear_dir =  "data/ldq/polyp/0-clear/";

    blur_dir =  "data/ldq/polyp/1-blurry/"

    clear_images = os.listdir(clear_dir)

    blur_images = os.listdir(blur_dir)


    model.eval()

    missScore=0

    with torch.no_grad():
        scores = []
        for image in clear_images:
            path = clear_dir + image
            im = Image.open(path).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(image +":"+str(score))
            if score > 30:
                missScore=missScore+1

        print("################")


        for image in blur_images:
            path = blur_dir + image
            im = Image.open(path).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(image+":"+str(score))
            if score < 30:
                missScore = missScore+1


        print("MissScore :"+str(missScore))




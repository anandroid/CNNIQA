"""
Test Cross Dataset
    For help
    ```bash
    python test_cross_dataset.py --help
    ```
 Date: 2018/5/26
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches
import numpy as np
import h5py, os
from CNNIQAnet import CNNIQAnet





if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test on the whole cross dataset')
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="dataset dir.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='models/CNNIQA-LIVE',
                        help="model file (default: models/CNNIQA-LIVE)")
    parser.add_argument("--save_path", type=str, default='scores',
                        help="save path (default: score)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    model.load_state_dict(torch.load(args.model_file))

    Info = h5py.File(args.names_info)
    im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()\
                        [::2].decode() for i in range(len(Info['im_names'][0, :]))]

    model.eval()
    with torch.no_grad(): 
        scores = []   
        for i in range(len(im_names)):
            im = Image.open(os.path.join(args.dataset_dir, im_names[i])).convert('L')
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(score)
            scores.append(score)
        np.save(args.save_path, scores)

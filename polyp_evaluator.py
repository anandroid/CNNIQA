"""
Test Demo
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
 Date: 2018/5/26
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches
from CNNIQAnet import CNNIQAnet




if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test demo')
    parser.add_argument("--im_path", type=str, default='data/I03_01_1.bmp',
                        help="image path")
    parser.add_argument("--model_file", type=str, default='models/CNNIQA-LIVE',
                        help="model file (default: models/CNNIQA-LIVE)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    model.load_state_dict(torch.load(args.model_file))
    '''

    model = torch.load('checkpoints/CNNIQA-POLYP-OLD')

    im = Image.open(args.im_path).convert('L')
    patches = NonOverlappingCropPatches(im, 32, 32)

    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))
        print(patch_scores.mean().item())

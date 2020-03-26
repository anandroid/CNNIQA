# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/19

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py
import scipy.io as sio


def default_loader(path):
    return Image.open(path).convert('L')  #


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


class PolypDataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = loadmat(datainfo)



        #Info = h5py.File(datainfo, 'r')

        ref_ids = Info['ref_ids']
        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']

        train_index, val_index, test_index = [], [], []


        for i in range(len(ref_ids)):
            if (i<497 and i>0) or (i<968 and i>=829):
                train_index.append(i)
                print("Train :"+str(i))
            elif (i>617 and i<829) or (i>1016 and i<=1062):
                val_index.append(i)
                print("Val :" + str(i))
            else:
                test_index.append(i)
                print("Test :" + str(i))






        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))


        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))

        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores']
        self.mos_std = Info['subjective_scoresSTD']
        im_names = Info['im_names']



        self.patches = ()
        self.label = []
        self.label_std = []
        print("Dmos scores")
        for idx in range(len(self.index)):
            #print("Preprocessing Image: {}".format(im_names[idx]))
            print(im_names[idx]+":"+str(self.mos[idx]))


            im = self.loader(os.path.join(im_dir, im_names[idx]))

            patches = NonOverlappingCropPatches(im, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + patches  #
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
                    self.label_std.append(self.mos_std[idx])
            else:
                self.patches = self.patches + (torch.stack(patches),)  #
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_std[idx]]))

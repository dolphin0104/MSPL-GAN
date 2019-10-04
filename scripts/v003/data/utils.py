"""
Data Utils
"""
import os
import random
import numpy as np
import cv2
import torch


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(apath):
    assert os.path.isdir(apath), '{:s} is not a valid directory'.format(apath)
    imgList = []
    for dirpath, _, fnames in sorted(os.walk(apath)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                imgList.append(img_path)
    assert imgList, '{:s} has no valid image file'.format(apath)
    return imgList


def img_resize(img, size=128, interp='cubic'):
    if img.ndim == 2:
        ih, iw = img.shape
    elif img.ndim == 3:
        ih, iw, _ = img.shape
    if ih != size or iw != size:
        # cubic | linear | area
        if interp == 'linear': 
            interp = cv2.INTER_LINEAR
        elif interp == 'cubic': 
            interp = cv2.INTER_CUBIC
        elif interp == 'area': 
            interp = cv2.INTER_AREA
        else: 
            raise NotImplementedError('interpolation [{:s}] not recognized'.format(interp))
        return cv2.resize(img, dsize=(size,size), interpolation=interp)
    else: return img


def get_patch(img, size=128):
    ih, iw = img.shape[:2]  
    
    ix = random.randrange(0, iw - size + 1)
    iy = random.randrange(0, ih - size + 1)
    
    img_patch = img[iy:iy + size, ix:ix + size, :]
    return img_patch


def img_randomcrop(img_in, size=128):
    img = np.copy(img_in)
    if img.ndim == 2:
        ih, iw = img.shape
    elif img.ndim == 3:
        ih, iw, _ = img.shape
    assert ih >= size, 'Crop sizes is larger than image size'
    assert iw >= size, 'Crop sizes is larger than image size'       
    ix = random.randrange(0, iw - size + 1)
    iy = random.randrange(0, ih - size + 1)
    img = img[iy:iy + size, ix:ix + size, :]
    return img


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)])
    return cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)
        return img
    return [_set_channel(_l) for _l in l]


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor
    return [_np2Tensor(_l) for _l in l]


def augment(l, hflip=True):
    hflip = hflip and random.random() < 0.5
    # vflip = rot and random.random() < 0.5
    # rot90 = rot and random.random() < 0.5
    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        # if vflip: img = img[::-1, :, :]
        # if rot90: img = img.transpose(1, 0, 2)        
        return img
    return [_augment(_l) for _l in l]


def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
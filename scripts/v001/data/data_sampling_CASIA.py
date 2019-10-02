import shutil
import os
import random

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def main(imgpath, savepath, n_valid=40):    
    if not os.path.exists(savepath): os.makedirs(savepath)

    trainpath = os.path.join(savepath, 'trainset', 'imgs')
    if not os.path.exists(trainpath): os.makedirs(trainpath)

    validpath = os.path.join(savepath, 'validset', 'imgs')
    if not os.path.exists(validpath): os.makedirs(validpath)
    
    trainset = []
    validset = []   

    imglist = get_paths_from_images(imgpath)
    random.seed(1234)
    random.shuffle(imglist)  
    for i, files in enumerate(imglist):
        if i < n_valid:
            validset.append(files)
        else:
            trainset.append(files)

    for i, files in enumerate(validset):
        filename = '{:0>7}.jpg'.format(i)
        print(filename)
        shutil.copy(files, os.path.join(validpath, filename))
    for i, files in enumerate(trainset):
        filename = '{:0>7}.jpg'.format(i)
        print(filename)
        shutil.copy(files, os.path.join(trainpath, filename))

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


if __name__ == '__main__':
    imgpath = 'D:/001.Benchmark_Datasets/FaceDatasets/CASIA-WebFace_Aligned_128/imgs'
    savepath = 'D:/001.Benchmark_Datasets/FaceDatasets/sampling_CASIA-WebFace_Aligned_128'
    n_valid = 40
    main(imgpath, savepath, n_valid)
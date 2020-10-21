# Progressive Semantic Face Deblurring

This repository is the Official Pytorch Implementation of [Progressive Semantic Face Deblurring]. 

![MSPL_GAN](/images/mspl_gan.png)

## Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## Train & Test
### Preapre dataset & vgg16 Face
+ Download training data and training kerles from here: MSPL_TrainingData [Google Drive](https://drive.google.com/drive/folders/1ZE5EAgYxW-KE0EGPGQfU8KHAv6qHV8gy?usp=sharing)
+ Extract 'train_img.zip' and 'train_label_cls4.zip' into the same folder path you want to specify.
```Example
- YOUR_TRAIN_DATA_PATH
    -- train_img
        -- files..
    -- train_label_cls4
        -- files..
```
+ Extract 'validationset.zip' to the folder path you want to specify.
+ Extract 'trainkernels.zip' to the folder path you want to specify. 
+ VGGFace16 trained model [Google Drive](https://drive.google.com/file/d/1MGSQpN-wsUe1EzADWSa13R7Czf00Xmmn/view?usp=sharing)


### Training
+ Specify the 'src/confgis/train_configs.py'.
```train_configs.py
line 10: cfg.GPU_IDS = [0]
    - Set the GPU ids you want to training. Default GPU id is 0.
    - Set to 'None' to train with CPU.
        cfg.GPU_IDS = None.
    - To train with multiple GPUs, specify the the GPU device ids. 
        cfg.GPU_IDS = [1,2] => The model will be trained with two GPUs(device ids(1,2)).

line 14: cfg.SAVE_DIR = '' 
    - Save result folder path you want to specify. 

line 37, 41: 'data_dir'
    - 'YOUR_TRAIN_DATA_PATH', 'YOUR_Validation_DATA_PATH' respectively.

line 48: cfg.TRAIN_KERNEL_PATH = 'YOUR_TrainKernel_DATA_PATH'

line 59: 'VGGFace16' = 'Path VGGFace16.pth"
```

+ To train the model(s) in the paper, run this command at 'src':
```train
python train.py
```


## Pre-trained Models
+ Pretrained model: MSPL_GAN [Google Drive](https://drive.google.com/drive/folders/1W55HWWkv3PhexuRBa9xCVjdC6WWcc5al?usp=sharing)


### Test
+ Specify the paths in 'src/inference.py'
```
MODEL_DIR = 'Trained model path'
INPUT_IMG_DIR = 'Test image path. Test all images in this path'
OUTPUT_IMG_DIR = 'Save result path'
USE_GPU = 'True of False. When set True, the model will be tested on 'GPU(cuda)' envirionment.'
```
+ To inference the model, run this command:
```inference
python inference.py
```


## MSPL_TestDatasets & Results
+ You can download here: MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1522V-vcngc48PdIKNEee0jVb3uGKMVpd?usp=sharing)
+ Test Restuls of ours on MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1mmK7qDhxOOehYCeTNMOTI0RhBj1HUDqx?usp=sharing)
+ Our model achieves the following performance on MSPL_Testsets :
+ PSNR and SSIM values are measured using MATLAB.

| MSPL_Center   | PSNR      | SSIM      | MSPL_Random   | PSNR      | SSIM      |
| ------------- | --------- | --------- | ------------- | --------- | --------- |
| CelebA        | 28.95     | 0.936     | CelebA        | 28.07     | 0.921     |
| CelebA-HQ     | 29.80     | 0.945     | CelebA-HQ     | 28.82     | 0.929     |
| FFHQ          | 29.22     | 0.941     | FFHQ          | 27.36     | 0.908     |


![result1](/images/fig1.png)

![result2](/images/fig2.png)

![result3](/images/fig3.png)

![result4](/images/fig4.png)

## Dataset Agreement
+ Our training and testset are synthesized using [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [FFHQ](https://github.com/NVlabs/ffhq-dataset).
+ The dataset is available for non-commercial research purposes only.
+ You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
+ You agree not to further copy, publish or distribute any portion of the dataset. 


## Related Datasets
+ CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
+ CelebAMask-HQ (https://github.com/switchablenorms/CelebAMask-HQ)
+ FFHQ (https://github.com/NVlabs/ffhq-dataset)

## Citations


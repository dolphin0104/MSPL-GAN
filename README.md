# Progressive Semantic Face Deblurring

This repository is the Official Pytorch Implementation of [Progressive Semantic Face Deblurring]. 

## Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## Train & Test
### Preapre dataset
+ Download training data from here: MSPL_TrainingData [Google Drive](https://drive.google.com/drive/folders/1ZE5EAgYxW-KE0EGPGQfU8KHAv6qHV8gy?usp=sharing)
 

### Training
+ Specify the 'src/confgis/train_configs.py'. See 'src/configs/README.MD' for details. 
+ To train the model(s) in the paper, run this command at 'src':
```train
python train.py
```

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

## Pre-trained Models
+ Pretrained model: MSPL_GAN [Google Drive](https://drive.google.com/drive/folders/1W55HWWkv3PhexuRBa9xCVjdC6WWcc5al?usp=sharing)


## MSPL_TestDatasets & Results
+ You can download here: MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1522V-vcngc48PdIKNEee0jVb3uGKMVpd?usp=sharing)
+ Test Restuls on MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1mmK7qDhxOOehYCeTNMOTI0RhBj1HUDqx?usp=sharing)
+ Our model achieves the following performance on MSPL_Testsets :
+ PSNR and SSIM values are measured using MATLAB.

| MSPL_Center   | PSNR      | SSIM      |
| ------------- | --------- | --------- |
| CelebA        | 28.07     | 0.921     |
| CelebA-HQ     | 28.82     | 0.929     |
| FFHQ          | 27.36     | 0.908     |

| MSPL_Random   | PSNR      | SSIM      |
| ------------- | --------- | --------- |
| CelebA        | 28.95     | 0.936     |
| CelebA-HQ     | 29.80     | 0.945     |
| FFHQ          | 29.22     | 0.941     |

## Dataset Agreemetn
+ Our training and testset are synthesized using [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [FFHQ](https://github.com/NVlabs/ffhq-dataset).
+ The dataset is available for non-commercial research purposes only.
+ You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
+ You agree not to further copy, publish or distribute any portion of the dataset. 


## Related Datasets
+ CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
+ CelebAMask-HQ (https://github.com/switchablenorms/CelebAMask-HQ)
+ FFHQ (https://github.com/NVlabs/ffhq-dataset)

## Citations


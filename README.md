# Progressive Semantic Face Deblurring

This repository is the Official Pytorch Implementation of [Progressive Semantic Face Deblurring]. 

## Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## Train
### Preapre dataset
+ Download training data from MSPL_TrainingData[Google Drive](https://drive.google.com/drive/folders/1ZE5EAgYxW-KE0EGPGQfU8KHAv6qHV8gy?usp=sharing)
 

### Training
+ Specify the 'src/confgis/train_configs.py'. See 'src/configs/README.MD' for details. 
+ To train the model(s) in the paper, run this command at 'src':
```train
python train.py
```

## Inference
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
+ You can download pretrained models here: MSPL_GAN [Google Drive](https://drive.google.com/drive/folders/1W55HWWkv3PhexuRBa9xCVjdC6WWcc5al?usp=sharing)

## MSPL_TestDatasets
+ You can download here: MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1522V-vcngc48PdIKNEee0jVb3uGKMVpd?usp=sharing)

## Results
+ Our model achieves the following performance on MSPL_Testsets:
+ PSNR and SSIM values are measured using MATLAB.
<!-- | Testset       | PSNR      | SSIM      |
| ------------- | --------- | --------- |
| MSPL_Center   |           |           |
| MSPL_Random   |           |           | -->

## Contributing

## Citations


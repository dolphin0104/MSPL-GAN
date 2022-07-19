# Progressive Semantic Face Deblurring

This repository is the Official Pytorch Implementation of [Progressive Semantic Face Deblurring](https://ieeexplore.ieee.org/abstract/document/9239928).
```
T. B. Lee, S. H. Jung and Y. S. Heo, "Progressive Semantic Face Deblurring," in IEEE Access, vol. 8, pp. 223548-223561, 2020, doi: 10.1109/ACCESS.2020.3033890.
```

![MSPL_GAN](/images/mspl_gan.png)

## 1. Dependencies
+ Python >= 3.5
+ Pytorch >= 1.2.0


## 2. Training
### 1) Prepre data
+ Download training data and training kerles from here: MSPL_TrainingData [Google Drive](https://drive.google.com/drive/folders/1ZE5EAgYxW-KE0EGPGQfU8KHAv6qHV8gy?usp=sharing)
+ Extract 'train_img.zip' and 'train_label_cls4.zip' into the same folder path you want to specify.
```Example
- /path/to/train_data (name is not important)
    -- train_img
        -- files..
    -- train_label_cls4
        -- files..
```
+ Extract 'validationset.zip' to the folder path you want to specify (/path/to/valid_data).
+ Extract 'trainkernels.zip' to the folder path you want to specify (/path/to/kernel_data).
+ Extract 'VGGFace16.zip' to the folder path you want to specify (/path/to/vggface).
    - VGGFace16 model weight is came from: [Github](https://github.com/ustclby/Unsupervised-Domain-Specific-Deblurring)
    - or you can download it from here: [Google Drive](https://drive.google.com/file/d/1MGSQpN-wsUe1EzADWSa13R7Czf00Xmmn/view?usp=sharing)


### 2) Training
+ Specify the 'src/confgis/train_configs.py'.
```train_configs.py
line 10: cfg.GPU_IDS = [0]
    - Set the GPU ids you want to training. Default GPU id is 0.
    - Set to 'None' to train with CPU.
        cfg.GPU_IDS = None.
    - To train with multiple GPUs, specify the the GPU device ids. 
        cfg.GPU_IDS = [1,2] => The model will be trained with two GPUs(device ids(1,2)).

line 14: cfg.SAVE_DIR = '/path/to/save' 
    - path to save models, training process, etc. 
    

line 37, 41: 'data_dir'
    - '/path/to/train_data', '/path/to/valid_data' respectively.

line 48: cfg.TRAIN_KERNEL_PATH = '/path/to/kernel_data'

line 59: 'VGGFace16' = 'Path VGGFace16.pth"
```

+ To train the model(s) in the paper, run this command in 'src':
```train
python train.py
```

## 3. Test
+ Specify the paths in 'src/inference.py'
```
MODEL_DIR = '/path/to/trained models/netG_last.pth'
INPUT_IMG_DIR = '/path/to/test image'
OUTPUT_IMG_DIR = '/path/to/save output image'
USE_GPU = 'True of False. When set True, the model will be tested on 'GPU(cuda)' envirionment.'
```
+ To inference the model, run this command in 'src':
```inference
python inference.py
```

## 4. Pre-trained Models
+ Pretrained model: MSPL_GAN [Google Drive](https://drive.google.com/drive/folders/1W55HWWkv3PhexuRBa9xCVjdC6WWcc5al?usp=sharing)


## 5. MSPL_Testsets & Results
+ You can download here: MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1522V-vcngc48PdIKNEee0jVb3uGKMVpd?usp=sharing)
+ Test Restuls of ours on MSPL_Testsets [Google Drive](https://drive.google.com/drive/folders/1mmK7qDhxOOehYCeTNMOTI0RhBj1HUDqx?usp=sharing)
+ Our model achieves the following performance on MSPL_Testsets :
+ PSNR and SSIM values are measured using MATLAB.

| MSPL_Center   | PSNR      | SSIM      | MSPL_Random   | PSNR      | SSIM      |
| ------------- | --------- | --------- | ------------- | --------- | --------- |
| CelebA        | 28.95     | 0.936     | CelebA        | 28.07     | 0.921     |
| CelebA-HQ     | 29.80     | 0.945     | CelebA-HQ     | 28.82     | 0.929     |
| FFHQ          | 29.22     | 0.941     | FFHQ          | 27.36     | 0.908     |


![result1](/images/fig1.PNG)
![result2](/images/fig2.PNG)
![result3](/images/fig3.png)
![result4](/images/fig4.PNG)


## 6. Dataset Agreement
+ Our training and testset are synthesized using [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [FFHQ](https://github.com/NVlabs/ffhq-dataset).
+ The dataset is available for non-commercial research purposes only.
+ You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
+ You agree not to further copy, publish or distribute any portion of the dataset. 


## 7. License, Acknowledgements
+ [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
+ [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
+ [FFHQ](https://github.com/NVlabs/ffhq-dataset)
+ [RealBlurrset](http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/)
+ Codes for MTCNN, MobienetArcface and weights came from: [HERE](https://github.com/TreB1eN/InsightFace_Pytorch)
+ Training & Testing kernels are synthesized using the method of,
    - Chakrabarti, Ayan. "A neural approach to blind motion deblurring." European conference on computer vision. Springer, Cham, 2016.



## 9. References
+ Xia, Zhihao, and Ayan Chakrabarti. "Training Image Estimators without Image Ground Truth." Advances in Neural Information Processing Systems. 2019.
+ Shen, Ziyi, et al. "Deep semantic face deblurring." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
+ Lu, Boyu, Jun-Cheng Chen, and Rama Chellappa. "Unsupervised domain-specific deblurring via disentangled representations." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
+ Yasarla, Rajeev, Federico Perazzi, and Vishal M. Patel. "Deblurring face images using uncertainty guided multi-stream semantic networks." IEEE Transactions on Image Processing 29 (2020): 6251-6263.


## 10. Citations
```
    @ARTICLE{lee2020progressive,
        author={T. B. {Lee} and S. H. {Jung} and Y. S. {Heo}},
        journal={IEEE Access}, 
        title={Progressive Semantic Face Deblurring}, 
        year={2020},
        volume={8},
        number={},
        pages={223548-223561},
        doi={10.1109/ACCESS.2020.3033890}}
```

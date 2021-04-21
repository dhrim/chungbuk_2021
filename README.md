# 충북대학교 딥러닝 특강

<br>
<br>

# 일자별 계획

## 1일차(2021/04/20)

- 딥러닝과 영상처리 : [deep_learning_and_image_processing.pptx](material/deep_learning/deep_learning_and_image_processing.pptx)


<br>

## 2일차(2021/04/22)

- Image Classification : [image_classification.ipynb](material/deep_learning/image_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/image_classification.ipynb)


<br>

## 3일차(2021/04/27)

- AutoEncoder 실습 : [autoencoder.ipynb](material/deep_learning/autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/autoencoder.ipynb)
- 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/denoising_autoencoder.ipynb)
- Super Resolution : [mnist_super_resolution.ipynb](material/deep_learning/mnist_super_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/mnist_super_resolution.ipynb)
- 이상 탐지 : [anomaly_detection_using_autoencoder.ipynb](material/deep_learning/anomaly_detection_using_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/anomaly_detection_using_autoencoder.ipynb)


<br>

## 4일차(2021/04/29)

- 영상 분할(Segementation)
    - U-Net을 사용한 영상 분할 실습 : [unet_segementation.ipynb](material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/unet_segementation.ipynb)
    - M-Net을 사용한 영상 분할 실습 : [mnet_segementation.ipynb](material/deep_learning/mnet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/mnet_segementation.ipynb)
    - U-Net을 사용한 컬러 영상 분할 실습 : [unet_segementation_color_image.ipynb](material/deep_learning/unet_segementation_color_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/chungbuk_2021/blob/master/material/deep_learning/unet_segementation_color_image.ipynb)


<br>

# 딥러닝 활용을 위한 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
    GCP virtual machine
linux
	ENV
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        load
        operation
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
	operation
        to numpy
    matplot
        draw line graph
        scatter
        show image

Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tuning
            overfitting, underfitting
            dropout, batch normalization, regularization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            vanilla, named CNN
        RNN
        GAN
    task
        Classification
        Object Detection
        Generation
	Segmentation
	Pose Extraction
	Noise Removing
	Super Resolution
	Question answering
	Auto Captioning
    data type
    	attribute data
	image data
	natural language data
	time series data

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate
        regularizer
        dropout
        batch normalization
    save/load
    compile
        optimizer
        loss
        metric
```

 

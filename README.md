Sparse-to-Dense
============================

This repo implements the training and testing of deep regression neural networks for ["Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"](https://arxiv.org/abs/1709.07492) by [Fangchang Ma](http://www.mit.edu/~fcma) and [Sertac Karaman](http://karaman.mit.edu/) at MIT. A video demonstration is available on [YouTube](https://youtu.be/vNIIT_M7x7Y).
<p align="center">
	<img src="http://www.mit.edu/~fcma/images/ICRA2018.png" alt="photo not available" width="50%" height="50%">
	<img src="https://j.gifs.com/Z4qDow.gif" alt="photo not available" height="50%">
</p>

## Contents
0. [Requirements](#requirements)
0. [Training](#training)
0. [Testing](#testing)
0. [Trained Models](#trained-models)
0. [Benchmark](#benchmark)
0. [Citation](#citation)


## Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU. 
- Install [cuDNN](https://developer.nvidia.com/cudnn)(v4 or above) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- If you already have both Torch and cuDNN installed, update `nn`, `cunn`, and `cudnn` packages.
	```bash
	luarocks install nn
	luarocks install cunn
	luarocks install cudnn
	```
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format libraries. Files in our pre-processed datasets are in HDF5 formats.
	```bash
	sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	git clone https://github.com/davek44/torch-hdf5.git
	cd torch-hdf5
	luarocks make
	```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and/or [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) datasets in HDF5 formats and place them under the `data` folder. The downloading process might take an hour or so. The NYU dataset requires 32G of storage space, and KITTI requires 81G.
	```bash
	cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
	tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz 
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz 
	cd ..
	```
- Download the networks pretrained on ImageNet datasets. In particular, use [ResNet-50](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7) for the NYU Depth V2 dataset, and [ResNet-18](https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7) for the KITTI dataset. Place them under the `pretrained` folder.
	```bash
	cd pretrained
	wget https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7
	wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
	cd ..
	```
## Training
The training scripts come with several options, which can be listed with the `--help` flag.
```bash
th main.lua --help
```

To run the training, simply run main.lua. By default, the script runs the RGB-based prediction network on NYU-Depth-V2 with 1 GPU and 2 data-loader threads without using pretrained weights.
```bash
th main.lua 
```

To train networks with different datasets, input modalities, loss functions, and components, see the example below:
```bash
th main.lua -dataset kitti -inputType rgbd -nSample 100 -criterion l1 -encoderType conv -decoderType upproj -pretrain true
```

Training results will be saved under the `results` folder.

#### Model Options
| Parameter     | Options     						| Remarks 		|
| ------------- | ----------- 						| -----------	|
| datasets     	| nyudepthv2,  kitti 				| 				|
| inputType     | rgb, rgbd, d, g, gd 				| d:sparse depth only; g: grayscale |
| nSample     	| non-negative integer (0 for rgb and g) | |
| criterion     | l1, l2, berhu         			| |
| pretrain      | false, true           			| |
| rep           | linear, log, inverse  			| representation of input depth |
| encoderType   | conv, depthsep, channeldrop  		| depthsep: depthwise separable convolution | 
| decoderType   | upproj, upconv, deconv2, deconv3  | deconv_n: transposed convolution with kernel size n-by-n | 

Please refer to "Deeper depth prediction with fully convolutional residual networks" for more details on `upproj` and `upconv` decoder modules.

## Testing
To test the performance of a trained model, simply run main.lua with the `-testOnly true` option, along with other model options. For instance,
```bash
th main.lua -testOnly true -dataset kitti -inputType rgbd -nSample 100 -criterion l1 -encoderType conv -decoderType upproj -pretrain true
```

## Trained Models
Download our trained models at http://datasets.lids.mit.edu/sparse-to-dense/results/ to the `results` folder. For instance,
```bash
cd results
wget -r -np -nH --cut-dirs=2 --reject "index.html*" http://datasets.lids.mit.edu/sparse-to-dense/results/nyudepthv2.input=rgbd.nsample=200.rep=linear.encoder=conv.decoder=upproj.criterion=l1.lr=0.01.bs=16.pretrained=true/
cd ..
```
More trained models will be released.

## Benchmark

- Error metrics on NYU Depth v2:

	| RGB     |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Roy & Todorovic](http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr16_NRF.pdf) (_CVPR 2016_) | 0.744 | 0.187 |  - | - | - |
	| [Eigen & Fergus](http://cs.nyu.edu/~deigen/dnl/) (_ICCV 2015_)  | 0.641 | 0.158 | 76.9 | 95.0 | 98.8 |
	| [Laina et al](https://arxiv.org/pdf/1606.00373.pdf) (_3DV 2016_)            | 0.573 | **0.127** | **81.1** | 95.3 | 98.8 |
	| Ours-RGB             | **0.514** | 0.143 | 81.0 | **95.9** | **98.9** |

	| RGBd-#samples   |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Liao et al](https://arxiv.org/abs/1611.02174) (_ICRA 2017_)-225 | 0.442 | 0.104 | 87.8 | 96.4 | 98.9 |
	| Ours-20 | 0.351 | 0.078 | 92.8 | 98.4 | 99.6 |
	| Ours-50 | 0.281 | 0.059 | 95.5 | 99.0 | 99.7 |
	| Ours-200| **0.230** | **0.044** | **97.1** | **99.4** | **99.8** |

	<img src="http://www.mit.edu/~fcma/images/ICRA18/acc_vs_samples_nyu.png" alt="photo not available" width="50%" height="50%">

- Error metrics on KITTI dataset:

	| RGB     |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Make3D](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) | 8.734 | 0.280 | 60.1 | 82.0 | 92.6 |
	| [Mancini et al](https://arxiv.org/pdf/1607.06349.pdf) (_IROS 2016_)  | 7.508 | - | 31.8 | 61.7 | 81.3 |
	| [Eigen et al](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) (_NIPS 2014_)  | 7.156 | **0.190** | **69.2** | 89.9 | **96.7** |
	| Ours-RGB             | **6.266** | 0.208 | 59.1 | **90.0** | 96.2 |

	| RGBd-#samples   |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Cadena et al](https://pdfs.semanticscholar.org/18d5/f0747a23706a344f1d15b032ea22795324fa.pdf) (_RSS 2016_)-650 | 7.14 | 0.179 | 70.9 | 88.8 | 95.6 |
	| Ours-50 | 4.884 | 0.109 | 87.1 | 95.2 | 97.9 |
	| [Liao et al](https://arxiv.org/abs/1611.02174) (_ICRA 2017_)-225 | 4.50 | 0.113 | 87.4 | 96.0 | 98.4 |
	| Ours-100 | 4.303 | 0.095 | 90.0 | 96.3 | 98.3 |
	| Ours-200 | 3.851 | 0.083 | 91.9 | 97.0 | 98.6 |
	| Ours-500| **3.378** | **0.073** | **93.5** | **97.6** | **98.9** |

	<img src="http://www.mit.edu/~fcma/images/ICRA18/acc_vs_samples_kitti.png" alt="photo not available" width="50%" height="50%">

	Note: our networks are trained on the KITTI odometry dataset, using only sparse labels from laser measurements.

## Citation 
If you use our code or method in your research, please cite:

	@article{Ma2017SparseToDense,
	  title={Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image},
	  author={Ma, Fangchang and Karaman, Sertac},
	  journal={arXiv preprint arXiv:1709.07492},
	  year={2017}
	}

Please direct any questions to [Fangchang Ma](http://www.mit.edu/~fcma) at fcma@mit.edu.


Sparse-to-Dense
============================

This repo implements the training and testing of deep regression neural networks for "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image" (available soon) by Fangchang Ma and Sertac Karaman at MIT.
<p align="center">
	<img src="http://www.mit.edu/~fcma/images/ICRA2018.png" alt="photo not available" width="50%" height="50%">
</p>

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
	git clone https://github.com/deepmind/torch-hdf5
	cd torch-hdf5
	luarocks make
	```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and/or [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) datasets in HDF5 formats. Place them under the `data` folder. (links available soon)
- Download the networks pretrained on ImageNet datasets. In particular, use [ResNet-50](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7) for the NYU Depth V2 dataset, and [ResNet-18](https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7) for the KITTI dataset. Place them under the `pretrained` folder.



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

## Testing

To test the performance of a trained model, simply run main.lua with the `-testOnly true` option, along with other model options. For instance,
```bash
th main.lua -testOnly true -dataset kitti -inputType rgbd -nSample 100 -criterion l1 -encoderType conv -decoderType upproj -pretrain true
```

## Trained models

Trained models will be provided soon.


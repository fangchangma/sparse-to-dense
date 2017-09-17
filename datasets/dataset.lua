--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  dataset parent class
--

local paths = require 'paths'
local ffi = require 'ffi'
local hdf5 = require 'hdf5'

local utils = require '../utils'

local M = {}
local Dataset = torch.class('resnet.Dataset', M)

function Dataset.createSparseDepthImage(depthImage, nSample)
   -- to be overridden by child class
   -- create sparse depth measurements 
   error('<datasets/dataset.lua> Dataset.createSparseDepthImage() should be overridden but is not.')
end   

function Dataset.loadFile(filename)
   -- to be overridden by child class
   error('<datasets/dataset.lua> Dataset.loadFile() should be overridden but is not.')
end

function Dataset.preprocess(rgbImage, depthImage, split)
   -- to be overridden by child class
   error('<datasets/dataset.lua> Dataset.preprocess() should be overridden but is not.')
end

function Dataset:getRawData(i)
   -- to be overridden by child class
   error('<datasets/dataset.lua> Dataset.getRawData() should be overridden but is not.')
end

function Dataset.addDimension(depthImage)
   -- add a singleton dimension, i.e., convert from 2-dim matrix (HxW) to 3-dim tensor (1xHxW)
   local outImage = torch.Tensor(1, depthImage:size(1), depthImage:size(2))
   outImage[{1, {}, {}}] = depthImage
   return outImage
end

function Dataset:createRgbdImage(rgbImage, depthImage)
   -- create input image with 4 channels: R, G, B, and sparse depth
   local outImage = torch.Tensor(4, rgbImage:size(2), rgbImage:size(3))
   outImage[{{1,3}, {}, {}}] = rgbImage
   outImage[{4, {}, {}}] = self.createSparseDepthImage(depthImage, self.opt.nSample)
   return outImage
end

function Dataset:createGdImage(grayscaleImage, depthImage)
   -- create input image with 4 channels: R, G, B, and sparse depth
   local outImage = torch.Tensor(2, grayscaleImage:size(2), grayscaleImage:size(3))
   outImage[{1, {}, {}}] = grayscaleImage
   outImage[{2, {}, {}}] = self.createSparseDepthImage(depthImage, self.opt.nSample)
   return outImage
end

function Dataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

-- local totalLoadtime = 0
-- local totalProcesstime = 0
-- local count = 0
function Dataset:get(i)
   -- count = count + 1
   local path = ffi.string(self.imageInfo.filePath[i]:data())
   -- local timer = torch.Timer()
   local rgbImage, depthImage = self.loadFile(paths.concat(self.dir, path))
   -- local loadtime = timer:time().real
   -- totalLoadtime = totalLoadtime + loadtime
   utils.setZeroToNan(depthImage)
   
   -- timer:reset()
   rgbImage, depthImage = self.preprocess(rgbImage, depthImage, self.split)
   -- local processtime = timer:time().real
   -- totalProcesstime = totalProcesstime + processtime
   utils.setZeroToNan(depthImage) -- deal with new zeros from preprocessing
   -- print(string.format('load=%.4f (%.4f), process=%.4f (%.4f)', loadtime, totalLoadtime/count,  processtime, totalProcesstime/count ))

   -- change input depth image representation
   local depthImagePreprocessed
   if self.opt.rep == 'linear' then
      -- depthImagePreprocessed = self.normalizeDepth(depthImage);
      depthImagePreprocessed = depthImage
   elseif self.opt.rep == 'log' then
      depthImagePreprocessed = torch.log(depthImage)
   elseif self.opt.rep == 'inverse' then
      depthImagePreprocessed = torch.pow(depthImage, -1)
   else
      error('<dataset.lua> invalid representation type: ' .. self.opt.rep)
   end

   local input, target
   if self.opt.inputType == 'rgb' then
      input = rgbImage
   elseif self.opt.inputType == 'g' then
      input = utils.rgb2gray(rgbImage)
   elseif self.opt.inputType == 'gd' then
      local grayscaleImage = utils.rgb2gray(rgbImage)
      input = self:createGdImage(grayscaleImage, depthImagePreprocessed)
   elseif self.opt.inputType == 'rgbd' then
      input = self:createRgbdImage(rgbImage, depthImagePreprocessed)
   elseif self.opt.inputType == 'd' then
      input = self.createSparseDepthImage(depthImagePreprocessed, self.opt.nSample)
   else
      error('invalid input type: ' .. self.opt.inputType)
   end
   -- input should not contain NaN values
   input[torch.ne(input, input)] = 0

   return {
      input = input,
      target = depthImage,
   }
end

function Dataset:size()
   return self.imageInfo.filePath:size(1)
end

-- Computed from random subset of ImageNet training images
Dataset.meanstdRGB = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
Dataset.pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

return M.Dataset

--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  NYU Depth V2 dataset loader
--

local image = require 'image'
local t = require 'datasets/transforms'

local NyuDepthV2Dataset = require 'datasets/dataset'

local oheight, owidth = 228, 304

function NyuDepthV2Dataset.loadFile(filename)
   local file = hdf5.open(filename, 'r')
   local rgbImage = file:read('/rgb'):all():float()
   local depthImage = file:read('/depth'):all()
   depthImage = NyuDepthV2Dataset.addDimension(depthImage)
   file:close()
   return rgbImage, depthImage
end

function NyuDepthV2Dataset.downsample(depthImage)
   return image.scale(depthImage, 160, 128)
end

function NyuDepthV2Dataset.createSparseDepthImage(depthImage, nSample)
   -- create sparse depth measurements 
   local randomMask = torch.zeros(1, depthImage:size(2), depthImage:size(3))
   local nPixels = depthImage:size(2) * depthImage:size(3)
   local percSample = nSample / nPixels
   randomMask:bernoulli(percSample)
   local sparseDepth = torch.cmul(depthImage, randomMask)
   return sparseDepth
end   

-- Computed from random subset (2000 images) of NYU Depth V2 training images
local meanstdDepth = {
   mean = 2.83617,
   std = 1.41854,
}

function NyuDepthV2Dataset.normalizeDepth(depthImage)
   local tDepth = t.Compose{
      t.DepthNormalize(meanstdDepth),
   }
   return tDepth(depthImage)
end

function NyuDepthV2Dataset.preprocess(rgbImage, depthImage, split)
   -- pre-processing function, including the following operations in order
   --    down-sampling: from 640×480 to 320×240
   --    rotation: rgb and depth are rotated by r ∈ [−5, 5] degrees.
   --    cropping: center crop to 304×228
   --    color jitter
   --    horizontal flip with probablity of 0.5
   rgbImage = rgbImage / 255

   if split == 'train' then 
      local s = torch.uniform(1.0, 1.5)   
      local degree = torch.uniform(-5.0, 5.0)
      
      local tRgb = t.Compose{
         t.Scale(torch.round(240*s)),  
         t.Rotation(degree),     
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, NyuDepthV2Dataset.pca.eigval, NyuDepthV2Dataset.pca.eigvec),
         t.ColorNormalize(NyuDepthV2Dataset.meanstdRGB),
         t.CenterCropRectangle(owidth, oheight), 
      }
      local tDepth = t.Compose{
         -- t.Scale(torch.round(240*s), 'simple'),  
         t.Scale(torch.round(240*s)),  
         t.Rotation(degree),
         -- t.DepthNormalize(meanstdDepth),
         t.CenterCropRectangle(owidth, oheight)   
      }

      rgbImage = tRgb(rgbImage)
      depthImage = tDepth(depthImage)
      depthImage:div(s)   -- element-wise division in-place

      if torch.uniform() < 0.5 then
         rgbImage = image.hflip(rgbImage)
         depthImage = image.hflip(depthImage)
      end

      -- if torch.uniform() < 0.5 then
      --    rgbImage = image.vflip(rgbImage)
      --    depthImage = image.vflip(depthImage)
      -- end

   elseif split == 'val' then
      local tRgb = t.Compose{
         t.Scale(240),  -- downsample 
         t.ColorNormalize(NyuDepthV2Dataset.meanstdRGB),
         t.CenterCropRectangle(owidth, oheight),
      }
      local tDepth = t.Compose{
         -- t.Scale(240, 'simple'),
         t.Scale(240),  
         -- t.DepthNormalize(meanstdDepth),
         t.CenterCropRectangle(owidth, oheight),
      }
      rgbImage = tRgb(rgbImage)
      depthImage = tDepth(depthImage)
   else   
      error('invalid split: ' .. split)
   end

   return rgbImage, depthImage
end

function NyuDepthV2Dataset.getRawData(i)
   local path = ffi.string(self.imageInfo.filePath[i]:data())

   local file = hdf5.open(paths.concat(self.dir, path), 'r')
   local rgbImage = file:read('/rgb'):all():transpose(2,3):float()
   local depthImage = file:read('/depth'):all():transpose(1,2)
   file:close()

   local tRgb = t.Compose{
      t.Scale(240),  -- downsample 
      -- t.ColorNormalize(NyuDepthV2Dataset.meanstdRGB),
      t.CenterCropRectangle(owidth, oheight),
   }
   local tDepth = t.Compose{
      -- t.Scale(240, 'simple'),
      t.Scale(240),  
      -- t.DepthNormalize(meanstdDepth),
      t.CenterCropRectangle(owidth, oheight),
   }
   rgbImage = tRgb(rgbImage)
   depthImage = tDepth(depthImage)

   return rgbImage, depthImage
end

return NyuDepthV2Dataset


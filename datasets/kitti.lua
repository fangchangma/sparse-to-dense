--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  KITTI Odometry dataset loader
--

local image = require 'image'
local t = require 'datasets/transforms'
local KittiDataset = require 'datasets/dataset'

local oheight, owidth = 228, 912

function KittiDataset.loadFile(filename)
   local file = hdf5.open(filename, 'r')
   -- local rgbImage = file:read('/rgb'):all():transpose(1,3):transpose(2,3):float()
   local rgbImage = file:read('/rgb'):all():float()
   local depthImage = file:read('/depth'):all()
   depthImage = KittiDataset.addDimension(depthImage)
   file:close()
   return rgbImage, depthImage
end

-- avoid repeated allocation and release of memory
local randomMask = nil
function KittiDataset.createSparseDepthImage(depthImage, nSample)
   -- create sparse depth measurements 
   if randomMask == nil or randomMask:nElement() ~= depthImage:nElement() then
      randomMask = torch.zeros(1, depthImage:size(2), depthImage:size(3)):typeAs(depthImage)
   end
   local nValidPixels = torch.sum(torch.gt(depthImage, 0))
   local percSample = nSample / nValidPixels
   percSample = percSample<1 and percSample or 1
   randomMask:bernoulli(percSample)
   local sparseDepth = torch.cmul(depthImage, randomMask)
   return sparseDepth
end   

function KittiDataset.preprocess(rgbImage, depthImage, split)
   -- pre-processing function, including the following operations in order
   -- note that the raw input size of KITTI dataset varies from 370×1226×3 to 376×1241×3
   --    bottom crop: cut the patch [130:370, 10:1210, :] from the raw input, reducing the size to 1200×240×3
   --    rotation: rgb and depth are rotated by r ∈ [−3, 3] degrees.
   --    cropping: center crop to 912×228
   --    color jitter
   --    horizontal flip with probablity of 0.5

   if split == 'train' then 
      local s = torch.uniform(1.0, 1.5)   
      local degree = torch.uniform(-5.0, 5.0)
      
      local tRgb = t.Compose{
         t.Crop(10, 1210, 130, 370),  
         t.Rotation(degree),     
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, KittiDataset.pca.eigval, KittiDataset.pca.eigvec),
         t.ColorNormalize(KittiDataset.meanstdRGB),
         t.CenterCropRectangle(owidth, oheight), 
      }
      local tDepth = t.Compose{
         t.Crop(10, 1210, 130, 370),    
         t.Rotation(degree),
         t.CenterCropRectangle(owidth, oheight)   
      }

      rgbImage = tRgb(rgbImage)
      depthImage = tDepth(depthImage)
      depthImage:div(s) 

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
         t.Crop(10, 1210, 130, 370),
         t.ColorNormalize(KittiDataset.meanstdRGB),
         t.CenterCropRectangle(owidth, oheight),
      }
      local tDepth = t.Compose{
         t.Crop(10, 1210, 130, 370), 
         t.CenterCropRectangle(owidth, oheight),
      }
      rgbImage = tRgb(rgbImage)
      depthImage = tDepth(depthImage)
   else   
      error('invalid split: ' .. split)
   end

   return rgbImage, depthImage
end

return KittiDataset


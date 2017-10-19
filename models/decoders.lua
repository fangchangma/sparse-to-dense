--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'modules/Unpool'

local decoders = {}

local Convolution = cudnn.SpatialConvolution
local DeConvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization
local DilatedConvolution = cunn.SpatialDilatedConvolution
-- local Avg = cudnn.SpatialAveragePooling
-- local Max = nn.SpatialMaxPooling
-- local SpatialReSampling = nn.DontCast(nn.SpatialReSampling(), true, true)

-- 2×2 deconvolution layer, each output pixel has a reception field of 2×2
function decoders.deConv2(nInputPlane, nOutputPlane)
   local model = nn.Sequential()
   local kW, kH = 2, 2
   local dW, dH = 2, 2
   local padW, padH = 0, 0
   local adjW, adjH = 0, 0
   -- formula:
   --  owidth  = (width  - 1) * dW - 2*padW + kW + adjW = 2 * width
   --  oheight = (height - 1) * dH - 2*padH + kH + adjH = 2 * height
   model:add(DeConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH))
   model:add(SBatchNorm(nOutputPlane))
   model:add(ReLU(true))

   return model
end

-- 3×3 deconvolution layer, each output pixel has a reception field of 3×3
function decoders.deConv3(nInputPlane, nOutputPlane)
   local model = nn.Sequential()
   local kW, kH = 3, 3
   local dW, dH = 2, 2
   local padW, padH = 1, 1
   local adjW, adjH = 1, 1
   -- formula:
   --  owidth  = (width  - 1) * dW - 2*padW + kW + adjW = 2 * width
   --  oheight = (height - 1) * dH - 2*padH + kH + adjH = 2 * height
   model:add(DeConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH))
   model:add(SBatchNorm(nOutputPlane))
   model:add(ReLU(true))

   return model
end

-- up-Convolution, described in "Deeper Depth Prediction with Fully Convolutional Residual Networks", 2016
--    each output pixel has a reception field of 3×3
function decoders.upConv(nInputPlane, nOutputPlane, iheight, iwidth)
   local model = nn.Sequential()

   -- double the size of feature maps
   model:add(nn.Unpool(nInputPlane, nInputPlane, iheight, iwidth, 2))

   -- 5×5 convolution, reduce the number of channels to half
   local kW, kH = 5, 5
   local dW, dH = 1, 1
   local padW = (kW-1)/2
   local padH = (kH-1)/2
   model:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
   model:add(SBatchNorm(nOutputPlane))
   model:add(ReLU(true))

   return model
end

-- up-Projection, described in "Deeper Depth Prediction with Fully Convolutional Residual Networks", 2016
function decoders.upProj(nInputPlane, nOutputPlane, iheight, iwidth)
   local model = nn.Sequential()

   model:add(nn.Unpool(nInputPlane, nInputPlane, iheight, iwidth, 2))

   local s = nn.Sequential()
   local kW, kH = 5, 5
   local dW, dH = 1, 1
   local padW = (kW-1)/2
   local padH = (kH-1)/2
   s:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
   s:add(SBatchNorm(nOutputPlane))
   s:add(ReLU(true))
   s:add(Convolution(nOutputPlane,nOutputPlane,3,3,1,1,1,1))
   s:add(SBatchNorm(nOutputPlane)) 

   local shortcut = nn.Sequential()
   shortcut:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
   shortcut:add(SBatchNorm(nOutputPlane))

   model:add(nn.ConcatTable()
      :add(s)
      :add(shortcut))
   model:add(nn.CAddTable(true))
   model:add(ReLU(true))

   return model
end

return decoders
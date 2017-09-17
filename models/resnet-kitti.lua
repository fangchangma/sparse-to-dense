--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'
local encoders = require 'models/encoders'
local decoders = require 'models/decoders'


local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
-- local DeConvolution = cudnn.SpatialFullConvolution

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   
   -- Configurations for ResNet:
   --  num. residual blocks, num features, residual block function
   local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      -- [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
   }

   assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))

   local def, nFeatures, block = table.unpack(cfg[depth])
   iChannels = 64
   print(' | resnet-' .. opt.dataset .. '-' .. depth)

   -- Encoder
   local encoderLayer = nil
   local numInputChannels = opt.inputType:len()
   assert(numInputChannels<=4, 'invalid input type: ' .. opt.inputType)

   if opt.encoderType == 'conv' then
      encoderLayer = encoders.conv
   elseif opt.encoderType == 'channeldrop' then
      encoderLayer = encoders.channelDrop
   elseif opt.encoderType == 'depthsep' then
      encoderLayer = encoders.depthSep
   else
      error('<resnet-kitti.lua> unknown encoder type: ' .. opt.encoderType)
   end

   model:add(encoderLayer(numInputChannels, 64))

   -- ResNet
   model:add(SBatchNorm(64))
   model:add(ReLU(true))
   model:add(Max(3,3,2,2,1,1))
   model:add(layer(block, 64, def[1]))
   model:add(layer(block, 128, def[2], 2))
   model:add(layer(block, 256, def[3], 2))
   model:add(layer(block, 512, def[4], 2))
   -- output: 29×8×2048
   
   -- 1×1 convolution and batch normalization: output 29×8×1024
   -- model:add(Convolution(2048,1024,1,1,1,1,0,0))
   -- model:add(SBatchNorm(1024))
   model:add(Convolution(512,256,1,1,1,1,0,0))
   model:add(SBatchNorm(256))
   -- model:add(ReLU(true))   -- should we add a ReLu layer here?

   -- Decoder 
   local decoderLayer = nil
   if opt.decoderType == 'deconv2' then
      decoderLayer = decoders.deConv2
   elseif opt.decoderType == 'deconv3' then
      decoderLayer = decoders.deConv3
   elseif opt.decoderType == 'upconv' then 
      decoderLayer = decoders.upConv
   elseif opt.decoderType == 'upproj' then 
      decoderLayer = decoders.upProj
   elseif opt.decoderType == 'upsample' then 
      decoderLayer = decoders.upSample
   else
      error('<resnet-kitti.lua> unknown decoder type: ' .. opt.decoderType)
   end

   -- -- decoder layer 1: output 58×16×512
   -- -- decoder layer 2: output 116×32×256
   -- -- decoder layer 3: output 232×64×128
   -- -- decoder layer 4: output 464×128×64
   -- local nInputPlane, nOutputPlane = 1024, 512

   -- decoder layer 1: output 58×16×128
   -- decoder layer 2: output 116×32×64
   -- decoder layer 3: output 232×64×32
   -- decoder layer 4: output 464×128×16
   local nInputPlane, nOutputPlane = 256, 128

   local iheight, iwidth = 8, 29
   for i = 1, 4 do
      model:add(decoderLayer(nInputPlane, nOutputPlane, iheight, iwidth))
      nInputPlane = nInputPlane / 2
      nOutputPlane = nOutputPlane / 2
      iheight = iheight * 2
      iwidth = iwidth * 2
   end

   -- 3×3 convolution: output 464×128×1
   model:add(Convolution(nInputPlane,1,3,3,1,1,1,1))
   model:add(nn.SpatialUpSamplingBilinear{owidth=912,oheight=228})


   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   ConvInit('nn.SpatialFullConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   -- Load pretrained model and update parameters for layers 5 - 8
   if opt.pretrain then
      print(' | Loading pretrained model ' .. opt.pretrainedPath)
      local pretrainedModel = torch.load(opt.pretrainedPath)
      if opt.inputType == 'rgb' and opt.encoderType == 'conv' then
         print(' | Loading pretrained module ' .. 1)
         model.modules[1]=pretrainedModel.modules[1]
      else
         print(' | Skip loading pretrained module 1')
      end
      for i = 2, 8 do
         print(' | Loading pretrained module ' .. i)
         model.modules[i]=pretrainedModel.modules[i]
      end
      pretrainedModel = nil
      collectgarbage()
   end

   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel

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
require 'modules/ChannelDropout'

local encoders = {}

local Convolution = cudnn.SpatialConvolution
local depthWiseConv = nn.SpatialDepthWiseConvolution

-- The default convolution layer
function encoders.conv(nInputPlane, nOutputPlane)
   return Convolution(nInputPlane,nOutputPlane,7,7,2,2,3,3)
end

-- The Channel-Dropout layer
function encoders.channelDrop(nInputPlane, nOutputPlane)
   local model = nn.Sequential()
   model:add(nn.ChannelDropout(nInputPlane, 0.5))
   model:add(Convolution(nInputPlane,nOutputPlane,7,7,2,2,3,3))
   return model
end

-- The Depthwise Spatial Separable layer
-- "Xception: Deep Learning with Depthwise Separable Convolutions"
function encoders.depthSep(nInputPlane, nOutputPlane)
   local model = nn.Sequential()
   model:add(depthWiseConv(nInputPlane,1,7,7,2,2,3,3))
   model:add(Convolution(nInputPlane,nOutputPlane,1,1))
   return model
end

return encoders
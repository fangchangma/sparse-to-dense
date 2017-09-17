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

local Unpool, Parent = torch.class('nn.Unpool', 'nn.Module')

function Unpool:__init(nInputPlane, nOutputPlane, iheight, iwidth, k)
   Parent.__init(self)
   if not nInputPlane or nInputPlane < 1 then
      error('<Unpool> illegal nInputPlane, must be nInputPlane >= 1')
   end
   assert(nInputPlane==nOutputPlane, "<Unpool> nInputPlane must be equal to nOutputPlane")
   if not k or k < 1 then
      error('<Unpool> illegal number of unpooling, must be k >= 1')
   end

   self.nInputPlane = nInputPlane
   self.k = k
   self.oheight = k*iheight
   self.owidth = k*iwidth

   self.upsample = nn.SpatialUpSamplingNearest(k)
end

function Unpool:updateOutput(input)
   local batchsize = input:size(1)

   -- creation of self.maask needs to be done for only once, and has to be done after model:cuda()
   -- This is because :cuda() command would convert self.mask into CudaTensor, but we need CudaByteTensor
   if not self.mask or batchsize > self.mask:size(1) then
      -- create mask
      local mask = torch.CudaByteTensor(self.oheight, self.owidth):zero()
      for i = 1, self.oheight, self.k do
         for j = 1, self.owidth, self.k do
            mask[i][j] = 1
         end
      end
      
      self.mask = torch.CudaByteTensor(batchsize, self.nInputPlane, self.oheight, self.owidth):zero()
      for i = 1, batchsize do
         for j = 1, self.nInputPlane do
            self.mask[{i, j, {}, {}}] = mask
         end
      end

      self.mask = torch.ne(self.mask, 1)
   end

   self.output = self.upsample:forward(input)

   -- handles varying batch size 
   if batchsize < self.mask:size(1) then
      self.output[self.mask[{{1, batchsize}, {}, {}, {}}]] = 0
   else
      self.output[self.mask] = 0
   end
   return self.output
end

function Unpool:updateGradInput(input, gradOutput)
   local batchsize = input:size(1)
   local gradInput = gradOutput:clone()

   -- handles varying batch size 
   if batchsize < self.mask:size(1) then
      gradInput[self.mask[{{1, batchsize}, {}, {}, {}}]] = 0 
   else
      gradInput[self.mask] = 0 
   end

   self.gradInput = self.upsample:backward(input, gradInput)  
   return self.gradInput
end

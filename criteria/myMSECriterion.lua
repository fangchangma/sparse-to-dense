--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--
--  A slightly modified version of the MSECriterion, where pixles with Nan values in the target will be ignored.
-- 
require 'nn'
local utils = require '../utils'

local myMSECriterion, parent = torch.class('nn.myMSECriterion', 'nn.MSECriterion')

function myMSECriterion:__init(sizeAverage)
   parent.__init(self) 

   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

-- forward pass
function myMSECriterion:updateOutput(input, target)
   assert( input:nElement() == target:nElement(), "input and target size mismatch")

   local _input, _target, nanMask, nValidElement = utils.setNanToZero(input, target)

   self.output = parent.updateOutput(self, _input, _target) -- not normalized

   if self.sizeAverage then
      -- all pixels in target are nan
      if nValidElement == 0 then
         self.output = 0
      -- target contains at least 1 valid pixel
      else
         self.output = self.output * _target:nElement() / nValidElement
      end 
   else
      self.output = parentOutput * _target:nElement()
   end

   return self.output
end

-- backward pass
function myMSECriterion:updateGradInput(input, target)
   assert( input:nElement() == target:nElement(), "input and target size mismatch")

   local _input, _target, nanMask, nValidElement = utils.setNanToZero(input, target)

   self.gradInput = parent.updateGradInput(self, _input, _target) -- not normalized
   self.gradInput[nanMask] = 0

   return self.gradInput
end
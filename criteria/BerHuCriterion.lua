--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--
--  The reverse Huber (berHu) loss function
-- 
--  B(x) = |x|,               if |x| <= c
--       (x^2 + c^2) / (2c),  if |x| > c
--         0,                 if x == nan
--  where c = 0.2 * max_i (|\hat{y}_i - y_i|), i.e., 20% of the maximal per-batch error
-- 

require 'nn'
local utils = require '../utils'

local BerHuCriterion, parent = torch.class('nn.BerHuCriterion', 'nn.Criterion')

function BerHuCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

-- forward pass
function BerHuCriterion:updateOutput(input, target)
   assert( input:nElement() == target:nElement(), "input and target size mismatch")
   
   local _input, _target, nanMask, nValidElement = utils.setNanToZero(input, target)

   local absDiff = torch.abs(_input - _target)
   self.c = torch.max(absDiff) / 5
   self.maskLessThanC = torch.le(absDiff, self.c):typeAs(target)
   self.maskGreaterThanC = torch.gt(absDiff, self.c):typeAs(target)

   if self.c > 0 then   
      local outputGreaterThanC = (torch.pow(absDiff, 2) + math.pow(self.c, 2)) / (2 * self.c)
      local outputTensor = torch.cmul(absDiff, self.maskLessThanC) + torch.cmul(outputGreaterThanC, self.maskGreaterThanC)
      self.output = torch.sum(outputTensor)
      outputGreaterThanC = nil
      collectgarbage()

      if self.sizeAverage then
         -- all pixels in target are nan
         if nValidElement == 0 then
            self.output = 0
         -- target contains at least 1 valid pixel
         else
            self.output = self.output / nValidElement
         end
      end

   -- perfect prediction
   else
      self.output = 0
   end

   return self.output
end

-- backward pass
-- grad(x) = x / c,  if |x| > c
--           -1,     if -c <= x < 0
--            1,     if 0 < x <= c
--            0,     if x == 0 or x == nan
function BerHuCriterion:updateGradInput(input, target)
   assert( input:nElement() == target:nElement(), "input and target size mismatch")
   local _input, _target, nanMask, nValidElement = utils.setNanToZero(input, target)

   if self.c > 0 then
      local diff = _input - _target 
      local gradLessThanC = torch.gt(diff, 0):typeAs(target) - torch.lt(diff, 0):typeAs(target)
      self.gradInput = torch.cmul(gradLessThanC, self.maskLessThanC) 
         + torch.cmul(torch.div(diff, self.c), self.maskGreaterThanC)  
      gradLessThanC = nil
      diff = nil
      collectgarbage()

      if self.sizeAverage then
         -- all pixels in target are nan
         if nValidElement == 0 then
            self.gradInput = input:zero()
         -- target contains at least 1 valid pixel
         else
            self.gradInput:div(nValidElement)
         end
      end

      self.gradInput[nanMask] = 0

   -- perfect prediction
   else
      self.gradInput = input:zero()
   end

   return self.gradInput
end
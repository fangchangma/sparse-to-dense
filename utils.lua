--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require "image"
require 'nn'

local utils = {}

function utils.lg10(x)
   return torch.div(torch.log(x), math.log(10))
end

-- return the total number of valid pixels in a batch
function utils.nValidElement(x)
   return torch.sum(torch.eq(x, x))
end

-- return the total number of nan in a batch
function utils.nNanElement(x)
   return torch.sum(torch.ne(x, x))
end

function utils.getNanMask(x)
   return torch.ne(x, x)  -- since NaN != NaN in torch
end

function utils.maxOfTwo(x, y)
   local z = x:clone()
   local maskYLarger = torch.lt(x, y)  -- true if x < y
   z[maskYLarger] = y[maskYLarger]
   return z
end

-- return two new tensors, where pixels corresponding to NaN in target are set to 0
function utils.setNanToZero(input, target)
   local nanMask = utils.getNanMask(target)
   nValidElement = utils.nValidElement(target)

   local _input = input:clone()
   _input[nanMask] = 0

   local _target = target:clone()
   _target[nanMask] = 0

   return _input, _target, nanMask, nValidElement
end

function utils.setNanTo0(input)
   local nanMask = utils.getNanMask(input)
   local output = input:clone()
   output[nanMask] = 0
   return output
end

function utils.setZeroToNan(input)
   input[torch.le(input, 0)] = 0/0
end

-- permute order of dimensions, since the order is different in Matlab vs. Torch
function utils.matlabToTorchTensor(tensorMatlab)
   local tensorTorch = torch.Tensor(tensorMatlab:size(3), tensorMatlab:size(1), tensorMatlab:size(2))
   for channel = 1, tensorMatlab:size(3) do
      tensorTorch[{channel, {}, {}}] = tensorMatlab[{{}, {}, channel}]
   end
   return tensorTorch
end

function utils.rgb2gray(rgb)
   assert( rgb:dim() == 3, "expect 3-dim rgb input, but instead got input with dimension=" .. rgb:dim() )
   local z = torch.Tensor(1, rgb:size(2), rgb:size(3))
   -- Same as Matlab: 0.2989 * R + 0.587 * G + 0.114 * B 
   z[{1, {}, {}}] = rgb:select(1, 1) * 0.2989 + rgb:select(1, 2) * 0.587 + rgb:select(1, 3) * 0.114
   -- z = z:round()
   return z
end  

function utils.getGradient(input)
   -- local laplacianKernel = image.laplacian(3, 1)
   local kernel = torch.Tensor(1, 3, 3):typeAs(input)
   kernel = kernel:zero()
   kernel[1][2][1] = -1
   kernel[1][2][3] = 1
   kernel[1][1][1] = -1
   kernel[1][3][1] = 1
   local gradient = torch.Tensor(input:size()):typeAs(input)
   image.convolve(gradient, input, kernel, 'same')
   return gradient
end 

function utils.getGradientCuda(input)
   local s = nn.Sequential()
   local layer = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
   layer.bias:zero()
   layer.weight:zero()
   -- interal filter
   -- [0, -1, 0]
   -- [-1, 0, 1]
   -- [0,  1, 0]
   layer.weight[1][1][2][1] = -1
   layer.weight[1][1][2][3] = 1
   layer.weight[1][1][1][2] = -1
   layer.weight[1][1][3][2] = 1
   s:add(layer)

   local gradient = s:forward(input)
   return gradient
   
   -- -- local laplacianKernel = image.laplacian(3, 1)
   -- local kernel = torch.Tensor(1, 3, 3):typeAs(input)
   -- kernel = kernel:zero()
   -- kernel[1][2][1] = -1
   -- kernel[1][2][3] = 1
   -- kernel[1][1][1] = -1
   -- kernel[1][3][1] = 1
   -- local gradient = torch.Tensor(input:size()):typeAs(input)
   -- image.convolve(gradient, input, kernel, 'same')
   -- return gradient
end 

function utils.evaluateError(output, target)
   assert( output, "output is nil")
   assert( target, "target is nil")
   assert( output:dim() == 4, "expect 4 dimensions in output, but instead got " .. output:dim())
   assert( target:dim() == 4, "expect 4 dimensions in target, but instead got " .. target:dim())
   assert( output:size(1) >= 1, 'invalid output batch size: ' .. output:size(1))
   assert( target:size(1) >= 1, 'invalid target batch size: ' .. target:size(1))

   local batchSize = output:size(1)

   local errors = {
      MSE = 0,
      RMSE = 0,
      ABS_REL = 0,
      LG10 = 0,
      MAE = 0,
      PERC = 0,   -- % of correct edge prediction for hybrid output

      -- DELTA_i: % of pixels s.t. max(y_i / z_i, z_i / y_i) < DELTA^i, where DELTA = 1.25
      DELTA1 = 0, 
      DELTA2 = 0,
      DELTA3 = 0,
   }

   if output:size(2) == 1 then
      _output = output:view(batchSize, -1)
      _target = target:view(batchSize, -1) 
      _output, _target, nanMask, nValidElement = utils.setNanToZero(_output, _target)

      if nValidElement > 0 then
         -- Compute the difference
         local diffMatrix = torch.abs(_output - _target)

         -- Mean Squared Error
         errors.MSE = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

         -- Root Mean Squared Error
         -- This overestimates the RMSE when batchsize > 1. Use batchsize=1 to get an accurate estimation
         errors.RMSE = math.sqrt(errors.MSE)

         -- Mean Absolute Error
         errors.MAE = torch.sum(diffMatrix) / nValidElement

         -- Mean Absolute Relative Error
         local relMatrix = torch.cdiv(diffMatrix, _target)
         relMatrix[nanMask] = 0
         errors.ABS_REL = torch.sum(relMatrix) / nValidElement

         -- release memory
         relMatrix = nil
         diffMatrix = nil
         collectgarbage()

         -- LOG10 Error
         -- nan when prediction is negative
         -- print('#(output<=0) = ' .. torch.sum(torch.le(output, 0)))
         
         local LG10Matrix = torch.abs(utils.lg10(_output) - utils.lg10(_target))
         LG10Matrix[nanMask] = 0
         errors.LG10 = torch.sum(LG10Matrix) / nValidElement
         LG10Matrix = nil
         collectgarbage()

         local yOverZ = torch.cdiv(_output, _target)
         local zOverY = torch.cdiv(_target, _output)
         local maxRatio = utils.maxOfTwo(yOverZ, zOverY)
         errors.DELTA1 = torch.sum(torch.le(maxRatio, 1.25):typeAs(output)) / nValidElement
         errors.DELTA2 = torch.sum(torch.le(maxRatio, math.pow(1.25,2)):typeAs(output)) / nValidElement
         errors.DELTA3 = torch.sum(torch.le(maxRatio, math.pow(1.25,3)):typeAs(output)) / nValidElement
      end
      
      return errors

   -- hybrid output
   elseif output:size(2) == 3 then
      error('TODO: implement utils.evaluateError() for hybrid output')

      local depthOutput = torch.Tensor(batchSize, 1, output:size(3), output:size(4)):typeAs(output)
      local depthTarget = torch.Tensor(batchSize, 1, output:size(3), output:size(4)):typeAs(target)
      depthOutput[{{}, 1}] = output[{{}, 1}]
      depthTarget[{{}, 1}] = target[{{}, 1}]

      local predictionEdge = torch.gt(output[{{}, 3}], output[{{}, 2}]):typeAs(output) + 1
      local predictionCorrect = torch.eq(predictionEdge, target[{{}, 2}])
      local correctPercentage = torch.sum(predictionCorrect) / predictionCorrect:nElement()

      errors = utils.evaluateError(depthOutput, depthTarget)
      -- return mse, rmse, relErr, LG10Err, mae, correctPercentage
   else
      error('unknown prediction type')
   end
end

function utils.addErrors(errorSum, errors, batchSize)
   errorSum.MSE = errorSum.MSE + errors.MSE * batchSize
   errorSum.RMSE = errorSum.RMSE + errors.RMSE * batchSize
   errorSum.ABS_REL = errorSum.ABS_REL + errors.ABS_REL * batchSize
   errorSum.LG10 = errorSum.LG10 + errors.LG10 * batchSize
   errorSum.MAE = errorSum.MAE + errors.MAE * batchSize
   errorSum.PERC = errorSum.PERC + errors.PERC * batchSize
   errorSum.DELTA1 = errorSum.DELTA1 + errors.DELTA1 * batchSize
   errorSum.DELTA2 = errorSum.DELTA2 + errors.DELTA2 * batchSize
   errorSum.DELTA3 = errorSum.DELTA3 + errors.DELTA3 * batchSize
end

function utils.averageErrors(errorSum, N)
   assert(N > 0, 'N must be positive, but instead got ' .. N)
   errorSum.MSE = errorSum.MSE / N
   errorSum.RMSE = errorSum.RMSE / N
   errorSum.ABS_REL = errorSum.ABS_REL / N
   errorSum.LG10 = errorSum.LG10 / N
   errorSum.MAE = errorSum.MAE / N
   errorSum.PERC = errorSum.PERC / N
   errorSum.DELTA1 = errorSum.DELTA1 / N
   errorSum.DELTA2 = errorSum.DELTA2 / N
   errorSum.DELTA3 = errorSum.DELTA3 / N
end

return utils
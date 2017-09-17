--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
local optim = require 'optim'
local utils = require 'utils'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer, dataTimer = torch.Timer(), torch.Timer()
   local totalTime, totalDataTime = 0, 0

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local errorSum = {
      MSE = 0,
      RMSE = 0,
      ABS_REL = 0,
      LG10 = 0,
      MAE = 0,
      PERC = 0,
      DELTA1 = 0, 
      DELTA2 = 0,
      DELTA3 = 0,
   }
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      totalDataTime = totalDataTime + dataTime

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      N = N + batchSize
      local time = timer:time().real
      totalTime = totalTime + time

      local errors = utils.evaluateError(output, sample.target:cuda())
      utils.addErrors(errorSum, errors, batchSize)
      print('=> ' .. self.opt.saveDir)
      print((' | %4s - Epoch: [%d][%d/%d]\t\tLR=%1.6f\n'
         .. '\tTime=%.3f(%.3f)\tData=%.3f(%.3f)\n'
         .. '\tMSE=%1.3f(%1.3f)\tRMSE=%1.3f(%1.3f)\tMAE=%1.3f(%1.3f)\n' 
         .. '\tDELTA1=%1.3f(%1.3f)\tDELTA2=%1.3f(%1.3f)\tDELTA3=%1.3f(%1.3f)\n'
         .. '\tREL=%1.3f(%1.3f)\tLG10=%1.3f(%1.3f)'
         ):format(
         self.opt.inputType, epoch, n, trainSize, 
         self.optimState.learningRate, 
         time, totalTime * batchSize / N, 
         dataTime, totalDataTime * batchSize / N,
         errors.MSE, errorSum.MSE / N, 
         errors.RMSE, errorSum.RMSE / N, 
         errors.MAE, errorSum.MAE / N,
         errors.DELTA1, errorSum.DELTA1 / N,
         errors.DELTA2, errorSum.DELTA2 / N,
         errors.DELTA3, errorSum.DELTA3 / N,
         errors.ABS_REL, errorSum.ABS_REL / N, 
         errors.LG10, errorSum.LG10 / N))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   utils.averageErrors(errorSum, N)
   return errorSum 
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()
   -- local size = dataloader:size() / 6 -- for efficiency, roughly 7.6k samples

   local nCrops = self.opt.tenCrop and 10 or 1
   local errorSum = {
      MSE = 0,
      RMSE = 0,
      ABS_REL = 0,
      LG10 = 0,
      MAE = 0,
      PERC = 0,
      DELTA1 = 0, 
      DELTA2 = 0,
      DELTA3 = 0,
   }
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      local batchSize = output:size(1) / nCrops

      N = N + batchSize
      local errors = utils.evaluateError(output, sample.target:cuda())
      utils.addErrors(errorSum, errors, batchSize)

      print('=> ' .. self.opt.saveDir)
      print((' | %4s - Test: [%d][%d/%d]\n'
         .. '\tTime=%.3f\tData=%.3f\n' 
         .. '\tMSE=%1.3f(%1.3f)\tRMSE=%1.3f(%1.3f)\tMAE=%1.3f(%1.3f)\n' 
         .. '\tDELTA1=%1.3f(%1.3f)\tDELTA2=%1.3f(%1.3f)\tDELTA3=%1.3f(%1.3f)\n'
         .. '\tREL=%1.3f(%1.3f)\tLG10=%1.3f(%1.3f)'
         
         ):format(
         self.opt.inputType, epoch, n, size, 
         timer:time().real, dataTime,
         errors.MSE, errorSum.MSE / N, 
         errors.RMSE, errorSum.RMSE / N, 
         errors.MAE, errorSum.MAE / N,
         errors.DELTA1, errorSum.DELTA1 / N,
         errors.DELTA2, errorSum.DELTA2 / N,
         errors.DELTA3, errorSum.DELTA3 / N,
         errors.ABS_REL, errorSum.ABS_REL / N, 
         errors.LG10, errorSum.LG10 / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   utils.averageErrors(errorSum, N)
   print((' * Finished epoch # %d     RMSE: %7.3f\n'):format(
      epoch, errorSum.RMSE))

   return errorSum
end

function Trainer:recomputeBatchNorm(dataloader)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local size = math.min(1000, dataloader:size())
   local N = 0

   local batchNorms = {}
   local means = {}
   local variances = {}
   local momentums = {}
   for _, m in ipairs(self.model:listModules()) do
      if torch.isTypeOf(m, 'nn.BatchNormalization') then
         table.insert(batchNorms, m)
         table.insert(means, m.running_mean:clone():zero())
         table.insert(variances, m.running_var:clone():zero())
         table.insert(momentums, m.momentum)
         -- Set momentum to 1
         m.momentum = 1
      end
   end

   print('=> Recomputing batch normalization staticstics')
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- Compute forward pass
      self.model:forward(self.input)

      -- Update running sum of batch mean and variance
      for i, sbn in ipairs(batchNorms) do
         means[i]:add(sbn.running_mean)
         variances[i]:add(sbn.running_var)
      end
      N = N + 1

      print((' | BatchNorm: [%d/%d]    Time %.3f  Data %.3f'):format(
         n, size, timer:time().real, dataTime))

      timer:reset()
      dataTimer:reset()

      if N == size then
         break
      end
   end

   for i, sbn in ipairs(batchNorms) do
      sbn.running_mean:copy(means[i]):div(N)
      sbn.running_var:copy(variances[i]):div(N)
      sbn.momentum = momentums[i]
   end

   -- Copy over running_mean/var from first GPU to other replicas, if using DPT
   if torch.type(self.model) == 'nn.DataParallelTable' then
      self.model.impl:applyChanges()
   end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = math.floor((epoch - 1) / 5)
   return self.opt.LR * math.pow(0.2, decay)
end

return M.Trainer

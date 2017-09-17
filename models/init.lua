--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath):type(opt.tensorType)
      model.__memoryOptimized = nil
   else
      local modelSetupFilename = 'models/resnet-' .. opt.dataset .. '.lua'
      assert(paths.filep(modelSetupFilename), 'Model setup script not found: ' .. modelSetupFilename)
      print('=> Creating model from file: ' .. modelSetupFilename)
      model = require(modelSetupFilename)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local numInputChannels = opt.inputType:len()
      assert(numInputChannels<=4, 'invalid input type: ' .. opt.inputType)
      
      local iheight, iwidth = 0, 0
      if opt.dataset == 'nyudepthv2' then
         iheight, iwidth = 228, 304
      elseif opt.dataset == 'kitti' then
         iheight, iwidth = 228, 912
      else
         error('<models/init.lua> unknown dataset: ' .. opt.dataset)
      end
      
      local sampleInput = torch.zeros(4,numInputChannels,iheight,iwidth):type(opt.tensorType)
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            -- required for DataParallelTable
            require 'modules/ChannelDropout'
            require 'modules/Unpool'

            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   local criterion = nil
   if opt.criterion == 'l2' then
      require 'criteria/myMSECriterion'
      criterion = nn.myMSECriterion():cuda()
   elseif opt.criterion == 'l1' then
      require 'criteria/myAbsCriterion'
      criterion = nn.myAbsCriterion():cuda()
   elseif opt.criterion == 'berhu' then
      require 'criteria/BerHuCriterion'
      criterion = nn.BerHuCriterion():cuda()
   else
      error('<models/init.lua> unknown criterion: ' .. opt.criterion)
   end
   return model, criterion
end

return M

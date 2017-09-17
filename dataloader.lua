--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   self.dataset = dataset
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   print(string.format('nThreads=%d', opt.nThreads))
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   self.permute = true
   local function getCPUType(tensorType)
      if tensorType == 'torch.CudaHalfTensor' then
         return 'HalfTensor'
      elseif tensorType == 'torch.CudaDoubleTensor' then
         return 'DoubleTensor'
      else
         return 'FloatTensor'
      end
   end
   self.cpuType = getCPUType(opt.tensorType)
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm
   if self.permute then
      perm = torch.randperm(size)
   else
      perm = torch.range(1, size)
   end

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops, cpuType)
               local sz = indices:size(1)
               local inputBatch, inputSize
               local targetBatch, targetSize
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  if not inputBatch then
                     inputSize = sample.input:size():totable()
                     if nCrops > 1 then table.remove(inputSize, 1) end
                     inputBatch = torch[cpuType](sz, nCrops, table.unpack(inputSize))
                  end
                  if not targetBatch then
                     targetSize = sample.target:size():totable()
                     if nCrops > 1 then table.remove(targetSize, 1) end
                     targetBatch = torch[cpuType](sz, nCrops, table.unpack(targetSize))
                  end
                  inputBatch[i]:copy(sample.input)
                  targetBatch[i]:copy(sample.target)
                  -- inputBatch[i] = sample.input
                  -- targetBatch[i] = sample.target
               end
               collectgarbage()
               return {
                  input = inputBatch:view(sz * nCrops, table.unpack(inputSize)),
                  target = targetBatch:view(sz * nCrops, table.unpack(targetSize)),
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops,
            self.cpuType
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
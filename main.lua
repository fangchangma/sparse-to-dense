--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
math.randomseed(opt.manualSeed)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

if opt.dataset == 'nyudepthv2' then
   -- NYU Depth V2 has a small test set. Use batchsize=1 for more accurate evaluation
   print('=> NYU Depth V2: set batchSize=1 at test for accurate evaluation.')
   valLoader.batchSize = 1
   print('=> NYU Depth V2: disabled random permute for testing')
   valLoader.permute = false
elseif opt.dataset == 'kitti' and valLoader.__size > 3200 then
   -- KITTI has a large test set. Use a small subset for speed
   print('=> KITTI: set testSize=3200 for speed.')
   valLoader.__size = 3200
end

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Test only
if opt.testOnly then
   require 'modules/ChannelDropout'
   require 'modules/Unpool'

   -- Load trained model
   model = torch.load(opt.bestmodelPath):type(opt.tensorType)

   -- Set batchsize to be 1
   print('=> Test-only: set batchSize=1.')
   valLoader.batchSize = 1

   -- Use the test function in trainer
   local trainer = Trainer(model, nil, opt, optimState)

   local timer = torch.Timer()
   local testLoss = trainer:test(1, valLoader)
   local testTime = timer:time().real
   -- print(string.format(' * Finished RMSE: %3.3f', testLoss.RMSE))
   -- print(string.format(' * Runtime per image: %3.3fms', 1000*testTime/valLoader.__size))
   print(('MSE=%1.3f\nRMSE=%1.3f\nMAE=%1.3f\nDELTA1=%1.3f\nDELTA2=%1.3f\nDELTA3=%1.3f\nREL=%1.3f\nLG10=%1.3f\nruntime(per image)=%1.3fms')
   :format( 
   testLoss.MSE,  testLoss.RMSE, testLoss.MAE,   
   testLoss.DELTA1, testLoss.DELTA2, testLoss.DELTA3,
   testLoss.ABS_REL, testLoss.LG10, 1000*testTime/valLoader.__size
   ))
   return
end

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Logger
local trainLogFile = paths.concat(opt.saveDir, 'trainlog.txt')
local testLogFile = paths.concat(opt.saveDir, 'testlog.txt')
if opt.resume == 'none' then
   os.execute('mkdir -p ' .. opt.saveDir)

   trainFD = io.open(trainLogFile, 'w')
   trainFD:write('epoch, bestModel, MSE, RMSE, MAE, DELTA1, DELTA2, DELTA3, ABS_REL, LG10, Time', "\n")
   trainFD:close()

   testFD = io.open(testLogFile, 'w')
   testFD:write('epoch, bestModel, MSE, RMSE, MAE, DELTA1, DELTA2, DELTA3, ABS_REL, LG10, Time', "\n")
   testFD:close()
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber

local bestRMSE = math.huge
local bestModelEpoch = 1

local function updateBestModel(testLoss, epoch)
   if testLoss.RMSE < bestRMSE then
      bestRMSE = testLoss.RMSE
      bestModelEpoch = epoch
      print(' * Best model is ' .. bestModelEpoch .. ' with RMSE ' .. bestRMSE)
      return true
   end
   return false
end

-- Training
local totalTime = 0
for epoch = startEpoch, opt.nEpochs do
   local timer = torch.Timer()

   -- Train for a single epoch
   local trainLoss = trainer:train(epoch, trainLoader)

   local trainTime = timer:time().real
   totalTime = totalTime + trainTime
   timer:reset()

   -- Run model on validation set
   local testLoss = trainer:test(epoch, valLoader)
   local testTime = timer:time().real
   totalTime = totalTime + testTime
   
   print(('=> Epoch Time = %g (%g)'):format( trainTime+testTime, totalTime / (epoch - startEpoch + 1) ))

   local bestModel = updateBestModel(testLoss, epoch)
   checkpoints.save(epoch, model, trainer.optimState, bestModel)

   print('=> Writing log data to ' .. trainLogFile)
   if paths.filep(trainLogFile) then
      trainFD = io.open(trainLogFile, 'a')
      trainFD:write(('%3i, %3i,' 
         .. ' %2.4f, %2.4f, %2.4f,'
         .. ' %2.4f, %2.4f, %2.4f'
         .. ' %2.4f, %2.4f, %2.4f\n')
         :format(
         epoch, bestModelEpoch, 
         trainLoss.MSE,  trainLoss.RMSE, trainLoss.MAE,   
         trainLoss.DELTA1, trainLoss.DELTA2, trainLoss.DELTA3,
         trainLoss.ABS_REL, trainLoss.LG10, trainTime / trainLoader.__size
         ))
      trainFD:close()
   else
      error('trainLogFile does not exist.')
   end

   if paths.filep(testLogFile) then
      print('=> Writing log data to ' .. testLogFile)
      testFD = io.open(testLogFile, 'a')
      testFD:write(('%3i, %3i,' 
         .. ' %2.4f, %2.4f, %2.4f,'
         .. ' %2.4f, %2.4f, %2.4f,'
         .. ' %2.4f, %2.4f, %2.4f\n')
         :format(
         epoch, bestModelEpoch, 
         testLoss.MSE,  testLoss.RMSE, testLoss.MAE, 
         testLoss.DELTA1, testLoss.DELTA2, testLoss.DELTA3,
         testLoss.ABS_REL, testLoss.LG10, testTime / valLoader.__size))
      testFD:close()
   else
      error('testLogFile does not exist.')
   end
end

if opt.recomputeBatchNorm then
   error('TODO: implement recomputeBatchNorm() for depth prediction')
   
   trainer:recomputeBatchNorm(trainLoader)

   local epoch = opt.nEpochs + 1
   local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)
   local bestModel = updateBestModel(testLoss)

   checkpoints.save(epoch, model, trainer.optimState, bestModel)
end

print(string.format(' * Finished RMSE: %3.3f', bestRMSE))

--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Sparse-to-Dense Training script')
   cmd:text('See https://github.com/fangchangma/sparse-to-dense for details')
   cmd:text()
   cmd:text('Train Options:')

   ------------ Model options --------------------
   cmd:option('-dataset',        'nyudepthv2',  'Options: nyudepthv2 | kitti')
   cmd:option('-inputType',      'rgb',         'Options: rgb | rgbd | d | g | gd')
   cmd:option('-nSample',         0,            'average number of depth samples')
   cmd:option('-criterion',       'l1',         'Options: l1 | l2 | berhu')
   cmd:option('-pretrain',     'false',         'use pretrained model')
   cmd:option('-rep',          'linear',        'Representation of depth. Options: linear | log | inverse')
   cmd:option('-encoderType',  'conv',          'Options: conv | channeldrop | depthsep')
   cmd:option('-decoderType',  'upproj',        'Options: deconv2 | deconv3 | upconv | upproj')

   ------------ General options --------------------
   cmd:option('-manualSeed', 0,                 'Manually set RNG seed')
   cmd:option('-nGPU',       1,                 'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',           'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',         'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',             'Path to save generated files')
   cmd:option('-precision',  'single',          'Options: single | double | half')

   ------------- Data options ------------------------
   cmd:option('-nThreads',        2,            'number of data loading threads')

   ------------- Training options --------------------
   cmd:option('-nEpochs',         20,           'Number of total epochs to run')
   cmd:option('-epochNumber',     1,            'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       16,           'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false',      'Run on validation set only')
   cmd:option('-tenCrop',         'false',      'Ten-crop testing')
   cmd:option('-resume',          'none',       'Path to directory containing checkpoint')

   ---------- Optimization options ----------------------
   cmd:option('-LR',                 0.01,      'initial learning rate')
   cmd:option('-momentum',           0.9,       'momentum')
   cmd:option('-weightDecay',        1e-4,      'weight decay')
   cmd:option('-recomputeBatchNorm', 'false',   'recompute batch norm statistics')

   ---------- Model options ----------------------------------
   cmd:option('-shortcutType', '',              'Options: A | B | C')
   cmd:option('-optimState',   'none',          'Path to an optimState to reload from')
   cmd:option('-optnet',          'true',      'Use optnet to reduce memory usage')
   cmd:option('-nClasses',         0,           'Number of classes in the dataset')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.recomputeBatchNorm = opt.recomputeBatchNorm ~= 'false'

   opt.data = paths.concat('data', opt.dataset)
   if opt.dataset == '' then
      cmd:error('-dataset required. Options: nyudepthv2 | kitti')
   elseif opt.dataset == 'nyudepthv2' or opt.dataset == 'kitti' then
      local trainDir = paths.concat(opt.data, 'train')
      local testDir = paths.concat(opt.data, 'val')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing directory ./data/' .. opt.dataset)
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ' .. opt.dataset .. ' missing `train` directory: ' .. trainDir)
      elseif not paths.dirp(testDir) then
         cmd:error('error: ' .. opt.dataset .. ' missing `val` directory: ' .. testDir)
      end
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   -- Default shortcutType=B and nEpochs=90
   opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
   opt.nEpochs = opt.nEpochs == 0 and 20 or opt.nEpochs

   -- Number of samples
   if opt.inputType == 'rgb' then
      opt.nSample = 0
   end

   -- Tensor type
   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   -- Save directory
   local saveDir = paths.concat('results', ('%s.input=%s.nsample=%d.rep=%s.encoder=%s.decoder=%s.criterion=%s.lr=%g.bs=%d.pretrained=%s'):format 
         (opt.dataset, opt.inputType, opt.nSample, opt.rep, opt.encoderType, opt.decoderType, opt.criterion, opt.LR, opt.batchSize, opt.pretrain))
   if opt.resume~='none' then
      opt.saveDir = opt.resume
   else
      opt.saveDir = saveDir
   end

   -- Model Depth
   if opt.dataset == 'nyudepthv2' then
      opt.depth = 50
   else
      opt.depth = 18
   end

   opt.pretrain = opt.pretrain ~= 'false'
   if opt.pretrain then
      if opt.resume~='none' then
         cmd:error('error: cannot simultaneously load pretrained model and resume previous training')
      end
      opt.pretrainedPath = paths.concat('pretrained', 'resnet-'..opt.depth..'.t7')
      if not opt.testOnly and not paths.filep(opt.pretrainedPath) then
         cmd:error('error: pretrained model ' .. opt.pretrainedPath .. ' not found')
      end
   end

   if opt.testOnly then
      opt.bestmodelPath = paths.concat(opt.saveDir, 'model_best.t7')
      if not paths.filep(opt.bestmodelPath) then
         cmd:error('error: trained model does not exist: ' .. opt.bestmodelPath)
      end
   end

   print('Result directory: ' .. opt.saveDir)
   return opt
end

return M

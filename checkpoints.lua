--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   checkpoint.saveDir = opt.saveDir

   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, bestModel)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   model = deepCopy(model):float():clearState()

   local modelFile = paths.concat(checkpoint.saveDir, 'model_' .. epoch .. '.t7')
   local optimFile = paths.concat(checkpoint.saveDir, 'optimState_' .. epoch .. '.t7')
   -- local modelFile = paths.concat(checkpoint.saveDir, 'model_latest.t7')
   -- local oldModelFile = paths.concat(checkpoint.saveDir, 'model_latest_backup.t7')
   local optimFile = paths.concat(checkpoint.saveDir, 'optimState_latest.t7')
   local latestFile = paths.concat(checkpoint.saveDir, 'latest.t7')

   if epoch>=3 then
      print('=> Saving latest model to ' .. modelFile .. '. Do not interrupt..')
      torch.save(modelFile, model)
      if epoch>1 then
         local oldModelFile = paths.concat(checkpoint.saveDir, 'model_' .. epoch-1 .. '.t7')
         if paths.filep(oldModelFile) then
            os.execute('rm ' .. oldModelFile)
         end
      end
      print('=> Saving latest model completes')

      torch.save(optimFile, optimState)
      torch.save(latestFile, {
         epoch = epoch,
         modelFile = modelFile,
         optimFile = optimFile,
      })

      if bestModel then
         local bestModelFile = paths.concat(checkpoint.saveDir, 'model_best.t7')
         print('=> Saving best model to ' .. bestModelFile .. '. Do not interrupt..')
         os.execute('cp ' .. modelFile .. ' ' .. bestModelFile)
         print('=> Saving best model completes')
      end
   else
      print('=> Skip saving any result until epoch>=3 for the sake of efficiency.')
   end
end

return checkpoint

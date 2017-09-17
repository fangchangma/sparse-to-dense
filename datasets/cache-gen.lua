--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
-- 
--
--  Script to compute list of filenames 
--
--  This generates a file gen/{dataset}.t7 which contains the list of all
--  training and validation hdf5 files that include both rgb and depth images. 
--

local sys = require 'sys'
local ffi = require 'ffi'
require 'hdf5'

local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   -- assert(#classList == 582, 'expected 582 NYU Depth v2 folders')
   return classList, classToIdx
end

local function findFiles(dir, classToIdx)
   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command 
   local findOptions = ' -iname "*.h5" -size +10k'

   -- Find all the hdf5 files using the find command
   local f = nil
   if doSort then
      f = io.popen('find -L ' .. dir .. findOptions .. ' | sort -n ')
   else
      f = io.popen('find -L ' .. dir .. findOptions)
   end

   local maxLength = -1
   local filePaths = {}

   -- Generate a list of all the hdf5 files
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = className .. '/' .. filename
      print(path)
      table.insert(filePaths, path)  
      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nMats = #filePaths
   local filePath = torch.CharTensor(nMats, maxLength):zero()
   for i, path in ipairs(filePaths) do
      ffi.copy(filePath[i]:data(), path)
   end
   return filePath
end

function M.exec(opt, cacheFile)
   local trainDir = paths.concat(opt.data, 'train')
   local valDir = paths.concat(opt.data, 'val')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)
   print(classToIdx)

   print(" | finding all validation images, please wait ..")
   local valFilePath = findFiles(valDir, classToIdx, true)

   print(" | finding all training images, please wait ..")
   local trainFilePath = findFiles(trainDir, classToIdx, false)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         filePath = trainFilePath,
      },
      val = {
         filePath = valFilePath,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M

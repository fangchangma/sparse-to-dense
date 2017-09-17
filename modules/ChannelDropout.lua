--
--  Copyright (c) 2016-2017, Fangchang Ma.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local ChannelDropout, Parent = torch.class('nn.ChannelDropout', 'nn.Module')

function ChannelDropout:__init(n, p, stochasticInference)
   Parent.__init(self)
   if not n or n < 1 then
      error('<ChannelDropout> illegal number of channel, must be 1 <= n')
   end
   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<ChannelDropout> illegal percentage, must be 0 <= p < 1')
   end
   self.train = true
   self.stochastic_inference = stochasticInference or false
   
   self.nChannels = n
   self.noise = torch.Tensor()
end

function ChannelDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)

   if self.train or self.stochastic_inference then
      self.noise = torch.Tensor(self.nChannels):bernoulli(1-self.p)

      -- -- generate the noise only once
      -- if self.noise:dim()==0 then
      --    self.noise = torch.Tensor(self.nChannels):bernoulli(1-self.p)
      -- end

      -- avoid dropping all input channels
      while torch.sum(self.noise) == 0 do
         self.noise = torch.Tensor(self.nChannels):bernoulli(1-self.p)
      end

      for i = 1, self.nChannels do
         if self.noise[i] == 0 then
            self.output[{{}, i, {}, {}}]:zero()
         end
      end
   end

   return self.output
end

function ChannelDropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   
   if self.train or self.stochastic_inference then
      for i = 1, self.nChannels do
         if self.noise[i] == 0 then
            self.gradInput[{{}, i, {}, {}}]:zero()
         end
      end
   end
   
   return self.gradInput
end
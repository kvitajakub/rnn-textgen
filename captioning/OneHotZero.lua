local OneHotZero, parent = torch.class('nn.OneHotZero', 'nn.Module')

-- adapted from https://github.com/karpathy/char-rnn
-- and https://github.com/hughperkins/char-rnn-er

--do not break with input "0" and do not add another tensor dimension,
-- just make last one bigger => dont break LSTM with minibatches

function OneHotZero:__init(outputSize)
   parent.__init(self)
   self.outputSize = outputSize
end


function OneHotZero:updateOutput(input)

    function distributeOneHotZero(input, output)
        for i=1,input:size()[1] do
            if torch.type(input[i]) == 'number' then
                if input[i] ~= 0 then
                    output[input[i]]=1
                end
            else
                distributeOneHotZero(input[i],output[i])
            end
        end
    end

   local size = input:size():totable()
   -- table.insert(size, self.outputSize)
   size[#size] = self.outputSize
   self.output:resize(unpack(size)):zero()

   distributeOneHotZero(input,self.output)

   return self.output
end

function OneHotZero:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size()):zero()
   return self.gradInput
end

function OneHotZero:type(type, typecache)
   self._input = nil
   return parent.type(self, type, typecache)
end

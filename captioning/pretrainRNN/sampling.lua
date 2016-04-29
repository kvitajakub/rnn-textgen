--usual
require 'torch'
--uncommon
require 'rnn'
require 'cunn'
--local
require '../cocodata'
require '../OneHotZero'
require 'sample'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample a language model for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','rnn.torch','Filename of the model and training data.')
cmd:option('-N',4,'How many captions will be generated.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


if path.exists(opt.modelName) then
    model = torch.load(opt.modelName)
else
    error("No file "..opt.modelName)
end

for i=1,opt.N do
    sample(model)
end

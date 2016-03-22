--usual
require 'torch'
--uncommon
require 'rnn'
require 'cunn'
--local
require '../OneHotZero'
require 'sampleRNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Sample a RNN part of the network for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','rnn.torch','Filename of the trained model.')
cmd:text()
cmd:option('-N',4,'How many captions will be generated.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


if path.exists(opt.modelName) then
    model = torch.load(opt.modelName)
else

    print("No file "..opt.modelName)
    os.exit()
end

for i=1,opt.N do
    sample(model)
end

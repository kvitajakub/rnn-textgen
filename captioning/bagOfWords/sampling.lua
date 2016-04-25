--usual
require 'torch'
--uncommon
require 'tds'
require 'rnn'
require 'dpnn'
require 'cutorch'
require 'cunn'
--local
require '../cocodata'
require '../RNN'
require 'sample'
require '../OneHotZero'
require '../connections.lua'
require 'makeBag'



cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Generate captions with model initialized by bag of words.')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','model_bag.torch','Name of the model to be loaded.')
cmd:text()
cmd:option('-N',3,'How many captions will be generated.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


if path.exists(opt.modelName) then
    --load saved model
    model = torch.load(opt.modelName)
    cutorch.setDevice(1)
    model.adapt = torch.load(opt.modelName..'.adapt')
    cutorch.setDevice(2)
    model.rnn = torch.load(opt.modelName..'.rnn')
    js = loadCaptions(model.opt.captionFile)
else
    error("No file "..opt.modelName)
end


local imageFiles, captions = imageSample(js, opt.N, model.opt.imageDirectory) --random
cutorch.setDevice(1)
local images = {}
for i = 1,#captions do
    table.insert(images,captionToBag(model.bag,captions[i]))
end
local size = images[1]:size():totable()
table.insert(size, 1, #images)
images = torch.cat(images):view(unpack(size))

local generatedCaptions = sampleModel(model, images)
printOutput(imageFiles, generatedCaptions, captions)

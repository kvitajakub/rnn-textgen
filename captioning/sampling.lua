--usual
require 'torch'
--uncommon
require 'rnn'
require 'dpnn'
require 'cutorch'
require 'cunn'
tds = require 'tds'
--local
require 'cocodata'
require 'sample'
require 'OneHotZero'
require 'connections'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Generate captions for images with trained model.')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','model.torch','Name of the model to be loaded.')
cmd:text()
cmd:option('-N',3,'How many captions will be generated.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


if path.exists(opt.modelName) then
    --load saved model
    model = torch.load(opt.modelName)
    cutorch.setDevice(1)
    model.cnn = torch.load(opt.modelName..'.cnn')
    model.adapt = torch.load(opt.modelName..'.adapt')
    cutorch.setDevice(2)
    model.rnn = torch.load(opt.modelName..'.rnn')

    js = loadCaptions(model.opt.captionFile)
else
    error("No file "..opt.modelName)
end


local imageFiles, captions = imageSample(js, opt.N, model.opt.imageDirectory) --random
cutorch.setDevice(1)
local images = loadAndPrepare(imageFiles, 224)
local generatedCaptions = sampleModel(model, images)
printOutput(imageFiles, generatedCaptions, captions)

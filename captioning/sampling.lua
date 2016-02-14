--usual
require 'torch'
--uncommon
require 'rnn'
require 'dpnn'
--local
require 'cocodata'
require 'sample'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Generate captions for images with trained model')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','model.dat','name of the model to be saved or loaded.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


if path.exists(opt.modelName) then
    --load saved model
    model = torch.load(opt.modelName)

    js = loadCaptions(model.opt.captionFile)

    print(' >>> Model '..opt.modelName..' loaded.')
    print(model.opt)
else

    print("No file "..opt.modelName)
    os.exit()
end


local imageFiles, captions = imageSample(js, 5, model.opt.imageDirectory)
local images = loadAndPrepare(imageFiles)

local generatedCaptions = sample(model, images)

printOutput(imageFiles, generatedCaptions)

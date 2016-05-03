--usual
require 'torch'
require 'optim'
--uncommon
require 'rnn'
require 'dpnn'
require 'cutorch'
require 'cunn'
tds = require 'tds'
--local
require 'cocodata'
require 'CNN'
require 'RNN'
require 'sample'
require 'OneHotZero'
require 'connections'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluation of the CNN-RNN network for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_val2014.json",'JSON file with the input data (captions, image names).')
cmd:option('-imageDirectory',"/storage/brno7-cerit/home/xkvita01/COCO/val2014/",'Directory with the images with names according to the caption file.')
cmd:text()
cmd:option('-modelName','/storage/brno7-cerit/home/xkvita01/combined_model/0.7244__3x300_view_ft.torch','File name of the saved or loaded model and training data.')
cmd:text()
cmd:option('-printError',10,'Print error once per N minibatches.')
cmd:option('-saveOutput','error.torch','Save the computed error')
cmd:text()


-- parse input params
opt = cmd:parse(arg)

-- minibatch computation
function nextBatch()
    local inputs, outputs = {}, {}

    --get samples from caption file
    local startIndex = model.evaluation_counter*model.opt.batchSize
    local imageFiles, capt = imageSample(js, model.opt.batchSize, opt.imageDirectory, startIndex)
    local maxlen = 0

    --prepare images
    cutorch.setDevice(1)
    local images = loadAndPrepare(imageFiles, 224)

    --maxlen of capt
    for i = 1,#capt do
        if #(capt[i])>maxlen then
            maxlen = #(capt[i])
        end
    end

    cutorch.setDevice(2)
    table.insert(inputs,torch.CudaTensor(#capt,1))
    for i = 1,#capt do
        inputs[1][i][1] = model.charToNumber["START"]
    end

    --for each time slice
    for j = 1,maxlen+1 do
        table.insert(inputs,torch.CudaTensor(#capt,1))
        table.insert(outputs,torch.CudaTensor(#capt))
        --for each sequence
        for i = 1,#capt do
            if j <= #(capt[i]) then
                inputs[#inputs][i][1] = model.charToNumber[capt[i]:sub(j,j)]
                outputs[#outputs][i] = model.charToNumber[capt[i]:sub(j,j)]
            elseif j == #(capt[i])+1 then
                inputs[#inputs][i][1] = 0
                outputs[#outputs][i] = model.charToNumber["END"]
            else
                inputs[#inputs][i][1] = 0
                outputs[#outputs][i] = 0
            end
        end
    end
    --remove last part of inputs because of START added to the beginning
    table.remove(inputs)

	return images, inputs, outputs
end


function training()

    local images, inputs, targets = nextBatch()

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    cutorch.setDevice(1)
    model.cnn:zeroGradParameters()
    model.adapt:zeroGradParameters()
    cutorch.setDevice(2)
    model.rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    cutorch.setDevice(1)
    model.cnn:forward(images)
    model.adapt:forward(model.cnn.output)
    cutorch.synchronizeAll()
    cutorch.setDevice(2)

    connectForward(model)

    local prediction = model.rnn:forward(inputs)
    local error = {}

    for i=1,#prediction do
        table.insert(error,criterion:forward(prediction[i], targets[i]))
    end

    return error
end

function loadModel(modelName)

    local model = torch.load(opt.modelName)
    cutorch.setDevice(1)
    model.cnn = torch.load(opt.modelName..'.cnn')
    model.adapt = torch.load(opt.modelName..'.adapt')
    cutorch.setDevice(2)
    model.rnn = torch.load(opt.modelName..'.rnn')

    return model
end
--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


if path.exists(opt.modelName) then
    --load saved model
    model = loadModel(opt.modelName)

    js = tds.Hash(loadCaptions(opt.captionFile))

    model.opt.printError = opt.printError
    model.evaluation_counter = 0

    print(' >>> Model '..opt.modelName..' loaded.')
    print(' >>> Parameters overriden.')
    print(model.opt)
else

    error("Model not located.")

end

--create criterion
criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1)
cutorch.setDevice(2)
criterion:cuda()


errorEpoch = torch.Tensor(150):zero()
errorCount = torch.Tensor(150):zero()

epochNum = math.floor((model.evaluation_counter * model.opt.batchSize) / #(js['annotations']))
while model.evaluation_counter * model.opt.batchSize - epochNum * #(js['annotations']) < #(js['annotations']) do
    error = training()
    model.evaluation_counter = model.evaluation_counter + 1

    local size
    if #error>150 then
        size = 150
    else
        size = #error
    end

    for i=1,size do
        errorEpoch[i] = errorEpoch[i] + error[i]
        errorCount[i] = errorCount[i] + 1
    end

    if model.evaluation_counter%model.opt.printError==0 then
        print(string.format('minibatch %d (epoch %2.4f) ', model.evaluation_counter, (model.evaluation_counter*model.opt.batchSize)/#(js['annotations'])))
    end

    if model.evaluation_counter%15 or collectgarbage("count")>400000==0 then
        collectgarbage()
    end

    if model.evaluation_counter%300==0 then
        data = {errorEpoch, errorCount}
        torch.save(opt.saveOutput,data)
        print(" >>> Error saved "..opt.saveOutput)
    end
end

data = {errorEpoch, errorCount}
torch.save(opt.saveOutput,data)

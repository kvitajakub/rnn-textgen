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


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a CNN-RNN network for generating image captions.')
cmd:text()
cmd:text('Options')
-- cmd:option('-captionFile',"/home/jkvita/DATA/Diplomka-data/coco/annotations/captions_train2014_small.json",'JSON file with the input data (captions, image names).')
-- cmd:option('-imageDirectory',"/home/jkvita/DATA/Diplomka-data/coco/train2014/",'Directory with the images with names according to the caption file.')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json",'JSON file with the input data (captions, image names).')
cmd:option('-imageDirectory',"/storage/brno7-cerit/home/xkvita01/COCO/train2014/",'Directory with the images with names according to the caption file.')
cmd:text()
-- cmd:option('-pretrainedCNN',"/storage/brno7-cerit/home/xkvita01/CNN/VGG_ILSVRC_16_layers.torch", 'Path to a ImageNet pretrained CNN in Torch format.')
cmd:option('-pretrainedCNN',"/storage/brno7-cerit/home/xkvita01/CNN/nin.torch", 'Path to a ImageNet pretrained CNN in Torch format.')
cmd:option('-pretrainedRNN',"/storage/brno7-cerit/home/xkvita01/RNN/1.0000__2x200.torch", 'Path to a pretrained RNN.')
cmd:option('-batchSize',10,'Minibatch size.')
cmd:option('-printError',2,'Print error once per N minibatches.')
cmd:option('-sample',20,'Try to sample once per N minibatches.')
cmd:option('-saveModel',1000,'Save model once per N minibatches.')
cmd:option('-modelName','model.torch','File name of the saved or loaded model and training data.')
cmd:option('-modelDirectory','/storage/brno7-cerit/home/xkvita01/combined_model/','Directory where to save the model.(add / at the end)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

training_params = {
    evaluation_counter = 0,
    cnn = {
        learningRate=0.001,
        beta1 = 0.92,
        beta2 = 0.999
    },
    rnn = {
        learningRate=0.001,
        beta1 = 0.92,
        beta2 = 0.999
    }
}

-- minibatch computation
function nextBatch()
    local inputs, outputs = {}, {}

    --get samples from caption file
    local startIndex = model.training_params.evaluation_counter*model.opt.batchSize
    local imageFiles, capt = imageSample(js, model.opt.batchSize, model.opt.imageDirectory, startIndex)
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

    function fevalCNN(x_new)
        return x[1], x_grad[1]
    end

    function fevalRNN(x_new)
        return x[2], x_grad[2]
    end

    local images, inputs, targets = nextBatch()
    local rnnLayer = model.rnn:get(1):get(1):get(1):get(1):get(2)

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    cutorch.setDevice(1)
    model.cnn:zeroGradParameters()
    cutorch.setDevice(2)
    model.rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    cutorch.setDevice(1)
    model.cnn:forward(images)
    cutorch.synchronizeAll()
    cutorch.setDevice(2)
    rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, model.cnn.output)

    local prediction = model.rnn:forward(inputs)
    local error = criterion:forward(prediction, targets)

    local gradOutputs = criterion:backward(prediction, targets)
    model.rnn:backward(inputs, gradOutputs)

    cutorch.synchronizeAll()
    cutorch.setDevice(1)
    local gradPrevOutput = rnnLayer.gradPrevOutput:clone()
    model.cnn:backward(images, gradPrevOutput)

    ----------------------------------------------------
    local res1, fs1 = optim.adam(fevalCNN, x[1], model.training_params.cnn)
    cutorch.setDevice(2)
    local res2, fs2 = optim.adam(fevalRNN, x[2], model.training_params.rnn)
    ----------------------------------------------------

    --SequencerCriterion just adds but not divide
    error = error / #inputs
    return error
end


function tryToGenerate(N)
    N = N or 3
    local imageFiles, captions = imageSample(js, N, model.opt.imageDirectory) --random
    cutorch.setDevice(1)
    local images = loadAndPrepare(imageFiles, 224)
    local generatedCaptions = sampleModel(model, images)
    printOutput(imageFiles, generatedCaptions, captions)
end

--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=




if path.exists(opt.modelName) then
    --load saved model
    model = torch.load(opt.modelName)
    model:cuda()

    js = tds.Hash(loadCaptions(model.opt.captionFile))
    -- js = loadCaptions(model.opt.captionFile)

    print(' >>> Model '..opt.modelName..' loaded.')
    print(' >>> Parameters overriden.')
    print(model.opt)
else
    --create new model
    js = tds.Hash(loadCaptions(opt.captionFile))
    -- js = loadCaptions(opt.captionFile)

    local charToNumber, numberToChar = generateCodes(js)

    print("Loading CNN.")
    cutorch.setDevice(1)
    local cnn
    if opt.pretrainedCNN ~= "" and path.exists(opt.pretrainedCNN) then
        cnn = torch.load(opt.pretrainedCNN)
    else
        cnn = CNN.createCNN(500)
        print("CNN created.")
    end

    print("Loading RNN.")
    cutorch.setDevice(2)
    local rnn
    local rnnHiddenUnits
    if opt.pretrainedRNN ~= "" and path.exists(opt.pretrainedRNN) then
        local rnnModel = torch.load(opt.pretrainedRNN)
        rnn = rnnModel.rnn
        rnnHiddenUnits = rnnModel.opt.hiddenUnits
    else
        rnnHiddenUnits = 500
        rnn = RNN.createRNN(#numberToChar, 5, rnnHiddenUnits)
        print("RNN created.")
    end
    collectgarbage()


    cutorch.setDevice(1)
    print("Adding adapter to CNN.")
    cnn:add(nn.Linear(1000, rnnHiddenUnits))

    print("Moving CNN to CUDA.")
    cnn:cuda()

    model = {}
    model.cnn = cnn
    model.rnn = rnn   --remove serial and repack it

    -- print("NOT wrapping in decorator Serial.")
    -- model = nn.Serial(model)
    -- model:mediumSerial()

    model.opt = opt
    model.training_params = training_params
    model.charToNumber = charToNumber
    model.numberToChar = numberToChar

    print("Model created.")
end


--create criterion
criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
cutorch.setDevice(2)
criterion:cuda()

x = {}; x_grad = {}
cutorch.setDevice(1)
x[1], x_grad[1] = model.cnn:getParameters() -- w,w_grad
cutorch.setDevice(2)
x[2], x_grad[2] = model.rnn:getParameters() -- w,w_grad

tryToGenerate()

while true do
    error = training()
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    if model.training_params.evaluation_counter%model.opt.printError==0 then
        print(string.format('minibatch %d (epoch %2.4f) has error %4.7f', model.training_params.evaluation_counter, (model.training_params.evaluation_counter*model.opt.batchSize)/#(js['annotations']), error))
    end


    if model.training_params.evaluation_counter%20==0 then
        collectgarbage()
    end


    if model.training_params.evaluation_counter%model.opt.sample==0 then
        tryToGenerate()
    end


    if model.training_params.evaluation_counter%model.opt.saveModel==0 then
        local name = string.format('%2.4f',(model.training_params.evaluation_counter*model.opt.batchSize)/#(js['annotations']))..'__'..model.opt.modelName
        torch.save(model.opt.modelDirectory..name, model)
        print(" >>> Model and data saved to "..model.opt.modelDirectory..name)
    end


end

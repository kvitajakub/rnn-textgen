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
cmd:option('-pretrainedCNN',"/storage/brno7-cerit/home/xkvita01/CNN/VGG_ILSVRC_16_layers.torch", 'Path to a ImageNet pretrained CNN in Torch format.')
cmd:option('-pretrainedRNN',"/storage/brno7-cerit/home/xkvita01/RNN/minibatch20/0.2415__5layers.torch", 'Path to a pretrained RNN.')
cmd:option('-batchSize',25,'Minibatch size.')
cmd:option('-printError',2,'Print error once per N minibatches.')
cmd:option('-sample',50,'Try to sample once per N minibatches.')
cmd:option('-saveModel',1000,'Save model once per N minibatches.')
cmd:option('-modelName','model.torch','File name of the saved or loaded model and training data.')
cmd:option('-modelDirectory','/storage/brno7-cerit/home/xkvita01/combined_model/','Directory where to save the model.(add / at the end)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


training_params = {
    evaluation_counter = 0,
    captions,  --how many training samples we have

    learningRate=0.001,
    beta1 = 0.92,
    beta2 = 0.999
}


-- minibatch computation
function nextBatch()
    local inputs, targets = {}, {}

    --get samples from caption file
    local imageFiles, captions = imageSampleRandom(js, model.opt.batchSize, model.opt.imageDirectory)

    --prepare images
    local images = loadAndPrepare(imageFiles, 224)
    for i=1,#images do
        images[i] = images[i]:cuda()
    end

    --encode captions
    local sequences = encodeCaption(captions, model.charToNumber)

    for k,v in ipairs(sequences) do
        local sequencerInputTable = {}
        local sequencerTargetTable = {}

        table.insert(sequencerInputTable, v:sub(1,1):cuda())
        for i = 2, v:size()[1]-1 do
            table.insert(sequencerInputTable, v:sub(i,i):cuda())
            table.insert(sequencerTargetTable, v:sub(i,i):cuda())
        end
        table.insert(sequencerTargetTable, v:sub(-1,-1):cuda())

        table.insert(inputs, sequencerInputTable)
        table.insert(targets, sequencerTargetTable)
    end

	return images, inputs, targets
end


function feval(x_new)

    -- copy the weight if are changed, not usually used
    if x ~= x_new then
        x:copy(x_new)
    end

	local images, inputs, targets = nextBatch()
    local error = 0
    local cnn = model:get(1):get(1)
    local rnn = model:get(1):get(2)
    local rnnLayer = rnn:get(1):get(1):get(2)

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    cnn:zeroGradParameters()
    rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    for i = 1, #images do
        -- print(images[i])
        cnn:forward(images[i])
        rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, cnn.output)

        local prediction = rnn:forward(inputs[i])
        error = error + criterion:forward(prediction, targets[i]) / #(targets[i])

        local gradOutputs = criterion:backward(prediction, targets[i])
        rnn:backward(inputs[i], gradOutputs)

        cnn:backward(images[i],rnnLayer.gradPrevOutput)
    end

    error = error / #images

	return error, x_grad
end


function tryToGenerate(N)
    N = N or 3
    local imageFiles, captions = imageSample(js, N, model.opt.imageDirectory)
    local images = loadAndPrepare(imageFiles, 224)

    model:double()
    local generatedCaptions = sample(model, images)
    model:training()
    model:cuda()
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

    training_params.captions = #(js['annotations'])

    print("Loading CNN.")
    local cnn
    if opt.pretrainedCNN ~= "" and path.exists(opt.pretrainedCNN) then
        cnn = torch.load(opt.pretrainedCNN)
    else
        cnn = CNN.createCNN(500)
    end
    collectgarbage()

    print("Loading RNN.")
    local rnn
    if opt.pretrainedRNN ~= "" and path.exists(opt.pretrainedRNN) then
        local rnnModel = torch.load(opt.pretrainedRNN)
        rnn = rnnModel.rnn
        charToNumber = rnnModel.charToNumber
        numberToChar = rnnModel.numberToChar
    else
        rnn = RNN.createRNN(#numberToChar, 5, 500)
    end
    collectgarbage()

    print("Connecting networks.")

    model = nn.Container()
    model:add(cnn)
    model:add(rnn:get(1))   --remove serial and repack it

    model = nn.Serial(model)
    model:mediumSerial()

    print("Moving model to CUDA.")
    model:cuda()

    model.opt = opt
    model.training_params = training_params
    model.charToNumber = charToNumber
    model.numberToChar = numberToChar
end


--create criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion:cuda()


model:training()
x, x_grad = model:getParameters() -- w,w_grad

tryToGenerate()

while true do
-- get weights and loss wrt weights from the model
    res, fs = optim.adam(feval, x, model.training_params)
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    if model.training_params.evaluation_counter%model.opt.printError==0 then
        -- print(string.format('Error for minibatch %4.1f is %4.7f.', model.training_params.evaluation_counter, fs[1]))
        print(string.format('minibatch %d (epoch %2.4f) has error %4.7f', model.training_params.evaluation_counter, (model.training_params.evaluation_counter*model.opt.batchSize)/model.training_params.captions, fs[1]))
    end


    if model.training_params.evaluation_counter%20==0 then
        collectgarbage()
    end


    if model.training_params.evaluation_counter%model.opt.sample==0 then
        tryToGenerate()
    end


    if model.training_params.evaluation_counter%model.opt.saveModel==0 then
        local name = string.format('%2.4f',(model.training_params.evaluation_counter*model.opt.batchSize)/model.training_params.captions)..'__'..model.opt.modelName
        model:double()
        torch.save(model.opt.modelDirectory..name, model)
        model:cuda()
        print(" >>> Model and data saved to "..model.opt.modelDirectory..name)
    end


end

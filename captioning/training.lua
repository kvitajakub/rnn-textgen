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
cmd:option('-pretrainedRNN',"/storage/brno7-cerit/home/xkvita01/RNN/3x200/0.0036__3x200newdrop.torch", 'Path to a pretrained RNN.')
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
    captions,  --how many training samples we have

    learningRate=0.001,
    beta1 = 0.92,
    beta2 = 0.999
}


-- minibatch computation
function nextBatch()
    local inputs, outputs = {}, {}

    --get samples from caption file
    local imageFiles, capt = imageSampleRandom(js, model.opt.batchSize, model.opt.imageDirectory)
    local maxlen = 0

    --prepare images (table of tensors)
    local images = loadAndPrepare(imageFiles, 224)
    --tensor of tensors
    local size = images[1]:size():totable()
    table.insert(size, 1, #images)
    images = torch.cat(images):view(unpack(size))

    --maxlen of capt
    for i = 1,#capt do
        if #(capt[i])>maxlen then
            maxlen = #(capt[i])
        end
    end

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


function feval(x_new)

    -- copy the weight if are changed, not usually used
    if x ~= x_new then
        x:copy(x_new)
    end

	local images, inputs, targets = nextBatch()
    local error = 0
    local cnn = model:get(1):get(1)
    local rnn = model:get(1):get(2)
    local rnnLayer = rnn:get(1):get(1):get(1):get(2)

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    cnn:zeroGradParameters()
    rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    cnn:forward(images)
    rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, cnn.output)

    local prediction = model.rnn:forward(inputs)
    local error = criterion:forward(prediction, targets)

    local gradOutputs = criterion:backward(prediction, targets)
    model.rnn:backward(inputs, gradOutputs)

    cnn:backward(images,rnnLayer.gradPrevOutput)

    --SequencerCriterion just adds but not divide
    error = error / #inputs

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
        print("CNN created.")
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
        print("RNN created.")
    end
    collectgarbage()

    print("Connecting networks.1")
    model = nn.Container()
    model:add(cnn)
    print("Connecting networks.1.5")
    model:add(rnn:get(1))   --remove serial and repack it

    print("Connecting networks.2")
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
criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
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

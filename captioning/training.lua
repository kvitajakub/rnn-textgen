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
cmd:text('Training of the CNN-RNN network for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json",'JSON file with the input data (captions, image names).')
cmd:option('-imageDirectory',"/storage/brno7-cerit/home/xkvita01/COCO/train2014/",'Directory with the images with names according to the caption file.')
cmd:text()
cmd:option('-pretrainedCNN',"/storage/brno7-cerit/home/xkvita01/CNN/VGG_ILSVRC_16_layers_fc7.torch", 'Path to a ImageNet pretrained CNN in Torch format.')
cmd:option('-ft',false,'Finetune CNN on the dataset. (Enable CNN training.)')
cmd:text()
cmd:option('-pretrainedRNN',"/storage/brno7-cerit/home/xkvita01/RNN/2.0000__3x300.torch", 'Path to a pretrained RNN.')
cmd:text()
cmd:option('-rnnLayers',3,'If no RNN is provided, number of recurrent layers while creating RNN. (At least one.)')
cmd:option('-rnnHidden',300,'If no RNN is provided, number of units in hidden layers while creating RNN. (At least one.)')
cmd:option('-rnnDropout',false,'If no RNN is provided, use dropout while creating RNN.')
cmd:text()
cmd:option('-initLayers',0,'How many reccurent layers initialize with CNN data. (0 - all of them)')
cmd:option('-batchSize',15,'Minibatch size.')
cmd:option('-printError',10,'Print error once per N minibatches.')
cmd:option('-sample',100,'Try to sample once per N minibatches.')
cmd:option('-saveModel',10000,'Save model once per N minibatches.')
cmd:option('-modelName','model.torch','File name of the saved or loaded model and training data.')
cmd:option('-modelDirectory','/storage/brno7-cerit/home/xkvita01/combined_model/','Directory where to save the model.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

--used only with opt.ft
training_params_cnn = {
    learningRate=0.001,
    beta1 = 0.92,
    beta2 = 0.999
}
training_params_adapt = {
    learningRate=0.001,
    beta1 = 0.92,
    beta2 = 0.999
}
training_params_rnn = {
    learningRate=0.001,
    beta1 = 0.92,
    beta2 = 0.999
}

-- minibatch computation
function nextBatch()
    local inputs, outputs = {}, {}

    --get samples from caption file
    local startIndex = model.evaluation_counter*model.opt.batchSize
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

    function fevalAdapt(x_new)
        return x[2], x_grad[2]
    end

    function fevalRNN(x_new)
        return x[3], x_grad[3]
    end

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
    local error = criterion:forward(prediction, targets)

    local gradOutputs = criterion:backward(prediction, targets)
    model.rnn:backward(inputs, gradOutputs)

    cutorch.synchronizeAll()
    cutorch.setDevice(1)

    local userGradPrevCell = connectBackward(model)

    model.adapt:backward(model.cnn.output, userGradPrevCell)
    model.cnn:backward(images, model.adapt.gradInput)

    ----------------------------------------------------
    if model.opt.ft then
        local res1, fs1 = optim.adam(fevalCNN, x[1], model.cnn.training_params)
    end
    local res2, fs2 = optim.adam(fevalAdapt, x[2], model.adapt.training_params)
    cutorch.setDevice(2)
    local res3, fs3 = optim.adam(fevalRNN, x[3], model.rnn.training_params)
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


function saveModel(modelName, model)

    torch.save(modelName..".cnn", model.cnn)
    torch.save(modelName..".adapt", model.adapt)
    torch.save(modelName..".rnn", model.rnn)
    local cnn = model.cnn
    model.cnn = nil
    local adapt = model.adapt
    model.adapt = nil
    local rnn = model.rnn
    model.rnn = nil
    torch.save(modelName, model)
    model.cnn = cnn
    model.adapt = adapt
    model.rnn = rnn
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

    js = tds.Hash(loadCaptions(model.opt.captionFile))

    print(' >>> Model '..opt.modelName..' loaded.')
    print(' >>> Parameters overriden.')
    print(model.opt)
else
    --create new model
    js = tds.Hash(loadCaptions(opt.captionFile))

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
    cnn.training_params = training_params_cnn

    print("Loading RNN.")
    cutorch.setDevice(2)
    local rnn
    if opt.pretrainedRNN ~= "" and path.exists(opt.pretrainedRNN) then
        local rnnModel = torch.load(opt.pretrainedRNN)
        rnn = rnnModel.rnn
        opt.rnnHidden = rnnModel.opt.hiddenUnits
    else
        rnn = RNN.createRNN(#numberToChar, opt.rnnLayers, opt.rnnHidden, opt.rnnDropout)
        rnn:cuda()
        print("RNN created.")
    end
    rnn.training_params = training_params_rnn

    --opt.initLayers check
    local rnnLayers = 0
    for i=1,rnn:get(1):get(1):get(1):get(1):size() do
        if torch.isTypeOf(rnn:get(1):get(1):get(1):get(1):get(i),nn.AbstractRecurrent) then
            rnnLayers = rnnLayers + 1
        end
    end
    if opt.initLayers == 0 then
        opt.initLayers = rnnLayers
    elseif rnnLayers < opt.initLayers then
        error("Option initLayers wants to initialize too many layers. Total number of recurrent layers: "..rnnLayers)
    end
    collectgarbage()

    cutorch.setDevice(1)
    print("Creating adapter.")
    local cnnOutput = cnn:forward(torch.randn(3,224,224))
    local adapt
    adapt = nn.Sequential()
    adapt:add(nn.Linear(cnnOutput:size()[cnnOutput:dim()], opt.rnnHidden * opt.initLayers))
    adapt:add(nn.Tanh())
    adapt:add(nn.Linear(opt.rnnHidden * opt.initLayers, opt.rnnHidden * opt.initLayers))
    adapt.training_params = training_params_adapt
    adapt:cuda()

    print("Moving CNN to CUDA.")
    cnn:cuda()

    model = {}
    model.cnn = nn.Serial(cnn)
    model.rnn = nn.Serial(rnn)
    model.adapt = nn.Serial(adapt)
    model.opt = opt
    model.evaluation_counter = 0
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
x[2], x_grad[2] = model.adapt:getParameters() -- w,w_grad
cutorch.setDevice(2)
x[3], x_grad[3] = model.rnn:getParameters() -- w,w_grad

tryToGenerate()


epochNum = math.floor((model.evaluation_counter * model.opt.batchSize) / #(js['annotations']))
while model.evaluation_counter * model.opt.batchSize - epochNum * #(js['annotations']) < #(js['annotations']) do
    error = training()
    model.evaluation_counter = model.evaluation_counter + 1

    if model.evaluation_counter%model.opt.printError==0 then
        print(string.format('minibatch %d (epoch %2.4f) has error %4.7f    memory in use: %d kB', model.evaluation_counter, (model.evaluation_counter*model.opt.batchSize)/#(js['annotations']), error,collectgarbage("count")))
    end

    if model.evaluation_counter%15 or collectgarbage("count")>400000==0 then
        collectgarbage()
    end


    if model.evaluation_counter%model.opt.sample==0 then
        tryToGenerate()
    end

    --save model each N minibatches
    if model.evaluation_counter%model.opt.saveModel==0 then
        local name = string.format('%2.4f',(model.evaluation_counter*model.opt.batchSize)/#(js['annotations']))..'__'..model.opt.modelName
        saveModel(model.opt.modelDirectory..'/'..name, model)
        print(" >>> Model and data saved to "..model.opt.modelDirectory..'/'..name)
    end
end

--save model after epoch
local name = string.format('%2.4f',(model.evaluation_counter*model.opt.batchSize)/#(js['annotations']))..'__'..model.opt.modelName
saveModel(model.opt.modelDirectory..'/'..name, model)
print(" >>> Model and data saved to "..model.opt.modelDirectory..'/'..name)

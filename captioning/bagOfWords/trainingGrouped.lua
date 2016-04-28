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
require '../cocodata'
require '../RNN'
require 'sample'
require '../OneHotZero'
require '../connections.lua'
require 'makeBag'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training of the RNN network for generating image captions initialized with binary bag of words.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json",'JSON file with the input data (captions, image names).')
cmd:option('-imageDirectory',"/storage/brno7-cerit/home/xkvita01/COCO/train2014/",'Directory with the images with names according to the caption file.')
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
cmd:option('-modelName','model_bag.torch','File name of the saved or loaded model and training data.')
cmd:option('-modelDirectory','/storage/brno7-cerit/home/xkvita01/combined_model/','Directory where to save the model.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

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
    local maxlen = 0

    cutorch.setDevice(1)
    local images = {}
    local capt = {}

    for j=1,model.opt.batchSize do

        --going sequentially from starting point
        index = (startIndex+j-1) % #jsGrouped + 1

        table.insert(capt, jsGrouped[index][1])

        for i=1,#(jsGrouped[index]) do
            if i==1 then
                table.insert(images,captionToBag(model.bag,jsGrouped[index][i]))
            else
                images[#images] = images[#images] + captionToBag(model.bag,jsGrouped[index][i])
            end
        end
    end

    local size = images[1]:size():totable()
    table.insert(size, 1, #images)
    images = torch.cat(images):view(unpack(size))


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
    model.adapt:zeroGradParameters()
    cutorch.setDevice(2)
    model.rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    cutorch.setDevice(1)
    model.adapt:forward(images)
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

    model.adapt:backward(images, userGradPrevCell)

    ----------------------------------------------------
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

    cutorch.setDevice(1)
    local images = {}
    local captions = {}

    for j=1,N do

        index = math.ceil(torch.random(1,#jsGrouped))

        for i=1,#(jsGrouped[index]) do
            if i==1 then
                table.insert(images,captionToBag(model.bag,jsGrouped[index][i]))
                table.insert(captions,jsGrouped[index][i])
            else
                images[#images] = images[#images] + captionToBag(model.bag,jsGrouped[index][i])
            end
        end
    end

    local size = images[1]:size():totable()
    table.insert(size, 1, #images)
    images = torch.cat(images):view(unpack(size))

    local generatedCaptions = sampleModel(model, images)
    printOutput(captions, generatedCaptions, captions)
end


function saveModel(modelName, model)

    torch.save(modelName..".adapt", model.adapt)
    torch.save(modelName..".rnn", model.rnn)

    local adapt = model.adapt
    model.adapt = nil
    local rnn = model.rnn
    model.rnn = nil
    torch.save(modelName, model)
    model.adapt = adapt
    model.rnn = rnn
end


function loadModel(modelName)

    local model = torch.load(opt.modelName)
    cutorch.setDevice(1)
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
    jsGrouped = tds.Hash(groupCaptions(js))

    print(' >>> Model '..opt.modelName..' loaded.')
    print(' >>> Parameters overriden.')
    print(model.opt)
else
    --create new model
    js = tds.Hash(loadCaptions(opt.captionFile))

    local charToNumber, numberToChar = generateCodes(js)

    print("Creating bag of words.")
    cutorch.setDevice(1)
    bag = tds.Hash(makeBag(js))

    jsGrouped = tds.Hash(groupCaptions(js))

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
    local adapt
    adapt = nn.Sequential()
    adapt:add(nn.Linear(bag['length'], opt.rnnHidden * opt.initLayers))
    adapt:add(nn.Tanh())
    adapt:add(nn.Linear(opt.rnnHidden * opt.initLayers, opt.rnnHidden * opt.initLayers))
    adapt.training_params = training_params_adapt
    adapt:cuda()

    model = {}
    model.bag = bag
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

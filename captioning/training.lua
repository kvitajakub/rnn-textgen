--usual
require 'torch'
require 'optim'
--uncommon
require 'rnn'
require 'dpnn'
require 'cutorch'
require 'cunn'
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
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/coco/captions_train2014_small.json",'JSON file with the input data (captions, image names).')
cmd:option('-imageDirectory',"/storage/brno7-cerit/home/xkvita01/coco/train2014/",'Directory with the images with names according to the caption file.')
cmd:text()
cmd:option('-recurLayers',4,'Number of recurrent layers. (At least one.)')
cmd:option('-batchSize',10,'Minibatch size.')
cmd:option('-modelName','model.dat','File name of the saved or loaded model and training data.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


training_params = {
    algorithm = optim.adam,
    evaluation_counter = 0,

    learningRate=0.002
}


-- minibatch computation
function nextBatch()
    local inputs, targets = {}, {}

    --get samples from caption file
    local imageFiles, captions = imageSample(js, model.opt.batchSize, model.opt.imageDirectory)

    --prepare images
    local images = loadAndPrepare(imageFiles)
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
    local rnnLayer = model:get(1):get(2):get(1):get(1):get(2)

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    cnn:zeroGradParameters()
    rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    for i = 1, #images do
        -- print(images[i])
        cnn:forward(images[i])
        rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, cnn.output)

        local prediction = rnn:forward(inputs[i])
        error = error + criterion:forward(prediction, targets[i])/#(targets[i])
        local gradOutputs = criterion:backward(prediction, targets[i])
        rnn:backward(inputs[i], gradOutputs)

        cnn:backward(images[i],rnnLayer.gradPrevOutput)
    end


	return error, x_grad
end

--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if path.exists(opt.modelName) then
    --load saved model
    model = torch.load(opt.modelName)
    model:cuda()

    js = loadCaptions(model.opt.captionFile)

    print(' >>> Model '..opt.modelName..' loaded.')
    print(' >>> Parameters overriden.')
    print(model.opt)
else
    --create new model
    js = loadCaptions(opt.captionFile)
    local charToNumber, numberToChar = generateCodes(js)

    local cnn = CNN.createCNN()
    local rnn = RNN.createRNN(#numberToChar, opt.recurLayers)

    model = nn.Container()
    model:add(cnn)
    model:add(rnn)

    model = nn.Serial(model)
    model:mediumSerial()

    model:cuda()

    model.opt = opt
    model.training_params = training_params
    model.charToNumber = charToNumber
    model.numberToChar = numberToChar
end


--create criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion:cuda()


x, x_grad = model:getParameters() -- w,w_grad


while true do
-- get weights and loss wrt weights from the model
    res, fs = model.training_params.algorithm(feval, x, model.training_params)
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    if model.training_params.evaluation_counter%5==0 then
        print(string.format('Error for minibatch %4.1f is %4.7f.', model.training_params.evaluation_counter, fs[1]/model.opt.batchSize))
    end
    if model.training_params.evaluation_counter%20==0 then
        collectgarbage()
    end
    -- if model.training_params.evaluation_counter%100==0 then
    --     sample()
    -- end
    if model.training_params.evaluation_counter%250==0 then
        model:double()
        torch.save(model.opt.modelName, model)
        model:cuda()
        print(" >>> Model and data saved to "..model.opt.modelName..".")
    end
end

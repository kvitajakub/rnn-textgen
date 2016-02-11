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
cmd:option('-batchSize',1,'Minibatch size.')
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
    local imageFiles, captions = imageSample(js, opt.batchSize, opt.imageDirectory)

    --prepare images
    local images = loadAndPrepare(imageFiles)
    for i=1,#images do
        images[i] = images[i]:cuda()
    end

    --encode captions
    local sequences = encodeCaption(captions,charToNumber)

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



-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. weigths is the vector of trainable weights,
-- it extracts a mini_batch via the nextBatch method
function feval(x_new)

    -- copy the weight if are changed, not usually used
    if x ~= x_new then
        x:copy(x_new)
    end

	local images, inputs, targets = nextBatch()

    local error = 0

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    cnn:zeroGradParameters()
    rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    for i = 1, #images do
        -- print(images[i])
        cnn:forward(images[i])
        rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, cnn.output)

        local prediction = rnn:forward(inputs[i])
        error = error + criterion:forward(prediction, targets[i])
        local gradOutputs = criterion:backward(prediction, targets[i])
        rnn:backward(inputs[i], gradOutputs)

        cnn:backward(images[i],rnnLayer.gradPrevOutput)
    end


	return error, x_grad
end









js = loadCaptions(opt.captionFile)
charToNumber, numberToChar = generateCodes(js)

rnn, rnnLayer = RNN.createRNN(#numberToChar, 5)
cnn = CNN.createCNN()

model = nn.Container()
model:add(cnn)
model:add(rnn)

model = nn.Serial(model)
model:mediumSerial()

model:cuda()


--create criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion:cuda()


x, x_grad = model:getParameters() -- w,w_grad




while true do
-- get weights and loss wrt weights from the model
    res, fs = training_params.algorithm(feval, x, training_params)
    training_params.evaluation_counter = training_params.evaluation_counter + 1

    if training_params.evaluation_counter%2==0 then
        print(string.format('error for minibatch %4.1f is %4.7f', training_params.evaluation_counter, fs[1]))
    end
    if training_params.evaluation_counter%5==0 then
        collectgarbage()
    end
    -- if training_params.evaluation_counter%100==0 then
    --     sample()
    -- end
    -- if training_params.evaluation_counter%1250==0 then
    --     torch.save(model.opt.modelName, model)
    --     print("Model saved to "..model.opt.modelName)
    -- end
end

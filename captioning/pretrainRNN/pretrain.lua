--usual
require 'torch'
require 'optim'
--uncommon
require 'rnn'
require 'cunn'
--local
require '../RNN'
require '../cocodata'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a RNN  part of the network for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/coco/captions_train2014_small.json",'JSON file with the input data (captions, image names).')
-- cmd:option('-captionFile',"/home/jkvita/DATA/Diplomka-data/coco/annotations/captions_train2014_small.json",'JSON file with the input data (captions, image names).')
cmd:text()
cmd:option('-recurLayers',3,'Number of recurrent layers. (At least one.)')
cmd:option('-hiddenUnits',400,'Number of units in hidden layers. (At least one.)')
cmd:option('-batchSize',5,'Minibatch size.')
cmd:option('-printError',5,'Print error once per N minibatches.')
cmd:option('-sample',100,'Try to sample once per N minibatches.')
cmd:option('-saveModel',1000,'Save model once per N minibatches.')
cmd:option('-modelName','rnn.torch','Where to save the model and training data.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

training_params = {
    algorithm = optim.adam,
    evaluation_counter = 0,

    learningRate=0.002
}

function listCaptions(js)
    local captions = {}
    for i=1,#js['annotations'] do
        table.insert(captions, js['annotations'][i]['caption'])
    end
    return captions
end

js = loadCaptions(opt.captionFile)
captions = listCaptions(js)

if opt.modelName ~= "" and path.exists(opt.modelName) then
    model = torch.load(opt.modelName)
    print('Model '..opt.modelName..' loaded.')
    print('Parameters overriden.')
    print(model.opt)

else

    local charToNumber, numberToChar = generateCodes(js)

    rnn = RNN.createRNN(#numberToChar, opt.recurLayers, opt.hiddenUnits)
    rnn:cuda()

    model = {}
    model.rnn = rnn
    model.opt = opt
    model.training_params = training_params
    model.charToNumber = charToNumber
    model.numberToChar = numberToChar

end

--create criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion:cuda()

-- minibatch computation
function nextBatch()
    local inputs, targets = {}, {}
    local capt

    for i = 1,model.opt.batchSize do
        local input, target = {}, {}
        capt = captions[math.ceil(torch.random(1,(#captions)))]

        table.insert(input,torch.CudaTensor(1))
        input[1][1] = model.charToNumber["START"]

        for j=1,#capt do
            table.insert(input,torch.CudaTensor(1))
            table.insert(target,torch.CudaTensor(1))

            local val = model.charToNumber[capt:sub(j,j)]
            input[#input][1] = val
            target[#target][1] = val
        end

        table.insert(target,torch.CudaTensor(1))
        target[#target][1] = model.charToNumber["END"]

        table.insert(inputs, input)
        table.insert(targets,target)
    end

    return inputs, targets
end


function feval(x_new)

    if x ~= x_new then
        x:copy(x_new)
    end

	local inputs, targets = nextBatch()

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    model.rnn:zeroGradParameters()

    local error = 0
    -- evaluate the loss function and its derivative wrt x, given mini batch
    for i=1,#inputs do

        local prediction = model.rnn:forward(inputs[i])
        error = error + criterion:forward(prediction, targets[i]) / #(inputs[i])
        local gradOutputs = criterion:backward(prediction, targets[i])
        model.rnn:backward(inputs[i], gradOutputs)
    end
    error = error / #inputs

	return error, x_grad
end


function sample()

    local samplingRnn = model.rnn:get(1):get(1):get(1)
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count
    print('======Sampling==============================================')

    local prediction, sample, char
    local description = ""
    local safeCounter = 0


    -- -- generation with initialization by random character
    -- local randomCharNumber = math.ceil(torch.random(1, #model.numberToChar))
    -- -- generation with initialization by specific character (space)
    local randomCharNumber = model.charToNumber['START']
    prediction = samplingRnn:forward(torch.CudaTensor{randomCharNumber})
    prediction:exp()
    sample = torch.multinomial(prediction,1)
    char = model.numberToChar[sample[1][1]]

    while char ~= "END" and safeCounter<200 do

        description = description .. char
        prediction = samplingRnn:forward(sample[1])
        prediction:exp()
        sample = torch.multinomial(prediction,1)
        char = model.numberToChar[sample[1][1]]

        safeCounter = safeCounter + 1
    end

    print(description)
    print('============================================================')

    samplingRnn:training()--!!!!!! IMPORTANT switch back to remembering state

    return description
end


x, x_grad = model.rnn:getParameters() -- w,w_grad

sample()

for i=1,10000 do
-- get weights and loss wrt weights from the model
    res, fs = model.training_params.algorithm(feval, x, model.training_params)
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    if model.training_params.evaluation_counter%model.opt.printError==0 then
        print(string.format('error for minibatch %4.1f is %4.7f', model.training_params.evaluation_counter, fs[1]))
    end
    if model.training_params.evaluation_counter%50==0 then
        collectgarbage()
    end
    if model.training_params.evaluation_counter%model.opt.sample==0 then
        sample()
    end
    if model.training_params.evaluation_counter%model.opt.saveModel==0 then
        torch.save(model.training_params.evaluation_counter..'__'..model.opt.modelName, model)
        print("Model saved to "..model.opt.modelName)
    end
end

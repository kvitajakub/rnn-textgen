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
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json",'JSON file with the input data (captions, image names).')
cmd:text()
cmd:option('-recurLayers',6,'Number of recurrent layers. (At least one.)')
cmd:option('-hiddenUnits',500,'Number of units in hidden layers. (At least one.)')
cmd:option('-batchSize',25,'Minibatch size.')
cmd:option('-printError',5,'Print error once per N minibatches.')
cmd:option('-sample',100,'Try to sample once per N minibatches.')
cmd:option('-saveModel',1000,'Save model once per N minibatches.')
cmd:option('-modelName','rnn.torch','Filename of the model and training data.')
cmd:option('-modelDirectory','/storage/brno7-cerit/home/xkvita01/RNN/','Directory where to save the model.(add / at the end)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

training_params = {
    evaluation_counter = 0,

    learningRate=0.0014,
    beta1 = 0.92,
    beta2 = 0.999
}

-- training_params = {
--     algorithm = optim.sgd,
--     evaluation_counter = 0,
--
--     learningRate=0.005,
--     weightDecay=0.02,
--     momentum = 0.90,
--     nesterov = true,
--     dampening = 0
-- }

function listCaptions(js)
    local captions = {}
    for i=1,#js['annotations'] do
        table.insert(captions, js['annotations'][i]['caption'])
    end
    return captions
end


if opt.modelName ~= "" and path.exists(opt.modelName) then
    model = torch.load(opt.modelName)
    print('Model '..opt.modelName..' loaded.')
    print('Parameters overriden.')
    print(model.opt)

    js = loadCaptions(model.opt.captionFile)
    captions = listCaptions(js)

else

    js = loadCaptions(opt.captionFile)
    captions = listCaptions(js)

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

        --compute starting index
        local index = (model.training_params.evaluation_counter*model.opt.batchSize+i-1) % (#captions) +1

        capt = captions[index]

        table.insert(input,torch.CudaTensor(1))
        input[1][1] = model.charToNumber["START"]

        for j=1,#capt do
            -- if j<= #capt then
                table.insert(input,torch.CudaTensor(1))
                table.insert(target,torch.CudaTensor(1))

                local val = model.charToNumber[capt:sub(j,j)]
                input[#input][1] = val
                target[#target][1] = val
            -- end
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
    -- -- generation with initialization by specific character (start character)
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

epochNum = math.floor((model.training_params.evaluation_counter * model.opt.batchSize) / #captions)

--do one epoch of training
while model.training_params.evaluation_counter * model.opt.batchSize - epochNum * #captions < #captions do
-- get weights and loss wrt weights from the model
    res, fs = optim.adam(feval, x, model.training_params)
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    if model.training_params.evaluation_counter%model.opt.printError==0 then
        print(string.format('minibatch %d (epoch %2.4f) has error %4.7f', model.training_params.evaluation_counter, (model.training_params.evaluation_counter*model.opt.batchSize)/#captions, fs[1]))
    end
    if model.training_params.evaluation_counter%50==0 then
        collectgarbage()
    end
    if model.training_params.evaluation_counter%model.opt.sample==0 then
        sample()
    end
    if model.training_params.evaluation_counter%model.opt.saveModel==0 then
        local name = string.format('%2.4f',(model.training_params.evaluation_counter*model.opt.batchSize)/#captions)..'__'..model.opt.modelName
        torch.save(model.opt.modelDirectory..name, model)
        print("Model saved to "..model.opt.modelDirectory..name)
    end
end

--save the trained epoch
local name = string.format('%2.4f',(model.training_params.evaluation_counter*model.opt.batchSize)/#captions)..'__'..model.opt.modelName
torch.save(model.opt.modelDirectory..name, model)
print("Model saved to "..model.opt.modelDirectory..name)

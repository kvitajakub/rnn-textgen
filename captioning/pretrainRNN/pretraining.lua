--usual
require 'torch'
require 'optim'
--uncommon
require 'rnn'
require 'cunn'
--local
require '../RNN'
require '../cocodata'
require '../OneHotZero'
require 'sampleRNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a RNN language model for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json",'JSON file with the input data (captions, image names).')
cmd:text()
cmd:option('-recurLayers',3,'Number of recurrent layers. (At least one.)')
cmd:option('-hiddenUnits',300,'Number of units in hidden layers. (At least one.)')
cmd:option('-dropout',false,'Use dropout.')
cmd:option('-batchSize',15,'Minibatch size.')
cmd:option('-printError',10,'Print error once per N minibatches.')
cmd:option('-sample',100,'Try to sample once per N minibatches.')
cmd:option('-saveModel',10000,'Save model once per N minibatches.')
cmd:option('-modelName','rnn.torch','Filename of the model and training data.')
cmd:option('-modelDirectory','/storage/brno7-cerit/home/xkvita01/RNN/','Directory where to save the model.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

training_params = {
    evaluation_counter = 0,

    learningRate=0.001,
    beta1 = 0.92,
    beta2 = 0.999
}


function listCaptions(js)
    local captions = {}
    for i=1,#js['annotations'] do
        table.insert(captions, js['annotations'][i]['caption'])
    end
    return captions
end

if not path.exists(opt.modelDirectory) then
    os.execute("mkdir -p "..opt.modelDirectory)
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

    rnn = RNN.createRNN(#numberToChar, opt.recurLayers, opt.hiddenUnits, opt.dropout)
    rnn:cuda()

    rnn:mediumSerial()

    model = {}
    model.rnn = rnn
    model.opt = opt
    model.training_params = training_params
    model.charToNumber = charToNumber
    model.numberToChar = numberToChar

end

--create criterion
criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
criterion:cuda()


-- minibatch computation
function nextBatch()
    local inputs, outputs = {}, {}
    local capt = {}
    local maxlen = 0

    --list of current captions
    for i = 1,model.opt.batchSize do
        local index = (model.training_params.evaluation_counter*model.opt.batchSize+i-1) % (#captions) +1
        table.insert(capt,captions[index])
        if #(captions[index])>maxlen then
            maxlen = #(captions[index])
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

    return inputs, outputs
end


function feval(x_new)

    if x ~= x_new then
        x:copy(x_new)
    end

	local inputs, targets = nextBatch()

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    model.rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    local prediction = model.rnn:forward(inputs)
    local error = criterion:forward(prediction, targets)

    local gradOutputs = criterion:backward(prediction, targets)
    model.rnn:backward(inputs, gradOutputs)

    --SequencerCriterion just adds but not divide
    error = error / #inputs

	return error, x_grad
end


x, x_grad = model.rnn:getParameters() -- w,w_grad

sample(model)

epochNum = math.floor((model.training_params.evaluation_counter * model.opt.batchSize) / #captions)

--do one epoch of training
while model.training_params.evaluation_counter * model.opt.batchSize - epochNum * #captions < #captions do
-- get weights and loss wrt weights from the model
    res, fs = optim.adam(feval, x, model.training_params)
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    --print error
    if model.training_params.evaluation_counter%model.opt.printError==0 then
        print(string.format('minibatch %d (epoch %2.4f) has error %4.7f', model.training_params.evaluation_counter, (model.training_params.evaluation_counter*model.opt.batchSize)/#captions, fs[1]))
    end

    --collect garbage
    if model.training_params.evaluation_counter%10==0 then
        collectgarbage()
    end

    --sample
    if model.training_params.evaluation_counter%model.opt.sample==0 then
        sample(model)
    end

    --save
    if model.training_params.evaluation_counter%model.opt.saveModel==0 then
        local name = string.format('%2.4f',(model.training_params.evaluation_counter*model.opt.batchSize)/#captions)..'__'..model.opt.modelName
        torch.save(model.opt.modelDirectory..'/'..name, model)
        print("Model saved to "..model.opt.modelDirectory..name)
    end
end

--save the trained epoch
local name = string.format('%2.4f',(model.training_params.evaluation_counter*model.opt.batchSize)/#captions)..'__'..model.opt.modelName
torch.save(model.opt.modelDirectory..'/'..name, model)
print("Model saved to "..model.opt.modelDirectory..name)

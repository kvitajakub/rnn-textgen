--usual
require 'torch'
require 'optim'
--uncommon
require 'rnn'
require 'dpnn'
require 'cutorch'
require 'cunn'
--local
require 'readFile'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network for next character prediction.')
cmd:text()
cmd:text('Options')
cmd:option('-inputFile',"../text/shakespeare.txt",'File with the input data.')
cmd:option('-hiddenSize',150,'Number of units in the hidden layer.')
cmd:option('-layers',2,'Number of recurrent layers. (At least one.)')
cmd:option('-rho',40,'How far past are we looking.')
cmd:option('-batchSize',20,'Minibatch size.')
cmd:option('-modelName','model.dat','name of the model to be saved or loaded.')
-- cmd:option('-gpu',0,'Use cpu(=0) or gpu(>0)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


sgd_params = {
   learningRate = 0.05,
   learningRateDecay = 1e-4,
   weightDecay = 0.002,
   momentum = 0.90,
   nesterov = true,
   dampening = 0
}

-- sgd_params = {
--    learningRate = 0.02,
--    learningRateDecay = 0,
--    weightDecay = 0.001,
--    momentum = 0,
--    nesterov = false,
--    dampening = 0
-- }

adam_params = {
    learningRate=0.002
}


-- --try to load model
if path.exists(opt.modelName) then
    rnn = torch.load(opt.modelName)
    print('Model '..opt.modelName..' loaded.')
    print('Parameters overriden.')
    print(rnn.opt)

    rnn:cuda()

    --load inputFile
    -- data loading and sequence creation
    text, sequence, charToNumber, numberToChar = readFile:processFile(rnn.opt.inputFile)
    sequence = sequence:cuda()
else
    --model not available, create new
    -- data loading and sequence creation
    text, sequence, charToNumber, numberToChar = readFile:processFile(opt.inputFile)
    sequence = sequence:cuda()
    --network creation
    -- rnn for training with Sequencer and negative log likelihood criterion
    rnn = nn.Sequential()
    rnn:add(nn.OneHot(#numberToChar))
    rnn:add(nn.LSTM(#numberToChar, opt.hiddenSize, opt.rho))
    for i=2,opt.layers do
        rnn:add(nn.LSTM(opt.hiddenSize, opt.hiddenSize, opt.rho))
    end
    rnn:add(nn.Linear(opt.hiddenSize, #numberToChar))
    rnn:add(nn.LogSoftMax())
    rnn = nn.Sequencer(rnn)
    rnn = nn.Serial(rnn)
    rnn:mediumSerial()

    --pridame opt do rnn aby se nam pak ulozilo
    rnn.opt = opt
    rnn:cuda()

    --INICIALIZATION
    -- A1: initialization often depends on each dataset.
    --rnn:getParameters():uniform(-0.1, 0.1)
end

--create criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion:cuda()

-- minibatch computation
-- random subsequences from whole text
function nextBatch()
    local offsets, inputs, targets = {}, {}, {}
    for i = 1,rnn.opt.batchSize do
        table.insert(offsets,math.ceil(torch.random(1,((#sequence)[1]-rnn.opt.rho))))
    end
    offsets = torch.LongTensor(offsets)
    for i = 1,rnn.opt.rho do
        table.insert(inputs,sequence:index(1,offsets))
        offsets:add(1)
        table.insert(targets,sequence:index(1,offsets))
    end
	return inputs, targets
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

	local inputs, targets = nextBatch()

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
    rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    local prediction = rnn:forward(inputs)
    local error = criterion:forward(prediction, targets)
    local gradOutputs = criterion:backward(prediction, targets)
    rnn:backward(inputs, gradOutputs)

	return error, x_grad
end

--sampling with current network
function sample(samples)

    samples = samples or 2*rnn.opt.rho

    local samplingRnn = rnn:get(1):get(1):get(1):clone()
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count
    print('======Sampling==============================================')

    local prediction, sample, sampleCoded
    local randomStart = math.ceil(torch.random(1,((#sequence)[1]-rnn.opt.rho)))
    for i = randomStart,randomStart+rnn.opt.rho do
        io.write(numberToChar[sequence[i]])
        prediction = samplingRnn:forward(torch.CudaTensor{sequence[i]})
    end
    io.write('__|||__')
    io.flush()

    for i=1,samples do
        prediction:exp()
        sample = torch.multinomial(prediction,1)

        io.write(numberToChar[sample[1][1]])

        prediction = samplingRnn:forward(sample[1])
    end
    io.write('\n')
    io.flush()
    print('============================================================')
end

x, x_grad = rnn:getParameters() -- w,w_grad


-- rnn:double()
-- torch.save(rnn.opt.modelName, rnn)
-- os.exit()
sample()
adam_params.evalCounter=0

while true do
-- get weights and loss wrt weights from the model
    res, fs = optim.adam(feval, x, adam_params)
    adam_params.evalCounter = adam_params.evalCounter + 1

    if adam_params.evalCounter%25==0 then
        print(string.format('error for minibatch %4.1f is %4.7f', adam_params.evalCounter, fs[1] / rnn.opt.rho))
    end
    if adam_params.evalCounter%50==0 then
        collectgarbage()
    end
    if adam_params.evalCounter%100==0 then
        sample()
    end
    if adam_params.evalCounter%250==0 then
        torch.save(rnn.opt.modelName, rnn)
        print("Model saved to "..rnn.opt.modelName)
    end
end

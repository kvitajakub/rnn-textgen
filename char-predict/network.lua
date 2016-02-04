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
cmd:text()

-- parse input params
opt = cmd:parse(arg)


-- training_params = {
--     algorithm = optim.sgd,
--     evaluation_counter = 0,
--
--    learningRate = 0.05,
--    learningRateDecay = 1e-4,
--    weightDecay = 0.002,
--    momentum = 0.90,
--    nesterov = true,
--    dampening = 0
-- }

training_params = {
    algorithm = optim.adam,
    evaluation_counter = 0,

    learningRate=0.002
}

--create new lstm model, input is one number in tensor
--not using global variables
function createLSTMNetwork(input_output, hidden, lstm_layers, rho)
    local rnn = nn.Sequential()
    rnn:add(nn.OneHot(input_output))
    rnn:add(nn.LSTM(input_output, hidden, rho))
    for i=2,lstm_layers do
        rnn:add(nn.LSTM(hidden, hidden, rho))
    end
    rnn:add(nn.Linear(hidden, input_output))
    rnn:add(nn.LogSoftMax())
    rnn = nn.Sequencer(rnn)
    rnn = nn.Serial(rnn)
    return rnn
end


-- minibatch computation
-- random subsequences from whole text
function nextBatch()
    local offsets, inputs, targets = {}, {}, {}
    for i = 1,model.opt.batchSize do
        table.insert(offsets,math.ceil(torch.random(1,((#sequence)[1]-model.opt.rho))))
    end
    offsets = torch.LongTensor(offsets)
    for i = 1,model.opt.rho do
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
    model.rnn:zeroGradParameters()

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--jak se dela inicializace stavu lstm bunek
    -- local neco = torch.rand(model.opt.batchSize, model.opt.hiddenSize)
    -- model.rnn:get(1):get(1):get(1):get(2).userPrevOutput = neco:cuda()
    -- model.rnn:get(1):get(1):get(1):get(2).userPrevCell = neco:cuda()
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

    -- evaluate the loss function and its derivative wrt x, given mini batch
    local prediction = model.rnn:forward(inputs)
    local error = criterion:forward(prediction, targets)
    local gradOutputs = criterion:backward(prediction, targets)
    model.rnn:backward(inputs, gradOutputs)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    -- print(model.rnn:get(1):get(1):get(1):get(2).userGradPrevOutput)
    -- print(model.rnn:get(1):get(1):get(1):get(2).userGradPrevCell)
    -- os.exit()
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

	return error, x_grad
end

--sampling with current network
function sample(samples)

    samples = samples or 2*model.opt.rho

    local samplingRnn = model.rnn:get(1):get(1):get(1)
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count
    print('======Sampling==============================================')

    local prediction, sample

    -- -- generation with initialization by rho characters
    -- local randomStart = math.ceil(torch.random(1,((#sequence)[1]-model.opt.rho)))
    -- for i = randomStart,randomStart+model.opt.rho do
    --     io.write(model.numberToChar[sequence[i]])
    --     prediction = samplingRnn:forward(torch.CudaTensor{sequence[i]})
    -- end
    -- io.write('__|||__')
    -- io.flush()

    -- -- generation with initialization by random character
    -- local randomCharNumber = math.ceil(torch.random(1, #model.numberToChar))
    -- -- generation with initialization by specific character (space)
    local randomCharNumber = model.charToNumber[' ']
    prediction = samplingRnn:forward(torch.CudaTensor{randomCharNumber})

    for i=1,samples do
        prediction:exp()
        sample = torch.multinomial(prediction,1)

        io.write(model.numberToChar[sample[1][1]])

        prediction = samplingRnn:forward(sample[1])
    end
    io.write('\n')
    io.flush()
    print('============================================================')

    samplingRnn:training()--!!!!!! IMPORTANT switch back to remembering state

end

--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==
----==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--

-- --try to load model
if path.exists(opt.modelName) then
    model = torch.load(opt.modelName)
    print('Model '..opt.modelName..' loaded.')
    print('Parameters overriden.')
    print(model.opt)

    model.rnn:cuda()

    --load inputFile
    -- data loading and sequence creation
    text, sequence, charToNumber, numberToChar = readFile:processFile(model.opt.inputFile)
    sequence = sequence:cuda()
else
    --model not available, create new
    -- data loading and sequence creation

    text, sequence, charToNumber, numberToChar = readFile:processFile(opt.inputFile)
    sequence = sequence:cuda()

    --network creation
    local rnn = createLSTMNetwork(#numberToChar, opt.hiddenSize, opt.layers, opt.rho)

    rnn:mediumSerial()
    rnn:cuda()

    model = {}
    model.rnn = rnn
    model.opt = opt
    model.training_params = training_params

    model.charToNumber = charToNumber
    model.numberToChar = numberToChar

    --INICIALIZATION
    -- A1: initialization often depends on each dataset.
    --rnn:getParameters():uniform(-0.1, 0.1)
end

--create criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion:cuda()


x, x_grad = model.rnn:getParameters() -- w,w_grad

sample()

while true do
-- get weights and loss wrt weights from the model
    res, fs = model.training_params.algorithm(feval, x, model.training_params)
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    if model.training_params.evaluation_counter%25==0 then
        print(string.format('error for minibatch %4.1f is %4.7f', model.training_params.evaluation_counter, fs[1] / model.opt.rho))
    end
    if model.training_params.evaluation_counter%50==0 then
        collectgarbage()
    end
    if model.training_params.evaluation_counter%100==0 then
        sample()
    end
    if model.training_params.evaluation_counter%1250==0 then
        torch.save(model.opt.modelName, model)
        print("Model saved to "..model.opt.modelName)
    end
end

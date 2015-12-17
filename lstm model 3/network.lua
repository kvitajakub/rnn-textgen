--usual
require 'rnn'
require 'optim'

--local
require 'readFile'

inputFile = "../text/input.txt"
hiddenSize = 150
rho = 40
batchSize = 20
sgd_params = {
   learningRate = 0.25,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0.9,
   nesterov = true,
   dampening = 0
}


-- data loading and sequence creation
text, charToNumber, numberToChar = readFile:processFile(inputFile)
sequence = torch.Tensor(#text):zero()  --tensor representing chars as numbers, suitable for NLL criterion output
sequenceCoded = torch.Tensor(#text, #numberToChar):zero() --tensor for network input, 1 from N coding
for i = 1,#text do
    sequence[i] = charToNumber[text:sub(i,i)]
    sequenceCoded[i][sequence[i]] = 1
end

--network creation
-- rnn for training with Sequencer and negative log likelihood criterion
rnn = nn.Sequential()
rnn:add(nn.LSTM(#numberToChar, hiddenSize, rho))
rnn:add(nn.LSTM(hiddenSize, hiddenSize, rho))
rnn:add(nn.Linear(hiddenSize, #numberToChar))
rnn:add(nn.LogSoftMax())
rnn = nn.Sequencer(rnn)

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- minibatch computation
-- random subsequences from whole text
function nextBatch()
	local offsets, inputs, targets = {}, {}, {}

    for i = 1,batchSize do
        table.insert(offsets,math.ceil(torch.random(1,((#sequence)[1]-rho))))
    end
    offsets = torch.LongTensor(offsets)

    for i = 1,rho do
        table.insert(inputs,sequenceCoded:index(1,offsets))
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

    samples = samples or 2*rho

    local samplingRnn = rnn:get(1):get(1):clone()
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:remove(#samplingRnn.modules) --remove last layer LogSoftMax
    samplingRnn:add(nn.SoftMax()) --add regular SoftMax
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count

    print('======Sampling==============================================')

    local prediction, sample, sampleCoded
    local randomStart = math.ceil(torch.random(1,((#sequence)[1]-rho)))
    for i = randomStart,randomStart+rho do
        io.write(numberToChar[sequence[i]])
        prediction = samplingRnn:forward(sequenceCoded[i])
    end
    io.write('__|||__')

    for i=1,samples do
        sample = torch.multinomial(prediction,1)
        io.write(numberToChar[sample[1]])

        sampleCoded = torch.Tensor(#numberToChar):zero()
        sampleCoded[sample[1]] = 1

        prediction = samplingRnn:forward(sampleCoded)
    end
    io.write('\n')
    print('============================================================')
end


if path.exists('model.dat') then
    rnn = torch.load('model.dat')
    print('Model loaded.')
end

x, x_grad = rnn:getParameters() -- w,w_grad

while true do
-- get weights and loss wrt weights from the model
    res, fs = optim.sgd(feval, x, sgd_params)

    if sgd_params.evalCounter%20==0 then
        print(string.format('error for minibatch %4.1f is %4.7f', sgd_params.evalCounter, fs[1] / rho))
    end
    if sgd_params.evalCounter%100==0 then
        sample()
    end
    if sgd_params.evalCounter%250==0 then
        torch.save('model.dat', rnn)
        print("Model saved to model.dat")
    end
end

--usual
require 'rnn'
require 'optim'

--local
require 'readFile'

epochs = 30
rho = 5
hiddenSize = 100
batchSize = 100


-- nasteni dat a vytvoreni sekvenci
text, charToNumber, numberToChar = readFile:processFile("../text/input.txt")

sequence = torch.Tensor(#text):zero()
sequenceCoded = torch.Tensor(#text, #numberToChar):zero()
for i = 1,#text do
    sequence[i] = charToNumber[text:sub(i,i)]
    sequenceCoded[i][sequence[i]] = 1
end

--vytvoreni site
mlp = nn.Sequential()
mlp:add(nn.LSTM(#numberToChar, hiddenSize, rho))
mlp:add(nn.LSTM(hiddenSize, hiddenSize, rho))
mlp:add(nn.Linear(hiddenSize, #numberToChar))

samplingRnn = mlp:clone('weight','bias')
samplingRnn.modules[1]:evaluate() --nastaveni LSTM vrstev at si nepamatuji aktivace
samplingRnn.modules[2]:evaluate() --neni potreba pro sampling rnn
samplingRnn = samplingRnn:add(nn.SoftMax())

mlp:add(nn.LogSoftMax())
rnn = nn.Sequencer(mlp)


--loss funkce
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

--napocitame kolik minibatches je potreba
minibatches_in_epoch = (#sequenceCoded)[1]-batchSize-rho+1

--inicializace indexu dat ktery budeme pouzivat
offsets = {}
for i = 1,batchSize do
    table.insert(offsets,i)
end
offsets = torch.LongTensor(offsets)

-- method to compute a batch
function nextBatch()
	local inputs, targets = {}, {}
    epoch_offset = epoch_offset or 0

    local offsets_rho = offsets:clone()
    offsets_rho:add(epoch_offset)
    for i = 1,rho do
        table.insert(inputs,sequenceCoded:index(1,offsets_rho))
        offsets_rho:add(1)
        table.insert(targets,sequence:index(1,offsets_rho))
    end
    epoch_offset = (epoch_offset+1)%minibatches_in_epoch

	return inputs, targets
end

-- get weights and loss wrt weights from the model
x, x_grad = rnn:getParameters() -- w,w_grad

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. weigths is the vector of trainable weights,
-- it extracts a mini_batch via the nextBatch method
function feval(x_new)
	-- copy the weight if are changed
	if x ~= x_new then
		x:copy(x_new)
	end

	-- select a training batch
	local inputs, targets = nextBatch()

	-- reset gradients (gradients are always accumulated, to accommodate batch methods)
	x_grad:zero()

	-- evaluate the loss function and its derivative wrt x, given mini batch
	local prediction = rnn:forward(inputs)
	local error = criterion:forward(prediction, targets)
    local gradOutputs = criterion:backward(prediction, targets)
	rnn:backward(inputs, gradOutputs)

	return error, x_grad
end


function sample()

    local ind = torch.LongTensor(1)
    local prediction, sample, sampleCoded
    for i = 1,rho do
        ind[1] = i
        -- print('. '..sequenceCoded:index(1,ind))
        io.write(numberToChar[sequence[i]])
        prediction = samplingRnn:forward(sequenceCoded:index(1,ind))
    end
    io.write(' || ')
    for i=1,10 do
        sample = torch.multinomial(prediction[1],1)
        io.write(numberToChar[sample[1]])

        sampleCoded = torch.Tensor(1, #numberToChar):zero()
        sampleCoded[1][sample[1]] = 1

        prediction = samplingRnn:forward(sampleCoded)
    end
    io.write('\n')
end

sample()

sgd_params = {
   learningRate = 0.1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

for e = 1,epochs do

    for i = 1,minibatches_in_epoch do
    -- train a mini_batch of batchSize in parallel
        _, fs = optim.sgd(feval, x, sgd_params)

        if i%100==0 then
            print(string.format('error for minibatch %5f from %f is %f', sgd_params.evalCounter, minibatches_in_epoch, fs[1] / rho))
        end
        if i%1000==0 then
            sample()
        end
    end
end

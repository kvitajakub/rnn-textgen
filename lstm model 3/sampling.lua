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
   learningRate = 0.15,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0.6,
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


--sampling with current network
function sample(samples)

    samples = samples or 2*rho

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
    samplingRnn = rnn:get(1):get(1):clone()
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:remove(#samplingRnn.modules) --remove last layer LogSoftMax
    samplingRnn:add(nn.SoftMax()) --add regular SoftMax
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count

    for i = 1,10 do
        sample(200)
    end

else
    print("No file model.dat")
end

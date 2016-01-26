--usual
require 'rnn'
require 'optim'

--local
require 'readFile'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Sample a simple LSTM network for next character prediction.')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','model.dat','name of the model to be saved or loaded.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- --try to load model
if path.exists(opt.modelName) then
    rnn = torch.load(opt.modelName)
    print('Model '..opt.modelName..' loaded.')
    print(rnn.opt)

    --load inputFile
    -- data loading and sequence creation
    text, charToNumber, numberToChar = readFile:processFile(rnn.opt.inputFile)
    sequence = torch.Tensor(#text):zero()  --tensor representing chars as numbers, suitable for NLL criterion output
    sequenceCoded = torch.Tensor(#text, #numberToChar):zero() --tensor for network input, 1 from N coding
    for i = 1,#text do
        sequence[i] = charToNumber[text:sub(i,i)]
        sequenceCoded[i][sequence[i]] = 1
    end
else
    print("No file model.dat")
    os.exit()
end


--sampling with current network
function sample(samples)

    samples = samples or 2*rnn.opt.rho

    local samplingRnn = rnn:get(1):get(1):clone()
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:remove(#samplingRnn.modules) --remove last layer LogSoftMax
    samplingRnn:add(nn.SoftMax()) --add regular SoftMax
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count

    print('======Sampling==============================================')

    local prediction, sample, sampleCoded
    local randomStart = math.ceil(torch.random(1,((#sequence)[1]-rnn.opt.rho)))
    for i = randomStart,randomStart+rnn.opt.rho do
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

for i = 1,10 do
    sample(400)
end

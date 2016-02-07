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
cmd:text('Sample a simple LSTM network for next character prediction.')
cmd:text()
cmd:text('Options')
cmd:option('-modelName','model.dat','name of the model to be saved or loaded.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

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
    print("No file "..opt.modelName)
    os.exit()
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
end

sample(1500)

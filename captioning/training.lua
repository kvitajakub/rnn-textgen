--usual
require 'torch'
require 'optim'
--uncommon
require 'rnn'
require 'dpnn'
-- require 'cutorch'
-- require 'cunn'
--local
require 'cocodata'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a CNN-RNN network for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/coco/captions_train2014_small.json",'JSON file with the input data (captions, image names).')
cmd:option('-imageDirectory',"/storage/brno7-cerit/home/xkvita01/coco/train2014/",'Directory with the images with names according to the caption file.')
cmd:text()
cmd:option('-recurLayers',4,'Number of recurrent layers. (At least one.)')
cmd:option('-batchSize',20,'Minibatch size.')
cmd:option('-modelName','model.dat','File name of the saved or loaded model and training data.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


training_params = {
    algorithm = optim.adam,
    evaluation_counter = 0,

    learningRate=0.002
}


--not using global variables
function createRNN(input_output, recurrent_layers)

    local rnn = nn.Sequential()

    rnn:add(nn.OneHot(input_output))
    rnn:add(nn.LSTM(input_output, 512))
    for i=2,recurrent_layers do
        rnn:add(nn.LSTM(512, 512))
    end
    rnn:add(nn.Linear(512, input_output))
    rnn:add(nn.LogSoftMax())
    rnn = nn.Sequencer(rnn)
    return rnn, rnn:get(1):get(1):get(2)  --return network and first recurrent layer
end

--not using global variables
function createCNN()

    local cnn = nn.Sequential()

    cnn:add(nn.SpatialConvolution(3, 256, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(4, 4, 4, 4))

    cnn:add(nn.SpatialConvolution(256, 512, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(4, 4, 4, 4))

    cnn:add(nn.SpatialZeroPadding(1, 0, 1, 0))  --512*32*32

    cnn:add(nn.SpatialConvolution(512, 256, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(4, 4, 4, 4))

    cnn:add(nn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(4, 4, 4, 4))

    cnn:add(nn.Reshape(512))

    return cnn
end

cnn = createCNN()
print(cnn:forward(torch.ones(3,500,500)):size())

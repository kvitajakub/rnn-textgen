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
require 'sample'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluating a RNN language model for generating image captions.')
cmd:text()
cmd:text('Options')
cmd:option('-captionFile',"/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json",'JSON file with the input data (captions, image names).')
cmd:text()
cmd:option('-printError',10,'Print error once per N minibatches.')
cmd:option('-modelName','/storage/brno7-cerit/home/xkvita01/RNN/1.0000__3x300.torch','Filename of the model and training data.')
cmd:option('-saveOutput','error.torch','Save the computed error')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

function listCaptions(js)
    local captions = {}
    for i=1,#js['annotations'] do
        table.insert(captions, js['annotations'][i]['caption'])
    end
    return captions
end

if opt.modelName ~= "" and path.exists(opt.modelName) then
    model = torch.load(opt.modelName)

    model.opt.printError = opt.printError
    model.training_params.evaluation_counter = 0

    print('Model '..opt.modelName..' loaded.')
    print('Parameters overriden.')
    print(model.opt)

    js = loadCaptions(model.opt.captionFile)
    captions = listCaptions(js)

else

    error("Model not located.")

end

--create criterion
criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1)
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

function training()

    local inputs, targets = nextBatch()

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    model.rnn:zeroGradParameters()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    local prediction = model.rnn:forward(inputs)
    local error = {}

    for i=1,#prediction do
        table.insert(error,criterion:forward(prediction[i], targets[i]))
    end

    return error
end


errorEpoch = torch.Tensor(150):zero()
errorCount = torch.Tensor(150):zero()

epochNum = math.floor((model.training_params.evaluation_counter * model.opt.batchSize) / #captions)

--do one epoch of training
while model.training_params.evaluation_counter * model.opt.batchSize - epochNum * #captions < #captions do
-- get weights and loss wrt weights from the model
    error = training()
    model.training_params.evaluation_counter = model.training_params.evaluation_counter + 1

    local size
    if #error>150 then
        size = 150
    else
        size = #error
    end

    for i=1,size do
        errorEpoch[i] = errorEpoch[i] + error[i]
        errorCount[i] = errorCount[i] + 1
    end

    --print error
    if model.training_params.evaluation_counter%model.opt.printError==0 then
        print(string.format('minibatch %d (epoch %2.4f) ', model.training_params.evaluation_counter, (model.training_params.evaluation_counter*model.opt.batchSize)/#(js['annotations'])))
    end

    --collect garbage
    if model.training_params.evaluation_counter%10==0 then
        collectgarbage()
    end

    if model.training_params.evaluation_counter%300==0 then
        data = {errorEpoch, errorCount}
        torch.save(opt.saveOutput,data)
        print(" >>> Error saved "..opt.saveOutput)
    end

end

data = {errorEpoch, errorCount}
torch.save(opt.saveOutput,data)

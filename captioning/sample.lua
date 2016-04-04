
--model - as saved in training, no cuda
--table of tensors of images 3x500x500
--input character, initialization, usually not used
function sampleModel(model, images, input_char)

    input_char = input_char or "START"
    local descriptions = {}

    model.cnn:evaluate() --no need to remember history
    model.rnn:evaluate() --no need to remember history

    local cnn = model.cnn
    local rnnNoseq = model.rnn:get(1):get(1):get(1)
    local rnnLayer = rnnNoseq:get(1):get(2)

    cutorch.setDevice(2)
    rnnNoseq:forget()

    local input_tensor = torch.CudaTensor{model.charToNumber[input_char]}
    local prediction
    local description = ""
    local char

    local safeCounter = 0

    for i=1, images:size()[1] do
        cutorch.setDevice(1)
        cnn:forward(images[i])

        cutorch.setDevice(2)
        rnnNoseq:forget()

        cutorch.synchronizeAll()
        rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, cnn.output)

        prediction = rnnNoseq:forward(input_tensor)
        prediction:exp()
        sample = torch.multinomial(prediction,1)
        char = model.numberToChar[sample[1]]

        while char ~= "END" and safeCounter<200 do

            description = description .. char
            prediction = rnnNoseq:forward(sample)
            prediction:exp()
            sample = torch.multinomial(prediction,1)
            char = model.numberToChar[sample[1]]

            safeCounter = safeCounter + 1
        end

        table.insert(descriptions, description)
        description = ""
        safeCounter = 0
    end

    model.cnn:training()--!!!!!! IMPORTANT switch back to remembering state
    model.rnn:training()--!!!!!! IMPORTANT switch back to remembering state

    return descriptions
end


function printOutput(imageFiles, generatedDescriptions, correctDescriptions)
    print("========SAMPLING=====================================================")
    for i=1,#imageFiles do
        print(correctDescriptions[i])
        print("+++++++++++")
        print(generatedDescriptions[i])
    print("------------------------------------------------------------")
    end
    print("=====================================================================")
end

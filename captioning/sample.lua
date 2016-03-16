
--model - as saved in training, no cuda
--table of tensors of images 3x500x500
--input character, initialization, usually not used
function sample(model, images, input_char)

    input_char = input_char or "START"
    local descriptions = {}

    --tensor of tensors
    local size = images[1]:size():totable()
    table.insert(size, 1, #images)
    images = torch.cat(images):view(unpack(size))

    model:evaluate() --no need to remember history

    local cnn = model:get(1):get(1)
    local rnnNoseq = model:get(1):get(2):get(1):get(1):get(1)
    local rnnLayer = rnnNoseq:get(2)

    rnnNoseq:forget()

    local input_tensor = torch.Tensor{model.charToNumber[input_char]}
    local prediction
    local description = ""
    local char

    local safeCounter = 0

    for i=1, size[1] do
        cnn:forward(images[i])

        rnnNoseq:forget()

        rnnLayer.userPrevOutput = nn.rnn.recursiveCopy(rnnLayer.userPrevOutput, cnn.output)

        prediction = rnnNoseq:forward(input_tensor)
        prediction:exp()
        sample = torch.multinomial(prediction,1)
        char = model.numberToChar[sample[1]]

        while char ~= "END" and safeCounter<200 do

            description = description .. char
            prediction = rnnNoseq:forward(sample[1])
            prediction:exp()
            sample = torch.multinomial(prediction,1)
            char = model.numberToChar[sample[1]]

            safeCounter = safeCounter + 1
        end

        table.insert(descriptions, description)
        description = ""
        safeCounter = 0
    end

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

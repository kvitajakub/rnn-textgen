

function sample(model)

    local samplingRnn = model.rnn:get(1):get(1):get(1):get(1)
    samplingRnn:evaluate() --no need to remember history
    samplingRnn:forget() --!!!!!! IMPORTANT reset inner step count
    print('======Sampling==============================================')

    local prediction, sample, char
    local description = ""
    local safeCounter = 0

    -- -- generation with initialization by specific character (start character)
    local randomCharNumber = model.charToNumber['START']
    prediction = samplingRnn:forward(torch.CudaTensor{randomCharNumber})
    -- prediction = samplingRnn:forward(torch.CudaTensor{randomCharNumber})
    prediction:exp()
    sample = torch.multinomial(prediction,1)
    char = model.numberToChar[sample[1]]

    while char ~= "END" and safeCounter<200 do

        description = description .. char
        prediction = samplingRnn:forward(sample)
        prediction:exp()
        sample = torch.multinomial(prediction,1)
        char = model.numberToChar[sample[1]]

        safeCounter = safeCounter + 1
    end

    print(description)
    print('============================================================')

    samplingRnn:training()--!!!!!! IMPORTANT switch back to remembering state

    return description
end

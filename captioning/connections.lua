


function connectForward(model)
    local rnnLayers = 0
    local adaptOutput = model.adapt.output:split(model.opt.rnnHidden,model.adapt.output:dim())
    for i=1,model.rnn:get(1):get(1):get(1):get(1):get(1):size() do
        if model.opt.initLayers ~= 0 and rnnLayers >= model.opt.initLayers then
            break
        end
        local layer = model.rnn:get(1):get(1):get(1):get(1):get(1):get(i)
        if torch.isTypeOf(layer, nn.LSTM) then
            rnnLayers = rnnLayers + 1
            layer.userPrevCell = nn.rnn.recursiveCopy(layer.userPrevCell, adaptOutput[rnnLayers])
        end
    end
end


function connectBackward(model)
    local rnnLayers = 0
    local userGradPrevCell = {}
    for i=1,model.rnn:get(1):get(1):get(1):get(1):get(1):size() do
        if model.opt.initLayers ~= 0 and rnnLayers >= model.opt.initLayers then
            break
        end
        local layer = model.rnn:get(1):get(1):get(1):get(1):get(1):get(i)
        if torch.isTypeOf(layer, nn.LSTM) then
            table.insert(userGradPrevCell, layer.userGradPrevCell:clone())
            rnnLayers = rnnLayers + 1
        end
    end
    --concat table of tensors
    for i=2,#userGradPrevCell do
        userGradPrevCell[1] = userGradPrevCell[1]:cat(userGradPrevCell[i])
    end

    return userGradPrevCell[1]
end




function connectForward(model)
    local rnnLayers = 0
    for i=1,model.rnn:get(1):get(1):get(1):get(1):get(1):size() do
        if model.opt.initLayers ~= 0 and rnnLayers >= model.opt.initLayers then
            break
        end
        local layer = model.rnn:get(1):get(1):get(1):get(1):get(1):get(i)
        if torch.isTypeOf(layer, nn.LSTM) then
            layer.userPrevCell = nn.rnn.recursiveCopy(layer.userPrevCell, model.adapt.output)
            rnnLayers = rnnLayers + 1
        end
    end
end


function connectBackward(model)
    local rnnLayers = 0
    local userGradPrevCell = nil
    for i=1,model.rnn:get(1):get(1):get(1):get(1):get(1):size() do
        if model.opt.initLayers ~= 0 and rnnLayers >= model.opt.initLayers then
            break
        end
        local layer = model.rnn:get(1):get(1):get(1):get(1):get(1):get(i)
        if torch.isTypeOf(layer, nn.LSTM) then
            if not userGradPrevCell then
                userGradPrevCell = layer.userGradPrevCell:clone()
            else
                userGradPrevCell:add(layer.userGradPrevCell:clone())
            end
            rnnLayers = rnnLayers + 1
        end
    end
    userGradPrevCell = userGradPrevCell / rnnLayers
    return userGradPrevCell
end

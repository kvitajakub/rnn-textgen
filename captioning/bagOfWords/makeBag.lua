

function makeBag(js)

    histogram = {}
    for i=1,#js['annotations'] do
        local cap = string.split(string.lower(js['annotations'][i]['caption']),'[ .,"]')
        for j=1, #cap do
            if cap[j] ~= '' then
                if not histogram[cap[j]] then
                    histogram[cap[j]] = 1
                else
                    histogram[cap[j]] = histogram[cap[j]] + 1
                end
            end
        end
    end

    histTable = {}
    i=0
    for k,v in pairs(histogram) do
        i = i+1
        histTable[k] = i
    end

    bag = {}
    bag['length'] = i
    bag['data'] = histTable

return bag
end


function captionToBag(bag, caption)
    local tensor = torch.CudaTensor(bag['length']):zero()
    local cap = string.split(string.lower(caption),'[ .,"]')
    for j=1, #cap do
        if cap[j] ~= '' then
            tensor[bag['data'][cap[j]]] = 1
        end
    end
    return tensor
end

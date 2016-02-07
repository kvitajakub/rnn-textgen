require 'torch'

function loadCaptions(captionFileName)

    cjson = require "cjson"
    file = io.open(captionFileName,"r")
    content = file:read("*a")
    file:close()
    js = cjson.decode(content)

    return js
end

function generateCodes(captions)

    local text = ""
    for i=1,#captions['annotations'] do
        text = text .. captions['annotations'][i]['caption']
    end

    local histogram = {}
    for i =1,#text do
        histogram[text:sub(i,i)] = (histogram[text:sub(i,i)] or 0)+1
    end

    local numberToChar = {}
    local charToNumber = {}

    for k,_ in pairs(histogram) do
        table.insert(numberToChar,k)
        charToNumber[numberToChar[#numberToChar]] = #numberToChar
    end

    return charToNumber, numberToChar
end


function imageSample(captions, N, imageDirectory)

    imageDirectory = imageDirectory or ""
    N = N or 1

    local imagesUrl = {}
    local imagesCaption = {}

    if not captions then
        error("No captions.")
    end

    for j=1,N do

        local randomCaptionNumber = math.ceil(torch.random(1,#captions['annotations']))
        local captionText = captions['annotations'][randomCaptionNumber]['caption']
        local imageId = captions['annotations'][randomCaptionNumber]['image_id']
        local imageName

        local i=1
        while i<= #captions['images'] do
            if imageId == captions['images'][i]['id'] then
                imageName = captions['images'][i]['file_name']
                break
            end
            i = i+1
        end

        local imagePath = imageDirectory .. imageName

        table.insert(imagesUrl, imagePath)
        table.insert(imagesCaption, captionText)

    end

    return imagesUrl, imagesCaption
end


js = loadCaptions('../../../Diplomka-data/coco/annotations/captions_train2014_small.json')
images, captions = imageSample(js,3,"../../../Diplomka-data/coco/train2014_small/")

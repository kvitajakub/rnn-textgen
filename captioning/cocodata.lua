require 'image'

function loadCaptions(captionFileName)

    cjson = require "cjson"
    file = io.open(captionFileName,"r")
    content = file:read("*a")
    file:close()
    js = cjson.decode(content)

    return js
end

function generateCodes(js)

    local histogram = {}
    for i=1,#js['annotations'] do
        local cap = js['annotations'][i]['caption']
        for j=1, #cap do
            histogram[cap:sub(j,j)] = 0
        end
    end

    local numberToChar = {}
    local charToNumber = {}

    table.insert(numberToChar,"START")
    charToNumber[numberToChar[#numberToChar]] = #numberToChar

    for k,_ in pairs(histogram) do
        table.insert(numberToChar,k)
        charToNumber[numberToChar[#numberToChar]] = #numberToChar
    end

    table.insert(numberToChar,"END")
    charToNumber[numberToChar[#numberToChar]] = #numberToChar

    return charToNumber, numberToChar
end


function imageSampleRandom(captions, N, imageDirectory)

    imageDirectory = imageDirectory or ""
    N = N or 1

    local imagesPath = {}
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

        table.insert(imagesPath, imagePath)
        table.insert(imagesCaption, captionText)

    end

    return imagesPath, imagesCaption
end


function loadAndPrepare(imageFile, outputSize)
    outputSize = outputSize or 500

    if type(imageFile) == "table" then

        local images = {}

        for k,v in ipairs(imageFile) do
            table.insert(images, loadAndPrepare(v,outputSize))
        end

        return images

    else

        local im = image.load(imageFile,3) -- nChannel x height x width
        local s = im:size()

        if s[2] > s[3] then
            --height > width
            local cropNumber = (s[2]-s[3])/2
            im = image.crop(im,0,cropNumber,s[3],s[2]-cropNumber) --height and width flipped
        elseif s[2] < s[3] then
            --height < width
            local cropNumber = (s[3]-s[2])/2
            im = image.crop(im,cropNumber,0,s[3]-cropNumber,s[2]) --height and width flipped
        end

        im = image.scale(im,outputSize)

        return im

    end

end

function encodeCaption(caption, charToNumber)

    if type(caption)=="table" then

        local sequences = {}

        for k,v in ipairs(caption) do
            table.insert(sequences, encodeCaption(v,charToNumber))
        end

        return sequences
    else

        local sequence = torch.Tensor(caption:len()+2):zero()

        sequence[1] = charToNumber["START"]
        for i = 1,caption:len() do
            sequence[i+1] = charToNumber[caption:sub(i,i)]
            if sequence[i+1] == nil then
                error("No code for character.")
            end
        end
        sequence[caption:len()+2] = charToNumber["END"]

        return sequence
    end

end

--
-- js = loadCaptions('../../../Diplomka-data/coco/annotations/captions_train2014_small.json')
-- charToNumber, numberToChar = generateCodes(js)
-- imageFiles, captions = imageSampleRandom(js,5,"../../../Diplomka-data/coco/train2014_small/")
-- sequences = encodeCaption(captions,charToNumber)
-- images = loadAndPrepare(imageFiles)

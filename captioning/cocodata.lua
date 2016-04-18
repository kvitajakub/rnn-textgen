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


function imageSample(captions, N, imageDirectory, startIndex)

    if not captions then
        error("No captions.")
    end

    N = N or 1
    imageDirectory = imageDirectory or ""

    local imagesPath = {}
    local imagesCaption = {}
    local index

    for j=1,N do

        if startIndex then
            --going sequentially from starting point
            index = (startIndex+j-1) % (#captions['annotations']) + 1
        else
            --getting random captions from dataset
            index = math.ceil(torch.random(1,#captions['annotations']))
        end

        local captionText = captions['annotations'][index]['caption']
        local imageId = captions['annotations'][index]['image_id']
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
    outputSize = outputSize or 224

    if type(imageFile) == "table" then

        local images = {}

        for k,v in ipairs(imageFile) do
            table.insert(images, loadAndPrepare(v,outputSize))
        end

        --tensor of tensors
        local size = images[1]:size():totable()
        table.insert(size, 1, #images)
        images = torch.cat(images):view(unpack(size))

        return images:cuda()

    else

        local im = image.load(imageFile,3) -- nChannel x height x width

        im = image.scale(im, outputSize, outputSize) --scale to 224x224 (not keeping aspect ratio)

        im = im * 255 --change range from [0,1] to [0,255]

        local MEAN_PIXEL = {123.68, 116.779, 103.939} --mean pixel from VGG paper for subtracting

        im[1] = im[1] - MEAN_PIXEL[1] --subtracting mean values
        im[2] = im[2] - MEAN_PIXEL[2]
        im[3] = im[3] - MEAN_PIXEL[3]

        im[1], im[3] = im[3], im[1] -- RGB to BGR

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
-- imageFiles, captions = imageSample(js,5,"../../../Diplomka-data/coco/train2014_small/")
-- sequences = encodeCaption(captions,charToNumber)
-- images = loadAndPrepare(imageFiles)

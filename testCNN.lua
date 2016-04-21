require 'nn'
require 'image'


cjson = require "cjson"
file = io.open("/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014_small.json","r")
content = file:read("*a")
file:close()
js = cjson.decode(content)

synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end


model = torch.load("/storage/brno7-cerit/home/xkvita01/CNN/VGG_ILSVRC_16_layers_full.torch")
-- model = torch.load("/storage/brno7-cerit/home/xkvita01/CNN/nin_imagenet_full.torch")


local MEAN_PIXEL = {123.68, 116.779, 103.939} --mean pixel from VGG paper for subtracting

print("Number of images: "..#js['images'])

for i=1,#js['images'] do

    local im = image.load("/storage/brno7-cerit/home/xkvita01/COCO/train2014/"..js['images'][i]['file_name'],3) -- nChannel x height x width
    im = image.scale(im, 224, 224) --scale to 224x224 (not keeping aspect ratio)
    im = im * 255 --change range from [0,1] to [0,255]
    im[1] = im[1] - MEAN_PIXEL[1] --subtracting mean values
    im[2] = im[2] - MEAN_PIXEL[2]
    im[3] = im[3] - MEAN_PIXEL[3]
    im[1], im[3] = im[3], im[1] -- RGB to BGR

    o = model:forward(im)
    print(i.."  "..torch.max(o))
    o = o:squeeze()
    if torch.max(o) > 0.7 then
        m = -1
        for j=1,o:size()[1] do
            if torch.max(o) == o[j] then
                m = j
            end
        end
        print(js['images'][i]['file_name'].."   "..m.."   "..synset_words[m])
    end
    collectgarbage()
end

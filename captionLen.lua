cjson = require "cjson"

caption_file = "/home/jkvita/DATA/Diplomka-data/coco/annotations/captions_train2014.json"
-- caption_file = "/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json"

file = io.open(caption_file,"r")
content = file:read("*a")
file:close()

js = cjson.decode(content)

len = torch.Tensor(250):zero()

for i=1, #js['annotations'] do
    len[#(js['annotations'][i]['caption'])] = len[#(js['annotations'][i]['caption'])] + 1
end

for i=len:size()[1]-1,1,-1 do
    len[i] = len[i] + len[i+1]
end

require 'gnuplot'

x = torch.linspace(1,150,150)

gnuplot.raw('set key font "FreeSerif,18"')
gnuplot.title("Length distribution of training sequences")
gnuplot.xlabel("Length")
gnuplot.ylabel("Percentage of sequences")

gnuplot.movelegend('left','top')

gnuplot.grid(true)

gnuplot.axis({0,120,0,1})

gnuplot.plot({x,len:narrow(1,1,150)/len[1],'+'} )

gnuplot.plotflush()

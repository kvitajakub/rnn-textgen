cjson = require "cjson"

caption_file = "/storage/brno7-cerit/home/xkvita01/COCO/captions_train2014.json"

file = io.open(caption_file,"r")
content = file:read("*a")
file:close()

js = cjson.decode(content)

for i=1, #js['annotations'] do
    local j,k = math.random(#js['annotations']), math.random(#js['annotations'])
    js['annotations'][j],js['annotations'][k] = js['annotations'][k],js['annotations'][j]
end

content = cjson.encode(js)
file = io.open(caption_file,"w")

file:write(content)
file:close()

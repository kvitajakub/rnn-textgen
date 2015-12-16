local f = assert(io.open('../text/input.txt', "r"))
local text = f:read("*a")
f:close()

histogram = {}

for i =1,#text do
    histogram[text:sub(i,i)] = (histogram[text:sub(i,i)] or 0)+1
end

print(histogram)

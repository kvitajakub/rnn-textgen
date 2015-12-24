
--read file to string variable and create list of all characters in the file
--usage: text,charToNumber,numberToChar =  readFile:processFile("../text/input.txt")
readFile = {}
function readFile:processFile(filename)
    local f = assert(io.open(filename, "r"))
    local text = f:read("*a")
    f:close()

    histogram = {}

    for i =1,#text do
        histogram[text:sub(i,i)] = (histogram[text:sub(i,i)] or 0)+1
    end

    numberToChar = {}
    charToNumber = {}
    for k,_ in pairs(histogram) do
        table.insert(numberToChar,k)
        charToNumber[numberToChar[#numberToChar]] = #numberToChar
    end

    return text, charToNumber, numberToChar
end

-- text,charToNumber,numberToChar =  readFile:processFile("../text/input.txt")

-- print(charToNumber[text:sub(25,25)])

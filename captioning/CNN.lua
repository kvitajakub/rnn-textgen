
CNN = {}

function CNN.createCNN()

    local cnn = nn.Sequential()

    -- NiN model
    -- https://gist.github.com/mavenlin/d802a5849de39225bcc6
    cnn:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(96, 96, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(96, 96, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    cnn:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(256, 256, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(256, 256, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    cnn:add(nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(384, 384, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(384, 384, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    cnn:add(nn.Dropout(0.500000))
    cnn:add(nn.SpatialConvolution(384, 1024, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(1024, 1024, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(1024, 1000, 1, 1, 1, 1, 0, 0))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialAveragePooling(6, 6, 1, 1):ceil())

    model:add(nn.Reshape(1000))

    return cnn
end

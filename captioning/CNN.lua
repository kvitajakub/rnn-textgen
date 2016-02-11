
CNN = {}

function CNN.createCNN()

    local cnn = nn.Sequential()

    cnn:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(5, 5, 5, 5))

    cnn:add(nn.SpatialZeroPadding(1, 0, 1, 0))  --256*32*32

    cnn:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(5, 5, 5, 5))

    -- cnn:add(nn.SpatialConvolution(512, 256, 5, 5, 1, 1, 2, 2))
    -- cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(5, 5, 5, 5))
    --
    -- cnn:add(nn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
    -- cnn:add(nn.ReLU())
    -- cnn:add(nn.SpatialMaxPooling(4, 4, 4, 4))

    cnn:add(nn.Reshape(512))

    return cnn
end


RNN = {}

function RNN.createRNN(input_output, recurrent_layers)

    local rnn = nn.Sequential()

    rnn:add(nn.OneHot(input_output))
    rnn:add(nn.LSTM(input_output, 512))
    for i=2,recurrent_layers do
        rnn:add(nn.LSTM(512, 512))
    end
    rnn:add(nn.Linear(512, input_output))
    rnn:add(nn.LogSoftMax())
    rnn = nn.Sequencer(rnn)

    --INICIALIZATION
    -- A1: initialization often depends on each dataset.
    --rnn:getParameters():uniform(-0.1, 0.1)

    return rnn
end

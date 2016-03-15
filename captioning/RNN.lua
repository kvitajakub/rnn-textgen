
RNN = {}

function RNN.createRNN(input_output, recurrent_layers, hidden, dropout)

    recurrent_layers = recurrent_layers or 3
    hidden = hidden or 500
    dropout = dropout or true

    local rnn = nn.Sequential()

    rnn:add(nn.OneHotZero(input_output))
    rnn:add(nn.LSTM(input_output, hidden))
    if dropout then
        rnn:add(nn.Dropout(0.5))
    end
    for i=2,recurrent_layers do
        rnn:add(nn.LSTM(hidden, hidden))
        if dropout then
            rnn:add(nn.Dropout(0.5))
        end
    end
    rnn:add(nn.Linear(hidden, input_output))
    rnn:add(nn.LogSoftMax())

    rnn = nn.Serial(nn.Sequencer(nn.MaskZero(rnn,1)))

    return rnn
end

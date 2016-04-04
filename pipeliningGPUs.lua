--https://groups.google.com/forum/#!topic/torch7/bFjoGIpKQkk

--Pipelining modules across GPUs is actually super super simple. This is an example for a 2-GPU pipelined model:

-- model definition
w = {}; gW = {}
cutorch.setDevice(1)
pipe1 = nn.Sequential():add(nn.Linear(10,20)):cuda()
w[1], gW[1] = pipe1:getParameters()
cutorch.setDevice(2)
pipe2 = nn.Sequential():add(nn.Linear(20,30)):cuda()
w[2], gW[2] = pipe2:getParameters()

-- create your criterions on GPU2 as that is where your model ends
criterion = nn.MSECriterion()
--------------------------------------------------------------------------
-- create input data on GPU1 as that is where your model starts
cutorch.setDevice(1)
input = torch.randn(512, 10):cuda() -- 512 mini-batches
-- create target data on GPU2 as that is where your model ends
cutorch.setDevice(2)
target = torch.randn(512, 30):cuda() -- 512 mini-batches

---------------------------------------------------------------------------
cutorch.setDevice(1)
output1 = pipe1:forward(input)
cutorch.setDevice(2)
output2 = pipe2:forward(output1)
err = criterion:forward(output2, target)
df_do = criterion:backward(output2, target)
gradInput2 = pipe2:backward(output2, df_do)
cutorch.setDevice(1)
gradInput1 = pipe1:backward(output1, gradInput2)
-- then do two separate optim.sgds, one per GPU

--You can of course, abstract this out nicer, that your model is GPU-aware and automatically switches things, it should be pretty simple to do (and at the lua level)

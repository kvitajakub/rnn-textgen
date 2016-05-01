require 'gnuplot'

data = torch.load('error3x300_bag1.torch')

x = torch.linspace(1,150,150)

gnuplot.title("")

gnuplot.xlabel("")
gnuplot.ylabel("negative log likelihood")

gnuplot.movelegend('left','top')
gnuplot.raw('set key font "FreeSerif,18"')

gnuplot.grid(true)

-- gnuplot.axis({-3.2,3.2,-1.2,2.4})

gnuplot.plot({"Pretrained 3x300 RNN error",x,data[1]:cdiv(data[2]),'+-'} )
-- gnuplot.plot({"Pretrained 3x300 RNN error",x,data[2],'+-'} )

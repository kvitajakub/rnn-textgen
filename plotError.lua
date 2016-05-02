require 'gnuplot'


for k,v in ipairs(arg) do

    data = torch.load(v)
    name = string.split(v,'[.]')[1]
    gnuplot.pdffigure(name..".pdf")

    x = torch.linspace(1,90,90)
    gnuplot.title("")
    gnuplot.xlabel("")
    gnuplot.ylabel("negative log likelihood")

    gnuplot.movelegend('left','top')
    gnuplot.raw('set key font "FreeSerif,18"')

    gnuplot.grid(true)

    -- gnuplot.axis({-3.2,3.2,-1.2,2.4})

    gnuplot.plot({name,x,torch.cdiv(data[1],data[2]):narrow(1,1,90),'+-'} )

    gnuplot.plotflush()

end

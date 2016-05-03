require 'gnuplot'

d = {}
x = torch.linspace(1,90,90)

for k,v in ipairs(arg) do

    data = torch.load(v)
    name = string.split(v,'[.]')[1]

    table.insert(d,{name,x,torch.cdiv(data[1],data[2]):narrow(1,1,90),'+-'})

end

gnuplot.pdffigure("out.pdf")
gnuplot.raw('set terminal pdf size 4.5,3')


gnuplot.title("")
gnuplot.xlabel("Character position")
gnuplot.ylabel("Negative log likelihood")

gnuplot.movelegend('right','top')
gnuplot.raw('set key box')
gnuplot.raw('set key font "Courier,15"')

gnuplot.grid(true)

gnuplot.axis({0,91,0.4,2.2})

gnuplot.plot(unpack(d))
gnuplot.plotflush()

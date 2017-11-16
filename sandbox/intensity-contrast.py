

import pylab 

t = pylab.arange(0, 1, 0.0001) 


A = 1 
a = 1 
b = 1
B = 1 

f = A*pylab.exp(a*t)*pylab.exp(-b*t)

pylab.plot(f)
pylab.show()


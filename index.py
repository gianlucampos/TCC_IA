import pylab as pl

x = [0, 1]
xTicks = ['LONDON', 'PARIS']
y = [2.39, 3.41]
pl.xticks(x, xTicks)
pl.xticks(range(2), xTicks, rotation=45)  # writes strings with 45 degree angle
pl.plot(x, y, '*')
pl.show()

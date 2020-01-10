import numpy as np
import tqdm

# input arguments
maxiter = 500
npts = 1000
xmin = -0.74877
xmax = -0.74872
ymin = 0.065053
ymax = 0.065103

x = np.linspace(xmin,xmax,npts) # real
y = np.linspace(ymin,ymax,npts) # imag

xv_real, yv = np.meshgrid(x,y)
yv_imag = yv * 1j;

z = xv_real + yv_imag

# get the number of iteration until it becomes unstable through elimination
mb_set = np.ones(npts**2).reshape(npts,npts)

c = z.copy()
m = mb_set.copy().astype(np.bool)
m_prev = m.copy()
for _ in tqdm.tqdm(range(maxiter)):
	m[m_prev] = np.abs(z[m_prev]) <= 2
	mb_set += m.astype(np.int)
	z[m] = np.power(z[m],2) + c[m]
	m_prev = m.copy()
	
# plot the set
import matplotlib.pyplot as plt
plt.imshow(mb_set)
plt.show()
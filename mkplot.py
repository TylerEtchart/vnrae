import pickle
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
from scipy import signal

SMOOTH = True

#FILE = 'losses_vnrae_1000.pkl'
FILE = 'losses_vnrae_300.pkl'

#plt.title("VNRAE Loss (smoothed)\nembedding size 1000")
plt.title("VNRAE Loss (smoothed)\nembedding size 300")

losses = pickle.load(open(FILE, 'rb'))

#print(losses)


x_ticks = []
for l in range(len(losses)):
    x_ticks.append(1000*l)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

if SMOOTH:

    #spline
    #xnew = np.linspace(x_ticks[0],x_ticks[-1],1000)
    #losses_smooth = spline(x_ticks,losses,xnew)
    #plt.plot(xnew, losses_smooth)

    #convolution
    window = signal.gaussian(51, std=7)
    losses_smooth = np.convolve(losses, window, mode='valid')
    plt.plot(x_ticks[:len(losses_smooth)], losses_smooth)
else:
    plt.plot(x_ticks,losses)


plt.xlabel('training steps')
plt.ylabel('ELBO loss')

#plt.savefig('losses_vnrae_1000.png')
plt.savefig('losses_vnrae_300.png')


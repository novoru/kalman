import matplotlib.pyplot as plt
from numpy import *
from random import random

class kalman(object):
    
    def __init__(self, x0, p0, a, b, c, vv, vw):
        self.x_ = 0       # A priori estimate state
        self.x  = x0      # A posteriori estimate state
        self.p_ = 0       # A priori error covariance matrix
        self.p  = p0      # A posteriori error covariance matrix
        self.a  = a       # Transition model
        self.b  = b       # 
        self.c  = c       # Observation model
        self.g  = 0       # Kalman gain
        self.vv = vv      # Variance of process noise
        self.vw = vw      # Variance of observation noise

    def predict(self):
        self.x_ = self.a * self.x
        self.p_  = self.a * self.p * self.a.T + self.vv * self.b * self.b.T
        
    def update(self, y):
        self.g = (self.p_ * self.c) / (self.c.T * self.p_ * self.c + self.vw)
        self.x = self.x_ + self.g * (y - self.c.T * self.x_)
        self.p = (1 - self.g * self.c.T) * self.p_

if __name__ == "__main__":

    x0 = 0.
    p0 = 0.
    a  = array(1)
    b  = array(1)
    c  = array(1)
    vv = 1.
    vw = 5.

    k = kalman(x0, p0, a, b, c, vv, vw)

    n = 300
    T = 2*pi

    s  = []
    r  = []
    rk = []
    t = arange(0., T, T/n)

    for i in t:
        k.predict()
        
        # True value
        x = 20. * sin(i) + sqrt(vv) *2*(random()-0.5)
        s.append(x)

        # Observation value
        y = x + sqrt(vw) * 2*(random()-0.5)
        k.update(y)
        r.append(y)

        # Estimate value
        rk.append(k.x)

    fig, ax = plt.subplots()

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Simulation result")

    ms = 2
    ax.plot(t, s, color='g', linestyle='--', marker='+', markersize=ms, label='True value')
    ax.plot(t, r, color='r', linestyle=':', marker=',', markersize=ms, label='Observation value')
    ax.plot(t, rk, color='b', linestyle='-', marker='.', markersize=ms, label='Estimate value')

    legend = ax.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()

    plt.show()

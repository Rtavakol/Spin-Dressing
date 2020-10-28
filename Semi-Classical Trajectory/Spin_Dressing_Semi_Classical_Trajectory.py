"""
Semi-Classical trajectory of dressed spin, for more details on dressed spin check out here:
https://summit.sfu.ca/item/17959

"""
__author__ = 'Min'
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

from scipy import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Runge–Kutta Algorithm
def rKN(x, fx, n, hs, dress_par_x, dress_par_y, dress_freq):
    k1, k2, k3, k4, xk = [], [], [], [], []
    for i in range(n):
        k1.append(fx[i](x,dress_par_x, dress_par_y, dress_freq)*hs)
    for i in range(n):
        xk.append(x[i] + k1[i]*0.5)
    for i in range(n):
        k2.append(fx[i](xk,dress_par_x, dress_par_y, dress_freq)*hs)
    for i in range(n):
        xk[i] = x[i] + k2[i]*0.5
    for i in range(n):
        k3.append(fx[i](xk,dress_par_x, dress_par_y, dress_freq)*hs)
    for i in range(n):
        xk[i] = x[i] + k3[i]
    for i in range(n):
        k4.append(fx[i](xk,dress_par_x, dress_par_y, dress_freq)*hs)
    for i in range(n):
        x[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
    return x

dress_par_x = 1.0 # dressing parameter x
dress_par_y = 0.1 # dressing parameter y
dress_freq = 1000.0 # Frequency of dressing field
num_dress_cyc = 4. # Number of dressing cycles
total_time = int(1e6 * num_dress_cyc / dress_freq + 10) ; delta_t = 1e-6 # Total time of interaction in us and also time resolution in s


Larmor_freq = dress_par_y * dress_freq # Define Larmor precession frequency
modified_larmor_freq = Larmor_freq * special.jv(0,dress_par_x) # Modified precession frequency calculated using modified gyromagnetic ratio
                                                                   # (zeroth order Bessel function)

# Class of 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# Terms in equation of motion
#Sx'(t) = ...
def fa1(x,dress_par_x, dress_par_y, dress_freq):
    w_l = dress_par_y * dress_freq * 2.0 * np.pi
    return w_l * x[1]
#Sy'(t) = ...
def fb1(x,dress_par_x, dress_par_y, dress_freq ):
    w = dress_par_x*dress_freq * 2.0 * np.pi
    w_l = dress_par_y*dress_freq * 2.0 * np.pi
    return w * x[2] *np.cos(x[3]) - w_l * x[0]
#Sz'(t) = ...
def fc1(x,dress_par_x, dress_par_y, dress_freq):
    w = dress_par_x*dress_freq * 2.0 * np.pi
    return - w * x[1] * np.cos(x[3])
#(omegat)' = omega
def fd1(x,dress_par_x, dress_par_y, dress_freq):
    w_d = dress_freq * 2.0 * np.pi
    return w_d

def plot3d(xs1,ys1,zs1,xs2,ys2,zs2 ):
    global dress_par_x
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_box_aspect([1.1, 1.1, 1])
    #ax.set_aspect("equal") # Recently equal feature does not work in matplotlib!

    ax.text(0.05, 0.05, 0.7, r'B$_{0}$', (0,0,0),fontsize=14,color='red')
    ax.text(0.2, 0.4, 0, r'B$_{d}$', (0,0,0),fontsize=14,color='green')
    ax.text(-0.3, -1.3, 0, 'M', (0,0,0),fontsize=14,color='black')
    subtitle = 'Dressing Parameters'  + '\n' '$y$=' + str(0.1) + '   '  + '$x$={:.2f}'.format(dress_par_x) + '\n' 'f$_{d}$=' + str(int(dress_freq)) + '  ' + 'f$_{L}$=' + \
           str(int(Larmor_freq)) + '   ' + 'f$_{mod}$=' + str(int(modified_larmor_freq))

    fig.suptitle(subtitle, fontsize=12)
    ax.text(-4, -0.5, -4, 'By: R. Tavakoli Dinani, SFU, 2016', (0,0,0), weight='bold',fontsize=10,color='k', alpha = 0.5)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # To draw surface of Bloch Sphere
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax._axis3don = False  # To hide axis
    ax.plot_surface(x, y, z, rstride=2, cstride=2, linewidth=0.1 , color='w', alpha = 0.20, shade = True)

    ax.text(1.1, 0, 0, 'X', (0,0,0),fontsize=16,color='blue')
    ax.text(0, 1.1, 0, 'Y', (0,0,0),fontsize=16,color='green')
    ax.text(0, 0, 1.1, 'Z', (0,0,0),fontsize=16,color='red')

    xc, yc, zc = [], [], []
    for i in range(0,100,1):
        xc.append(np.cos(i*np.pi/50))
        yc.append(np.sin(i*np.pi/50))
        zc.append(0.0)
    ax.plot(xc, yc, zc,linestyle='--', linewidth=0.5 , c='k')
    ax.plot([1,-1], [0,0], [0,0],linestyle='--', linewidth=2 , c='b')
    ax.plot([0,0], [1,-1], [0,0],linestyle='--', linewidth=2 , c='g')
    ax.plot([0,0], [0,0], [1,-1],linestyle='--', linewidth=2 , c='r')

    # Draw arrows
    ax = fig.gca(projection='3d')
    a1 = Arrow3D([0,0],[0,0],[0,1], mutation_scale=10, lw=1, arrowstyle="-|>", color="r") # to plot arrow
    a2 = Arrow3D([0,1],[0,0],[0,0], mutation_scale=10, lw=1, arrowstyle="-|>", color="b") # to plot arrow
    a3 = Arrow3D([0,0],[0,1],[0,0], mutation_scale=10, lw=1, arrowstyle="-|>", color="g") # to plot arrow
    a4 = Arrow3D([0,xs1[-1]],[0,ys1[-1]],[0,zs1[-1]], mutation_scale=10, lw=2, arrowstyle="-|>", color="k") # to plot arrow

    a6 = Arrow3D([0, xs2[-1]], [0, ys2[-1]], [0, zs2[-1]], linestyle='--', mutation_scale=10, lw=2, arrowstyle="-|>",
                 color="k")  # to plot arrow
    a7 = Arrow3D([0,0],[0,0],[0,1], mutation_scale=10, linestyle='-' , lw=2, arrowstyle="-|>", color="r")
    a8 = Arrow3D([0,0],[0,1],[0,0], mutation_scale=10, linestyle='-' , lw=2, arrowstyle="-|>", color="g")
    a9 = Arrow3D([0, 2*xs1[-1]], [0, 2*ys1[-1]], [0, 2*zs1[-1]], linestyle='-' , mutation_scale=10, lw=1, arrowstyle="-", color="k")
    a10 = Arrow3D([0, 2 * xs2[-1]], [0, 2 * ys2[-1]], [0, 2 * zs2[-1]], linestyle='-', mutation_scale=10, lw=1,
                  arrowstyle="-", color="k")

    ax.add_artist(a1)
    ax.add_artist(a2)
    ax.add_artist(a3)
    ax.add_artist(a4)
    ax.add_artist(a9)
    ax.add_artist(a6)
    ax.add_artist(a7)
    ax.add_artist(a8)
    ax.add_artist(a10)

    # Trajectory of the tip of nuclear magnetization for a dressed and undressed spin
    ax.plot(xs1, ys1, zs1,linewidth=2, c='k')
    ax.plot(xs2, ys2, zs2,linestyle='--' ,linewidth=2, c='k')

    # Save the figure
    params = [dress_par_y, dress_par_x, dress_freq, Larmor_freq, modified_larmor_freq]
    file_name = 'Dressing Parameters_y {}, x {:.2f}, fd {:.0f}, fL {:.0f}, fmod {:.0f}.png'.format(*params)
    fig.savefig(file_name,dpi=500)
    #plt.show()

def VDP1(dress_par_x, dress_par_y, dress_freq):
    global total_time, delta_t
    f = [fa1, fb1, fc1,fd1]
    x = [1., 0.0, 0.0, 0]

    Sx1=[]; Sy1=[]; Sz1=[]
    for i in range(total_time):
        x = rKN(x, f, 4, delta_t, dress_par_x, dress_par_y, dress_freq)
        Sx1.append(x[0]); Sy1.append(x[1]); Sz1.append(x[2])


    f = [fa1, fb1, fc1,fd1]
    x = [1., 0.0, 0.0, 0]
    Sx2=[]; Sy2=[]; Sz2=[]
    for i in range(total_time):
        x = rKN(x, f, 4, delta_t, 0.0, dress_par_y, dress_freq)
        Sx2.append(x[0]); Sy2.append(x[1]); Sz2.append(x[2])

    plot3d(Sx1,Sy1,Sz1,Sx2,Sy2,Sz2)


for x in np.linspace(0.01, 20, 60):
    dress_par_x = x
    modified_larmor_freq = Larmor_freq * special.jv(0, x)
    VDP1(x, dress_par_y, dress_freq)






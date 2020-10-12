"""
To animate the semiclassical trajectory of spin magnetization
under spin dressing conditions. The animation can be save as .mp4 or any other version.
dressing parameters:
x = gamma * B_d / omega_d
y  = gamma * B_0 / omega_d

"""

import numpy as np
from scipy import integrate
from scipy import special

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation


dress_freq = 1000.0 # Frequency of dressing field
Larmor_freq = 200.0 # f_L = gamma * B_0 / (2 * pi) Larmor Precession frequency of spin at a given static field B_0.
                    # Here I used Hydrogen at ~ 4.7 uT corresponds to f_L = 200 Hz

dress_y = Larmor_freq/dress_freq ; dress_x = 1.0   # Defining dressing parameter y (dress_y) and x (dress_x)
mod_larmor_fre = Larmor_freq * special.jv(0,dress_x) # Modified Larmor frequency obtained from theory
dress_duration_time = 0.002  # Duration of dressing pulse in s


dress_cyc = 4.0  # Number of dressing cycle at dressing frequency dress_freq
dress_duration_time = dress_cyc / dress_freq # Duration of dressing pulse in s
num_time_div = 200
dt = dress_duration_time / num_time_div

class Arrow3D(FancyArrowPatch):
    """Can be used to draw 3D arrow"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def time_deriv(y,t0):
    """ Compute the time derivative"""
    global dress_y, dress_x, dress_freq
    w = dress_x * 2.0 * np.pi * dress_freq
    w_l = dress_y * 2.0 * np.pi * dress_freq
    w_d = 2.0 * np.pi * dress_freq
    return [w_l * y[1], w * y[2] * np.cos(w_d * y[3]) - w_l * y[0],-w * y[1] * np.cos(w_d * y[3]), 1.0]

x0 = [[1.0, 0.0, 0.0, 0.0 ]] # Initial direction of Magnetization before applying dressing pulse
t = np.linspace(0, dress_duration_time, num_time_div) # 0.002 means 2 ms
x_t = np.asarray([integrate.odeint(time_deriv, x0i, t) for x0i in x0])


# Set up figure & 3D axis for animation
def set_up_3D():
    global dress_freq, dress_x, dress_y, Larmor_freq, mod_larmor_fre
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_aspect("auto")

    # Magnetic field labels, B_0: static field, B_d: dressing field
    ax.text(1.2, 0, 0, r'B$_{d}$', (0, 0, 0), fontsize=16, color='blue')
    ax.text(0, 0, 1.2, r'B$_{0}$', (0, 0, 0), fontsize=16, color='red')

    # Add dressing parameters as subtitle
    subtitle = 'Dressing parameters: y = {}, x = {}'.format(dress_y, dress_x) + '\n' \
               'f$_{d}$=' + str(int(dress_freq)) + \
               r' (Hz), f$_{L}$=' + str(int(Larmor_freq)) + \
               r' (Hz), f$_{mod}$=' + str(int(mod_larmor_fre)) + ' (Hz)'



    plt.figtext(0.15, 0.02, subtitle, fontsize=12, color='k', ha='left', fontweight='bold')
    fig.suptitle(subtitle, fontsize=14, fontweight='bold')
    a1 = Arrow3D([0,0],[0,0],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="r") # to plot arrow
    a2 = Arrow3D([0,1],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="b") # to plot arrow
    a3 = Arrow3D([0,0],[0,1],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g") # to plot arrow
    ax.add_artist(a1)
    ax.add_artist(a2)
    ax.add_artist(a3)
    xc, yc, zc = [], [], []
    for i in range(0, 100, 1):
        xc.append(np.cos(i * np.pi / 50))
        yc.append(np.sin(i * np.pi / 50))
        zc.append(0.0)
    ax.plot(xc, yc, zc, linestyle='--', linewidth=1, c='k')
    ax.plot([1, -1], [0, 0], [0, 0], linestyle='--', linewidth=1, c='b')
    ax.plot([0, 0], [1, -1], [0, 0], linestyle='--', linewidth=1, c='g')
    ax.plot([0, 0], [0, 0], [1, -1], linestyle='--', linewidth=1, c='r')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=5, cstride=5, linewidth=0, color='w', alpha = 0.5)

    color = 'k'
    lines = sum([ax.plot([], [], [], '-', c=color)], [])
    pts = sum([ax.plot([], [], [], 'o', c=color)], [])


    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_zlim((-1.1, 1.1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('on')

    ax.view_init(30, 0)
    return fig, ax, lines, pts


fig, ax, lines, pts = set_up_3D()

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z, tt = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    num = float(i) * dt * 1000.0
    num1 = '{:04.3f}'.format(num)
    lab = str(num1) + 'msec'
    if num/1000.0 > dress_cyc * (1.0/dress_freq) :
        lab = lab + '\n' r'B$_{d}$ is OFF'
    else:
        lab = lab + '\n' r'B$_{d}$ is ON'
    fig.suptitle('Time = ' +  lab  , fontsize=14, fontweight='bold')

    # Use view_init to adjust view of 3D bloch sphere.
    # The 2nd argument can be used to set speed of rotationg camera
    # -1.435 fpr 2000 data point and fL = 200 Hz
    ax.view_init(30, -2.0 * 0.715 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=800, interval=30, blit=True)

# Set the filename of video
filename = 'y=' + str(dress_y) + 'x=' + str(dress_x) + 'fL=' + str(dress_freq) + 'Hz' + '.mp4'

# Save as mp4. This requires mplayer or ffmpeg packages to be installed
#anim.save(filename, fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()

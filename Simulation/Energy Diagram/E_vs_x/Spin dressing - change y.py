import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy import linalg


delta_x = 0.1
dress_par_y = 0.9
N = 100   # Even number
num = 160
MM = N -1  # Odd number

def modify_even_diagonal(matrix, index, y):

    matrix[index, index] = y/2.0
    return matrix

def modify_odd_diagonal(matrix, index, y):

    matrix[index, index] = -y/2.0
    return matrix

def diagonal_elements(matrix, y):
    (n, m) = matrix.shape
    for i in range(0, n, 2):
        matrix = modify_even_diagonal(matrix, i, y)
        matrix = modify_odd_diagonal(matrix, i - 1, y)

    return matrix

def diagonal_elements_potons_number(matrix):
    (n,m) = matrix.shape
    for i in range(0, n, 2):
        matrix[i,i] = matrix[i,i] + N/2.0 - i/2.0
        matrix[i+1,i+1] = matrix[i+1,i+1] + N/2.0 - i/2.0
    return matrix

def modify_off_diagonal_even(matrix, index, x):

    try:
        if index +1 <= N :
            matrix[index, index + 1] = x/4.0
            matrix[index + 1, index] = x/4.0
    except Exception:
        pass
    try:
        if (index-3) >= 0 :
            matrix[index, index - 3] = x/4.0
            matrix[index - 3, index] = x/4.0
    except Exception:
        pass
    return matrix

def modify_off_diagonal_odd(mmatrix, index, x):

    try:
        if index + 1 <= N :
            mmatrix[index, index + 3] = x/4.0
            mmatrix[index + 3, index] = x/4.0
    except Exception:
        pass
    try:
        if (index - 1) >= 0 :
            mmatrix[index, index - 1] = x/4.0
            mmatrix[index - 1, index] = x/4.0
    except Exception:
        pass
    return mmatrix

def off_diagonal_elements(matrix, x):

    (n, m) = matrix.shape
    for i in range(1, n + 1, 2):
        matrix = modify_off_diagonal_even(matrix, i, x)
        matrix = modify_off_diagonal_odd(matrix, i + 1, x)

if __name__ == '__main__':
    s = (N, N)
    j = (num,MM)

    zeors = np.zeros(s)
    zeors2 = np.zeros(j)
    z = Matrix(zeors2)

    for i in range(0,num,1):
        zeors = np.zeros(s)
        new_matrix = diagonal_elements(zeors, dress_par_y)
        diagonal_elements_potons_number(new_matrix)
        off_diagonal_elements(new_matrix, i*delta_x)

        b = linalg.eigvals(new_matrix)
        b = b - N/4.0
        c = b[int(N/2 - MM/2 + 1): int(N/2 + MM/2 + 1),]

        #c=b[0:M,]
        f = np.array(c)
        d = f.real

        z[i,0] = i*delta_x
        for k in range(1,MM,1):
            z[i,k] = d[k-1,]

        file_name = 'Dressing parameter y =' + str(dress_par_y) + '.txt'
        np.savetxt(file_name, z)



dataset = np.genfromtxt(fname=file_name,skip_header=1)
x=dataset[:,0]
plt.figure(1)
for h in range(1, MM ,1):
    y=dataset[:,h]

    plt.plot(x,y,'r.')

    x1, x2 = 0, 8
    y1, y2 = -0.5, 0.5
    plt.xlim([x1, x2])
    plt.ylim([y1,y2])
    plt.xlabel('Dressing Parameter-y', fontsize=14)
    plt.ylabel('E/$\hbar \omega_{d}$ - <n>', fontsize=14)
    plt.xticks( [w/2.0 for w in range(16)] )
    plt.yticks([(v-4)/4.0 for v in range(9)])




txt = 'y = ' + str(dress_par_y)
filename = 'Dressing parameter y =' + str(dress_par_y) + '.jpg'
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }
plt.text(6.5, 0.7, txt  , fontdict=font) # first two numbers represent the position of text in plot
plt.suptitle('Energy Diagram', fontsize=20)
plt.savefig(filename)
plt.show()

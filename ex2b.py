"""Write a code implementing the five-point Laplacian scheme. 
This applies to POisson pb: uxx + uyy = f, on [0,1]x[0,1] with 
Dirichlet boundary conditions. Check that you get the
expected order of convergence, e.g. on a problem with the exact solution
uexact(x, y) = sin(4π(x + y)) + cos(4πxy)"""

import numpy as np
from scipy.sparse import spdiags, eye, kron
import matplotlib.pyplot as plt
from typing import Literal


# function suggested from slides
def poisson5(m):
    e = np.ones(m)
    S = spdiags([e, -2*e, e], [-1, 0, 1], m, m)
    I = eye(m, format='csr')
    A = kron(I, S) + kron(S, I)
    A = (m + 1)**2 * A #scaling from grid spacing
    return A

def poisson9(m):
    pass

def Laplacian_scheme(
        m, # number of interior points
        f, # function to solve
        g, # function for Dirichlet boundary conditions
        n: Literal[5,9] = 5 # default -5 points Poisson
):

    # full grid including boundary
    x = np.linspace(0, 1, m+2)
    y = np.linspace(0, 1, m+2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # f on interior points (vectorized)
    F = f(X[1:-1, 1:-1], Y[1:-1, 1:-1]).reshape(m*m)

    # choose stencil
    if n == 5:
        A = poisson5(m)
    else:
        A = poisson9(m)

    # Correct RHS = Right Hand-Side
    # we add the analytical value of g 
    RHS = F.copy()

    # bottom boundary: fixed i=0, variate y from j=1 to m
    bottom = g(x[0], y[1:-1])
    RHS[:m] += bottom

    # top boundary: i=m+1, j=1..m
    top = g(x[-1], y[1:-1])
    RHS[-m:] += top

    # left boundary: i=1..m, j=0
    left = g(x[1:-1], y[0])
    RHS[::m] += left

    # right boundary: i=1..m, j=m+1
    right = g(x[1:-1], y[-1])
    RHS[m-1::m] += right


    return A, RHS



        

# Test function
u = lambda x, y: np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

# Grid
h = 0.05
X, Y = np.meshgrid(np.arange(0, 1 + h, h),
                   np.arange(0, 1 + h, h))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u(X, Y), cmap='viridis')
plt.show()

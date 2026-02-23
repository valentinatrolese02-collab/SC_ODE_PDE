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
    e = np.ones(m)
    S = spdiags([-e, -10*e, -e], [-1, 0, 1], m, m)
    I = spdiags([-0.5*e, e, -0.5*e], [-1, 0, 1], m, m)
    A = (1/6) * (m + 1)**2 * (kron(I, S) + kron(S, I))
    return A

def Laplacian_scheme(
        f, # function to solve
        g, # function for Dirichlet boundary conditions
        m: int = 10, # number of interior points
        n: Literal[5,9] = 5 # default -5 points Poisson
):

    # full grid including boundary
    x = np.linspace(0, 1, m+2)
    y = np.linspace(0, 1, m+2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # f on interior points (vectorized)
    F = f(X[1:-1, 1:-1], Y[1:-1, 1:-1]).reshape(m*m)

    # Correct RHS = Right Hand-Side: function we are going to correct by
    # adding g evaluated on boundary poins, accordingly to the laplancian scheme used
    RHS = F.copy()

    # choose stencil
    if n == 5:
        A = poisson5(m)

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
    
    else:
        A = poisson9(m)
        # bottom boundary: fixed i=0, variate y from j=1 to m
        bottom = g(x[0], y[1:-1])
        RHS[:m] += 4*bottom 

        # top boundary: i=m+1, j=1..m
        top = g(x[-1], y[1:-1])
        RHS[-m:] += top

        # left boundary: i=1..m, j=0
        left = g(x[1:-1], y[0])
        RHS[::m] += left

        # right boundary: i=1..m, j=m+1
        right = g(x[1:-1], y[-1])
        RHS[m-1::m] += right

    return x, y, A, RHS.flatten() # outpus also the grid


# Test function
u = lambda x, y: np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)
u_xx = lambda x, y: - 16* np.pi**2 * (np.sin(4 * np.pi * (x + y)) + y**2 * np.cos(4 * np.pi * x * y))
u_yy = lambda x, y: - 16* np.pi**2 * (np.sin(4 * np.pi * (x + y)) + x**2 * np.cos(4 * np.pi * x * y))

x, y, A, b = Laplacian_scheme(u, lambda x,y : 0*x)
u_vec = np.linalg.solve(A.toarray(), b)

m = int(np.sqrt(len(u_vec)))
U = np.zeros((m+2, m+2))
U[1:-1, 1:-1] = u_vec.reshape(m, m)

# Complete grid
X, Y = np.meshgrid(x, y, indexing='ij')

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis')
plt.show()


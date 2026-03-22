import numpy as np
import matplotlib.pyplot as plt
import math

def Amult(U, m):
    """
    Matrix-free implementation of the 2D negative Laplacian (-Ah * U).
    Uses array shifting to avoid explicit matrix storage and padding.
    """
    U2d = U.reshape((m, m))
    # this is true if we assume the row-wise ordering

    AU_2d = 4.0 * U2d

    AU_2d[:-1, :] -= U2d[1:, :]  # Down neighbor
    AU_2d[1:, :]  -= U2d[:-1, :] # Up neighbor
    AU_2d[:, :-1] -= U2d[:, 1:]  # Right neighbor
    AU_2d[:, 1:]  -= U2d[:, :-1] # Left neighbor
    
    # Scale by h^2 where h = 1 / (m + 1)
    h2 = (1.0 / (m + 1))**2
    return -(AU_2d / h2).flatten()

def Vcycle(U, omega, nsmooth, m, F):
    """Approximately solve: A*U = F"""
    h = 1.0 / (m + 1)
    l2m = math.log2(m + 1)
    
    # Assertions to ensure dimensions are correct
    assert l2m == round(l2m)
    assert len(U) == m * m

    if m == 1:
        # If we are at the coarsest level
        # TODO: solve the only remaining equation directly!
        Unew = U.copy() # Placeholder
    else:
        # 1. TODO: pre-smooth the error
        #    perform <nsmooth> Jacobi iterations
        
        # 2. TODO: calculate the residual
        
        # 3. TODO: coarsen the residual
        mc = (m - 1) // 2
        Rcoarse = np.zeros(mc * mc) # Placeholder to avoid NameError
        
        # 4. recurse to Vcycle on a coarser grid
        Ecoarse = Vcycle(np.zeros(mc * mc), omega, nsmooth, mc, -Rcoarse)
        
        # 5. TODO: interpolate the error
        
        # 6. TODO: update the solution given the interpolated error
        
        # 7. TODO: post-smooth the error
        #    perform <nsmooth> Jacobi iterations
        Unew = U.copy() # Placeholder

    return Unew

def plotU(m, U):
    h = 1.0 / (m + 1)
    # Corrected grid generation: MATLAB had linspace(1/h, 1-1/h, m) 
    # which calculates to linspace(m+1, -m, m). You likely meant linspace(h, 1-h, m).
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)
    X, Y = np.meshgrid(x, y)
    
    # Clear the figure to update it in the loop
    plt.clf()
    ax = plt.axes(projection='3d')
    
    # Reshape matching MATLAB's column-major format ('F'), then transpose
    U_2d = U.reshape((m, m), order='F').T
    
    ax.plot_surface(X, Y, U_2d, cmap='viridis', edgecolor='none')
    ax.set_title('Computed solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('U')
    
    # Draw and pause to create an animation effect
    plt.draw()

# --- Main Script ---
if __name__ == "__main__":
    # Turn on interactive mode for plotting in a loop
    plt.ion()
    
    # Exact solution and RHS
    u = lambda x, y: np.exp(np.pi * x) * np.sin(np.pi * y) + 0.5 * (x * y)**2
    f = lambda x, y: x**2 + y**2
    
    m = 2**6 - 1
    U = np.zeros(m * m)
    F = form_rhs(m, f, u) # TODO: Form the right-hand side
    
    epsilon = 1.0e-10
    omega = 2.0 / 3.0 # Defined omega (standard for weighted Jacobi) since it was missing
    
    for i in range(1, 101):
        R = F + Amult(U, m)
        rel_resid = np.linalg.norm(R, 2) / np.linalg.norm(F, 2)
        
        print(f'*** Outer iteration: {i:3d}, rel. resid.: {rel_resid:e}')
        
        if rel_resid < epsilon:
            break
            
        U = Vcycle(U, omega, 3, m, F)
        plotU(m, U)
        plt.pause(0.5)
        
    # Keep the final plot open
    plt.ioff()
    plt.show()
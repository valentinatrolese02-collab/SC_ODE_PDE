import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')

def coarsen(R, m):
    # Calculate expected coarse grid dimension
    mc = int((m - 1) / 2)
    
    # Reshape the 1D flat array into a 2D fine grid matrix
    R_mat = R.reshape((m, m))
    
    # Extract the center points (starts at 1, step 2)
    center = R_mat[1::2, 1::2]
    
    # Extract the 4 direct neighbors
    top    = R_mat[0:-2:2, 1::2]
    bottom = R_mat[2::2,    1::2]
    left   = R_mat[1::2,   0:-2:2]
    right  = R_mat[1::2,   2::2]
    
    # Extract the 4 diagonal neighbors
    top_left     = R_mat[0:-2:2, 0:-2:2]
    top_right    = R_mat[0:-2:2, 2::2]
    bottom_left  = R_mat[2::2,    0:-2:2]
    bottom_right = R_mat[2::2,    2::2]
    
    # Apply the full weighting stencil 2D equation in a single vectorized operation
    Rc_mat = (
        4.0 * center + 
        2.0 * (top + bottom + left + right) + 
        1.0 * (top_left + top_right + bottom_left + bottom_right)
    ) / 16.0
    
    # Safety check: ensure the resulting matrix matches the expected coarse size
    assert Rc_mat.shape == (mc, mc), f"Shape mismatch: expected ({mc}, {mc}) but got {Rc_mat.shape}"
    
    # Flatten back to a 1D array
    Rc = Rc_mat.flatten()
    return Rc

def interpolate(Rc, m):
    mc = int((m - 1) / 2)
    Rc_mat = Rc.reshape((mc, mc))
    
    Rc_pad = np.pad(Rc_mat, pad_width=1, mode='constant', constant_values=0)
    R_mat = np.zeros((m, m))
    
    R_mat[1::2, 1::2] = Rc_mat
    #even rows
    R_mat[0::2, 1::2] = 0.5 * (Rc_pad[:-1, 1:-1] + Rc_pad[1:, 1:-1])
    #even columns
    R_mat[1::2, 0::2] = 0.5 * (Rc_pad[1:-1, :-1] + Rc_pad[1:-1, 1:])
    # even both
    R_mat[0::2, 0::2] = 0.25 * (Rc_pad[:-1, :-1] + Rc_pad[1:, :-1] + 
                                Rc_pad[:-1, 1:] + Rc_pad[1:, 1:])                      
    R=R_mat.flatten()

    return R

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

def smooth(U, F, m, omega):
    """
    Performs under-relaxed Jacobi smoothing for the 2D Poisson equation.
    
    Args:
        U (np.ndarray): Current approximation of the solution (1D array of size m*m).
        F (np.ndarray): Right-hand side vector (1D array of size m*m).
        m (int): Number of interior grid points in one dimension.
        omega (float, optional): Relaxation parameter. Defaults to 0.8 (optimal for 2D).
        num_iters (int, optional): Number of smoothing sweeps. Defaults to 2.
        
    Returns:
        np.ndarray: The smoothed approximation vector.
    """
    h = 1.0 / (m + 1)

    D_inv = (h**2) / 4.0  
    U_k = np.copy(U)
    AU = Amult(U_k, m)
    R = F + AU
    U_k = U_k + omega * D_inv * R
        
    return U_k

def Vcycle(U,omega,nsmooth,m,F):
    h = 1/(m+1)
    l2m = np.log2(m+1)
    assert l2m == round(l2m), "Grid size 'm' must be 2^k - 1"
    assert len(U) == m * m, f"Expected length {m*m}, got {len(U)}"

    if m==1:
        U = F / (4/h**2)
    else:
        for _ in range(nsmooth):
            U = smooth(U, F, m, omega)
        R = F+Amult(U,m)
        Rc = coarsen(R,m)
        mc = int((m - 1) / 2)
        Ec = Vcycle(np.zeros(mc*mc),omega,nsmooth,mc,Rc)
        E = interpolate(Ec,m)
        U = U+E
        for _ in range(nsmooth):
            U = smooth(U, F, m, omega)

    return U

def u(x, y):
    return np.exp(np.pi * x) * np.sin(np.pi * y) + 0.5 * (x * y)**2

def f(x, y):
    return x**2 + y**2

def form_rhs(m, f_func, u_func):
    """
    Forms the right-hand side vector for the 2D Poisson problem.
    Includes the source term f(x,y) and the Dirichlet boundary conditions from u(x,y).
    """
    h = 1.0 / (m + 1)
    
    # Create 1D coordinate arrays for the interior points
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the source function on the interior grid
    F_2d = f_func(X, Y)
    
    # Add Dirichlet boundary conditions scaled by 1/h^2
    # Left (x=0) and Right (x=1) boundaries
    F_2d[:, 0]  += u_func(0.0, y) / h**2
    F_2d[:, -1] += u_func(1.0, y) / h**2
    
    # Bottom (y=0) and Top (y=1) boundaries
    F_2d[0, :]  += u_func(x, 0.0) / h**2
    F_2d[-1, :] += u_func(x, 1.0) / h**2
    
    return F_2d.flatten()

def plotU(m, U):
    h = 1 / (m + 1)
    x = np.linspace(1/h, 1 - 1/h, m)
    y = np.linspace(1/h, 1 - 1/h, m)
    X, Y = np.meshgrid(x, y)
    Z = np.reshape(U, (m, m)).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Computed solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('U')

m = 2**6 - 1
U = np.zeros(m * m)
F = form_rhs(m, f, u)  # TODO: Form the right-hand side
epsilon = 1.0E-10
omega = 0.8

for i in range(1, 101):
    R = F + Amult(U, m)
    print(f'*** Outer iteration: {i:3d}, rel. resid.: {np.linalg.norm(R, 2) / np.linalg.norm(F, 2):e}')
    if np.linalg.norm(R, 2) / np.linalg.norm(F, 2) < epsilon:
        break
    U = Vcycle(U, omega, 3, m, F)
plotU(m, U)
# --- Multigrid Convergence Study ---
# Append this at the bottom of your file (replacing the previous __main__ block)

if __name__ == '__main__':
    # Define the range of k values to study
    k_values = [3, 4, 5, 6, 7]
    m_values = [2**k - 1 for k in k_values]
    
    # Keep parameters constant for a fair comparison
    omega = 0.8
    nsmooth = 3
    epsilon = 1.0e-6
    max_iters = 100
    
    # List to store the number of iterations for each grid size
    iterations_needed = []
    
    print("Starting Multigrid Convergence Study...")
    print(f"{'k':>2} | {'m':>4} | {'Grid points (m*m)':>17} | {'Outer Iters':>11}")
    print("-" * 43)
    
    for k, m in zip(k_values, m_values):
        # 1. Initialize the solution vector with zeros
        U = np.zeros(m * m)
        
        # 2. Form the right-hand side using the exact analytical function
        F = form_rhs(m, f, u)
        
        iters = 0
        
        # 3. Outer iteration loop
        for i in range(1, max_iters + 1):
            # Calculate residual R = F - A*U (Amult returns -A*U)
            R = F + Amult(U, m)
            
            # Calculate the relative residual norm
            rel_resid = np.linalg.norm(R, 2) / np.linalg.norm(F, 2)
            
            # Check if the constant tolerance is met
            if rel_resid < epsilon:
                break
                
            # Perform one V-cycle
            U = Vcycle(U, omega, nsmooth, m, F)
            iters += 1
            
        iterations_needed.append(iters)
        print(f"{k:2d} | {m:4d} | {m*m:17d} | {iters:11d}")
        
    # --- Generate the convergence dependence plot ---
    plt.figure(figsize=(9, 6))
    
    # Plot iterations vs grid size
    plt.plot(m_values, iterations_needed, color='#2ca02c', marker='D', 
             linestyle='-', linewidth=2.5, markersize=8, 
             label=fr'Tolerance $\epsilon = 10^{{-10}}$')
    
    # Plot formatting
    plt.title('Multigrid V-Cycle Convergence Study\nIndependence of Outer Iterations from Grid Size', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of grid points in 1D ($m$)', fontsize=12)
    plt.ylabel('Outer Iterations to Converge', fontsize=12)
    
    # Set y-axis to start at 0 to visually emphasize the flat line
    max_iter_val = max(iterations_needed)
    plt.ylim(0, max_iter_val + 5)
    
    # Set x-ticks to correspond exactly to our m values
    plt.xticks(m_values, [f"{m}\n(k={k})" for m, k in zip(m_values, k_values)])
    
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    
    plt.show()
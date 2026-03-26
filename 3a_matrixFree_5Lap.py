import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator

def Amult(U, m):
    """
    Matrix-free implementation of the 2D negative 5-points Laplacian (-Ah * U).
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
    return (AU_2d / h2).flatten()

def solve_poisson_cg(m,F, rtol=1e-8):
    """
    Solves the discretized Poisson equation using Conjugate Gradient.
    Equations: -Ah * U = -F (to ensure a symmetric positive definite system).
    """
    N = m**2
    
    # Define the right-hand side F (e.g., a simple source term F = 1)
    # The problem requires solving -Ah * U = -F
    minus_F = -F 
    
    # Wrap in a SciPy LinearOperator
    # Whenever the function calls A @ x --> it runs Amult(x, m)
    A_op = LinearOperator(shape=(N, N), matvec=lambda x: Amult(x, m))
    
    # Setup the callback to record the residual history
    # We store the L2-norm of the residual at each iteration
    residuals = []
    
    def cg_callback(xk):
        # Calculate the current true residual: r = b - A*x
        # Here, b is minus_F and A is our operator
        current_res = np.linalg.norm(minus_F - A_op.matvec(xk))
        residuals.append(current_res)
        
    # Run the Conjugate Gradient solver
    print(f"Starting CG solver for grid {m}x{m} ({N} unknowns)...")
    U_sol, exit_code = cg(A_op, minus_F, rtol=1e-8, callback=cg_callback)
    
    if exit_code == 0:
        print(f"CG converged successfully in {len(residuals)} iterations.")
    else:
        print("CG did not converge.")

    return U_sol, residuals


m = 100  # Grid size (interior points)
def g(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

x_int = np.linspace(0, 1, m+2)[1:-1]
y_int = np.linspace(0, 1, m+2)[1:-1]
X_int, Y_int = np.meshgrid(x_int, y_int)

F = g(X_int, Y_int).flatten()  # Flatten to a 1D array for CG solver
U_sol, res_history = solve_poisson_cg(m, F)

# Estimate the Convergence Rate 
# The average per-iteration convergence factor is (r_final / r_initial)^(1 / iterations)
if len(res_history) > 1:
    initial_res = res_history[0]
    final_res = res_history[-1]
    num_iters = len(res_history)
    
    # Calculate average reduction factor per iteration
    avg_rate = (final_res / initial_res)**(1.0 / num_iters)
    
    print(f"Initial Residual: {initial_res:.2e}")
    print(f"Final Residual:   {final_res:.2e}")
    print(f"Estimated average convergence factor (rho): {avg_rate:.4f}")

# Plot Convergence History 
plt.figure(figsize=(8, 5))
plt.semilogy(range(1, len(res_history) + 1), res_history, 'b.-', markersize=4)
plt.title(f"Conjugate Gradient Convergence History (m={m})")
plt.xlabel("Iteration number")
plt.ylabel("Residual L2-norm (log scale)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

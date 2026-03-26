import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

def newtonn(x0, tol, itmax, fun, jac_fun):
    """
    Newton's method for n-dimensional systems of nonlinear equations.
    
    Parameters:
    x0      : initial guess (1D array)
    tol     : tolerance so that ||x_{k+1} - x_k|| < tol
    itmax   : max number of iterations
    fun     : function that returns the residual vector F(x)
    jac_fun : function that returns the Jacobian matrix J(x)
    
    Returns:
    XK      : list containing all iterated vectors
    resd    : list of residual norms at each iteration ||F_k||
    it      : number of required iterations to satisfy tolerance
    """
    xk = np.array(x0, dtype=float)
    resd = [norm(fun(xk))]
    XK = [np.copy(xk)]
    it = 0
    tolk = 1.0
    
    while it < itmax and tolk > tol:
        Fk = fun(xk)
        DFk = jac_fun(xk)
        
        # Solve the linear system: J(x_k) * dx_k = -F(x_k)
        dxk = solve(DFk, -Fk)
        
        xk = xk + dxk
        XK.append(np.copy(xk))
        resd.append(norm(fun(xk)))
        
        tolk = norm(XK[-1] - XK[-2])
        it += 1
        
    return XK, resd, it

def G(U, epsilon, h, alpha, beta):
    """
    Function that calculates the residuals G(U). 
    alpha and beta are the boundary conditions.
    """
    N = len(U)
    F = np.zeros(N)
    
    # Append boundary conditions to the extremes to simplify indexing
    U_full = np.concatenate(([alpha], U, [beta]))
    
    for i in range(1, N + 1):
        term_diff = epsilon * (U_full[i-1] - 2*U_full[i] + U_full[i+1]) / h**2
        term_conv = U_full[i] * ((U_full[i+1] - U_full[i-1]) / (2*h) - 1)
        F[i-1] = term_diff + term_conv
        
    return F

def Jacobian(U, epsilon, h, alpha, beta):
    """
    Calculates the exact analytical Jacobian matrix based on Eq. 2.106.
    """
    N = len(U)
    J = np.zeros((N, N))
    U_full = np.concatenate(([alpha], U, [beta]))
    
    for i in range(N):
        u_i = U_full[i+1]
        u_im1 = U_full[i]
        u_ip1 = U_full[i+2]
        
        # Derivative with respect to U_i (Main diagonal)
        J[i, i] = -2*epsilon / h**2 + (u_ip1 - u_im1) / (2*h) - 1
        
        # Derivative with respect to U_{i-1} (Lower diagonal)
        if i > 0:
            J[i, i-1] = epsilon / h**2 - u_i / (2*h)
            
        # Derivative with respect to U_{i+1} (Upper diagonal)
        if i < N - 1:
            J[i, i+1] = epsilon / h**2 + u_i / (2*h)
            
    return J

def solve_bvp(N, epsilon=0.01, a=0.0, b=1.0, alpha=1.0, beta=-1.0, tol=1e-8, itmax=50):
    """
    Solves the boundary value problem for a given number of interior points N.
    
    Returns:
    x       : array of grid points (including boundaries)
    U_final : array of the solution at the grid points
    h       : grid spacing
    """
    # 1. Setup grid
    h = (b - a) / (N + 1)
    x = np.linspace(a, b, N + 2)
    x_int = x[1:-1] # Interior points
    
    # 2. Initial guess - Equation 2.105
    w0 = 1.0
    x_bar = 0.5 
    u0_guess = -w0 * np.tanh(w0 * (x_int - x_bar) / (2 * epsilon))
    
    # 3. Function wrappers
    fun_wrapper = lambda U: G(U, epsilon, h, alpha, beta)
    jac_wrapper = lambda U: Jacobian(U, epsilon, h, alpha, beta)
    
    # 4. Call Newton's method
    XK, residuals, iterations = newtonn(u0_guess, tol, itmax, fun_wrapper, jac_wrapper)
    
    # 5. Construct the final solution array including boundaries
    U_final = np.concatenate(([alpha], XK[-1], [beta]))
    
    print(f"Solved for N={N} in {iterations} iterations.")
    return x, U_final, h

# --- 1. Compute Reference Solution ---
N_ref = 3999  # N_ref + 1 = 4000 intervals
print("Computing reference solution...")
x_ref, U_ref, h_ref = solve_bvp(N=N_ref)

# --- 2. Define Test Grids ---
# N values chosen so that 4000 is perfectly divisible by N+1
N_list = [49, 99, 199, 399]
h_values = []
errors = []

print("\nComputing test solutions...")
for N in N_list:
    x, U, h = solve_bvp(N=N)
    
    # Calculate downsampling step to match reference grid points
    step = int((N_ref + 1) / (N + 1))
    U_ref_matched = U_ref[::step]
    
    # Calculate global error (Infinity norm)
    error = np.max(np.abs(U - U_ref_matched))
    
    h_values.append(h)
    errors.append(error)

# --- 3. Estimate Convergence Order (p) ---
# Fit a line to the log-log data: log(E) = p * log(h) + log(C)
# The slope of this line is the order of convergence 'p'
log_h = np.log(h_values)
log_E = np.log(errors)
p_slope, log_C = np.polyfit(log_h, log_E, 1)

print(f"\nEstimated global order of convergence (slope): p = {p_slope:.4f}")

# --- 4. Plot the Convergence Figure ---
plt.figure(figsize=(8, 6))

# Plot actual errors
plt.loglog(h_values, errors, marker='o', linestyle='-', linewidth=2, label=f'Numerical Error (slope $\\approx$ {p_slope:.2f})')

# Plot a reference line for O(h^2) to visually compare the slopes
# We anchor the reference line to the first error point
h_ref_line = np.array([h_values[0], h_values[-1]])
C_ref = errors[0] / (h_values[0]**2)
error_ref_line = C_ref * (h_ref_line**2)

plt.loglog(h_ref_line, error_ref_line, linestyle='--', color='gray', label='Reference $O(h^2)$')

# Formatting the plot
plt.xlabel('Grid spacing $h$ (log scale)')
plt.ylabel('Global Error $E(h)$ (log scale)')
plt.title('Convergence of the Global Error')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()
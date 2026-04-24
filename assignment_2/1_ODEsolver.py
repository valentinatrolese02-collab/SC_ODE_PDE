# RK solver that can solve systems of first order ODE’s in the form of the general initial value problem
# The stepsize control must be based on keeping an estimate of the local error smaller than tol
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# Choice of methods are in general based on concerns about
# •Accuracy and eﬃciency requirements
# •Robustness (Stability)
# •Memory requirements


# general RungeKutta solver - no order is defined
def RKsolver(f, t0, y0, h, c, A, b, d, reps, aeps):
    """
    f: function f(t, y). The function accepts also multidim y
    t0, y0: initial condition
    h: initial step-size
    c, A, b, d: Butcher Tableau components for method of order 3

    reps: relative tolerance, a ratio that will give a tol wrt to the abs value of y.
        for example reps = 10-4 --> 0.01% of abs(y)

    aeps: abs tolerance

    y_next = y_n + h * sum(b_i * k_i)
    """
    stages = len(b)
    y0 = np.atleast_1d(y0)  # Handles both scalar and vector systems

    while True:

        # storage for the slopes (k_i) at each stage
        k = np.zeros((stages, len(y0)))
        
        for i in range(stages):
            ti = t0 + c[i] * h
            # we only sum up to i-1 because it's an explicit method
            yi = y0 + h * (A[i, :i] @ k[:i, :])
            
            # Calculate the slope at this stage
            k[i, :] = f(ti, yi)
        
        # estimate LTE, and update step size to produce an error of exactly tolerance
        y_next = y0 + h * (b @ k) # y_next is an array !! we keep it likr this in the solver to handle also 2d cases
        tol = reps * np.linalg.norm(y_next) + aeps

        LTE = np.linalg.norm(h * (d@k))
        # h_new = h*(tol/LTE)**(1/3) <-- in theory, we add safety factprs
        h_new = h * 0.9 * (tol / (LTE + 1e-16))**(1/3) 
        
        if LTE <= tol:
            return y_next, h_new
        else:
            # Update h and restart the 'while' loop for this same t0
            h = h_new


# method of order 2 and 3. It implements a (simple) step-size control that attempts to satisfy E < tol
# adaptive 2-3 RKsolver; it handles 1dim or multidim input y
def RK_23(f, t0, y0, h, tf, reps, aeps):
    # Define tableau for Runge Kutta 2-3
    c = np.array([0, 1/2, 1])
    A = np.array([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]])
    b = np.array([1/6, 2/3, 1/6])
    d = np.array([1/12, -1/6, 1 /12])

    # Storage for plotting later
    t_list = [t0]
    y_list = [np.atleast_1d(y0)]
    h_list = [] 

    t = t0
    y = y0
    while t < tf:
        h_attempt = min(h, tf - t)
        
        # Take the step using the general solver
        y_next, h_correct = RKsolver(f, t, y, h_attempt, c, A, b, d, reps, aeps)
        
        # Update the state for the next iteration
        h = h_correct
        t += h
        y = y_next
        
        # Store results
        t_list.append(t)
        y_list.append(y)
        h_list.append(h)
    
    return np.array(t_list), np.array(y_list), np.array(h_list)


######################################
# TEST on model for flame propagation
delta = 0.02
# flame propagation function
def f(t, y): return y**2 - y**3

# initial condition (and final condition on t)
t0 = 0
tf = 2 / delta
y0 = delta  # y0
h0 = 1e-4  # Initial guess

times, ys, hs = RK_23(f, t0, y0, h0, tf, reps=1e-4, aeps=1e-4)

# print(y_list)
# 0.02, array([0.02000004]), array([0.02036146]), array([0.02603735]), array([0.03177517]), array([0.03835837]), array([0.0458342]), array([0.05427486]), array([0.06375781]), array([0.07436741]), array([0.08619645]), array([0.09934804]), array([0.11393811]), array([0.13009873]), array([0.1479826]), array([0.16776949]), array([0.18967544]), array([0.21396693]), array([0.24423908]), array([0.27830121]), array([0.31638271]), array([0.35924291]), array([0.40811796]), array([0.46521783]), array([0.53559044]), array([0.62505862]), array([0.69990383]), array([0.77288291]), array([0.842226]), array([0.88268961]), array([0.9136221]), array([0.93779357]), array([0.955325]), array([0.96843891]), array([0.9782658]), array([0.98554303]), array([0.9908119]), array([0.99450032]), array([0.99696299]), array([0.99850215]), array([0.99937766]), array([0.99981007]), array([0.99997814]), array([1.00000952]), array([0.99992988]), array([1.00012741]), array([0.99989599]), array([1.00008452]), array([0.99990812]), array([1.00013058]), array([0.99989709]), array([0.99999474])]

y_list = np.array(ys)[:, 0]
print(y_list)


# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(times, y_list, 'b-o', markersize=3, label='Flame Position $y(t)$')
ax1.set_ylabel('$y$')
ax1.set_title('Adaptive RK23: Flame Propagation')
ax1.grid(True)
ax1.legend()

# 2. Fix: plot h_list here, not y_list again!
ax2.semilogy(times[1:], hs, 'r-x', markersize=3, label='Step Size $h$')
ax2.set_ylabel('Step Size $h$')
ax2.set_xlabel('Time $t$')
ax2.grid(True)
ax2.legend()

#plt.savefig(f'assignment_2/plots/delta{delta}.svg', format='svg')
plt.tight_layout()
plt.show()

# COMMENT: delta = 1e-4 becomes very problematic if we keep the tolerance as 1e-4; we have to change it to 10-6 to produce the graph

##############
# comparison with premade ode solvers
# --- Run SciPy's version for comparison ---
# We use your same tolerance logic here
sol = solve_ivp(f, [t0, tf], [y0], method='RK23', rtol=1e-4, atol=1e-4)

# --- Comparison Plot ---
plt.figure(figsize=(10, 6))

# Your custom results
plt.plot(times, y_list, 'b-', label='Custom RK_23', linewidth=2)

# SciPy results
plt.plot(sol.t, sol.y[0], 'r--', label='SciPy solve_ivp', linewidth=2)

plt.xlabel('Time $t$')
plt.ylabel('Flame Position $y$')
plt.title(f'Comparison: Custom vs library ($\delta$={delta})')
plt.legend()
plt.grid(True)
plt.show()

################
# f is differenctiable in R --> is differentiable in any closed interval --> is Lipschitz continuous
# f is independent of t 
# we conclude that we can apply Picard Lindeløf
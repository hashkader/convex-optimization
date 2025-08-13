# Welcome to My Convex Optimization
***

## Task
This project implements core convex optimization algorithms from scratch. The main challenge is building efficient numerical methods for finding function minima and solving linear programming problems without relying heavily on external optimization libraries. Key problems tackled include root finding for non-linear equations, gradient-based optimization, and constrained linear programming using the simplex method.

## Description
I've implemented five core optimization functions:

*Bisection Method*: Uses binary search to find function roots by iteratively narrowing down intervals where sign changes occur
*Newton-Raphson*: Faster root finding using derivative information and tangent line approximations
*Gradient Descent*: Iterative optimization that follows the negative gradient to find local minima
*Linear Programming*: Solves constrained optimization problems using scipy's simplex implementation
*Function Plotting*: Visualizes functions to understand their behavior and verify optimization results

The algorithms handle edge cases like functions without roots and use appropriate convergence criteria. For the linear programming component, I manually solved the constraint system when scipy had issues with the specific test case.

## Installation
Make sure you have the required dependencies:

```bash
    pip install numpy matplotlib scipy
```
No additional build steps required - just standard Python packages.

## Usage
Run the main script to test all optimization methods:
```bash
    python convex_optimization.py
```
Or import individual functions:
```python
from convex_optimization import gradient_descent, find_root_bisection

# Find minimum of f(x) = (x-1)^4 + x^2
f = lambda x: (x - 1)**4 + x**2
f_prime = lambda x: 4*((x-1)**3) + 2*x

minimum = gradient_descent(f, f_prime, start=-1, learning_rate=0.01)
print(f"Minimum found at x = {minimum}")
```

### The Core Team
This was a solo project done by me, Hashim Kader

<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px' /></span>

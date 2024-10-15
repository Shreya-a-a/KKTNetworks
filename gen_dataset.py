import cvxpy as cp
import numpy as np

def generate_dataset(num_samples, m, n):
    # Prepare lists to store each parameter separately
    c_data = []
    A_data = []
    b_data = []
    x_opt = []
    lamb_opt = []

    for i in range(num_samples):

        c1 = np.random.uniform(-3, 3, n)
        A1 = np.random.uniform(-3, 3, (m, n))
        b1 = np.random.uniform(-3, 3, m)

        max_value = max(np.max(np.abs(c1)), np.max(np.abs(A1)), np.max(np.abs(b1)))

        # Normalize A, b, and c by dividing by the max value
        c = c1 / max_value
        A = A1 / max_value
        b = b1 / max_value

        x = cp.Variable(n)

        # Define the LP problem
        objective = cp.Minimize(c.T @ x)
        constraints = [A @ x <= b]

        # Create and solve the problem
        problem = cp.Problem(objective, constraints)

        try:
            # Solve the problem
            problem.solve()

            # Check if the problem is solvable and optimal
            if problem.status == cp.OPTIMAL:
                # Extract the optimal solution and dual values
                optimal_solution = x.value
                dual_solution = [constraint.dual_value for constraint in problem.constraints]

                # Append each parameter to the respective list
                c_data.append(c)
                A_data.append(A)
                b_data.append(b)
                x_opt.append(optimal_solution)
                lamb_opt.append(dual_solution)

                # print(f"Generated and solved LP problem {i+1}.")

        except cp.SolverError:
            print(f"Problem {i+1} is not solvable, discarding.")

    return np.array(c_data), np.array(A_data), np.array(b_data), np.array(x_opt), np.array(lamb_opt)


num_samples = 7000 
m = 2
n = 2
c_data, A_data, b_data, x_data, lamb_data = generate_dataset(num_samples, m, n)

# Save all arrays to a single .npz file
np.savez('/content/drive/My Drive/btp/dataset.npz', c=c_data, A=A_data, b=b_data, x=x_data, lamb=lamb_data)

print("\nLP solutions have been saved to 'dataset.npz'.")

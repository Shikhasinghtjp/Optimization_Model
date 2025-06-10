# product_mix_optimization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, value, LpStatus, lpSum

# Define the LP problem
model = LpProblem(name="product-mix", sense=LpMaximize)

# Decision variables
x = LpVariable(name="Product_A", lowBound=0)
y = LpVariable(name="Product_B", lowBound=0)

# Objective Function: Maximize Profit
model += 20 * x + 30 * y, "Total_Profit"

# Constraints
model += (2 * x + 3 * y <= 100, "Labor_Constraint")
model += (4 * x + 2 * y <= 120, "Material_Constraint")

# Solve the model
model.solve()

# Results
print(f"Status: {LpStatus[model.status]}")
print(f"Optimal Product A: {x.value()}")
print(f"Optimal Product B: {y.value()}")
print(f"Maximum Profit: ${value(model.objective)}")

# Save results to Excel
results_df = pd.DataFrame({
    "Variable": ["Product_A", "Product_B", "Total_Profit"],
    "Value": [x.value(), y.value(), value(model.objective)]
})
results_df.to_excel("product_mix_optimization_results.xlsx", index=False)

# Plot feasible region
x_vals = np.linspace(0, 60, 400)
y1 = (100 - 2 * x_vals) / 3  # Labor Constraint
y2 = (120 - 4 * x_vals) / 2  # Material Constraint

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y1, label="2x + 3y ≤ 100 (Labor)", color='blue')
plt.plot(x_vals, y2, label="4x + 2y ≤ 120 (Material)", color='green')
plt.fill_between(x_vals, 0, np.minimum(y1, y2), where=(y1 > 0) & (y2 > 0), color='skyblue', alpha=0.4)

# Optimal point
plt.plot(x.value(), y.value(), 'ro', label=f"Optimal: ({x.value():.1f}, {y.value():.1f})")

plt.xlabel("Product A")
plt.ylabel("Product B")
plt.title("Feasible Region with Optimal Point")
plt.legend()
plt.grid(True)
plt.savefig("feasible_region_plot.png")
plt.show()

# Sensitivity approximation (basic)
print("\nConstraint Slack:")
for name, constraint in model.constraints.items():
    print(f"{name}: Slack = {constraint.slack}")

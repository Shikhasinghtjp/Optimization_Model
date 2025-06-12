# 🧠 Optimization Model using Linear Programming

This repository demonstrates how to solve a real-world business problem using **Linear Programming (LP)** and **Python’s PuLP library**. 
The goal is to formulate and optimize a decision-making model to maximize profit (or minimize cost) under given constraints.

---

## 🚀 Project Highlights

- ✅ Realistic Business Problem
- 🛠️ Formulated as a Linear Programming model
- 📊 Visualizations for feasible region and constraints
- 📁 Excel-based input/output for accessibility
- 📈 Sensitivity Analysis for decision insights

---

## 🧮 Problem Statement

The project models a **product mix optimization** problem where a company needs to decide the quantity of different products to produce in order to **maximize profit**, subject to:
- Resource availability (e.g., labor, materials)
- Production time constraints
- Market demand limits

---

## 📂 Project Structure

Optimization_Model/

├── optimization_model.py # Main LP model using PuLP
├── data.xlsx # Input data for resources, profit, constraints
├── solution_output.xlsx # Model results saved in Excel
├── feasible_region_plot.png # Visualization of constraints and feasible region
└── README.md # Project overview


---

## 📌 Key Features

- 📦 **PuLP-based LP Solver**
- 📊 **Matplotlib** plot of the feasible region
- 📁 **Pandas** integration with Excel files
- 🧠 **Interpretation of optimal solution and constraints**
- 🔍 **Sensitivity Analysis** on variable coefficients

---

## 🔧 How to Run

1. Clone the repository:
   
   git clone https://github.com/Shikhasinghtjp/Optimization_Model.git
   cd Optimization_Model
   
2. Install dependencies:

   pip install pulp pandas matplotlib openpyxl

3. Run the model:

   python optimization_model.py
   
4. Check the output:

   Results: solution_output.xlsx

   Plot: feasible_region_plot.png

   ---

📌 Dependencies

      pulp
      pandas
      matplotlib
      openpyxl

  ---

📈 Sample Output


---

🧠 Insights & Learnings

  This project demonstrates how optimization can improve business decision-making by:

  Modeling constraints mathematically

  Finding the most efficient use of resources

  Analyzing the impact of changes in constraints or objectives



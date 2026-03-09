# PoA Upper Bound

Computes exact Price of Anarchy upper bounds for homogeneous games using Gurobi.

## Capabilities

* **Supported Rules**: Both Proportional Sharing (PS) and Shapley Value (SV) support arbitrary polynomial cost degrees d >= 3.
* **Two step grid search**: Locates the optimal region via a global coarse scan, then refines precision with a local fine scan.
* **Fault tolerance**: Saves progress continuously to CSV files. Catches manual interruptions to preserve computed data.

## Requirements

Requires `numpy`, `pandas`, and `gurobipy` with an active Gurobi license.

## Execution

The main computation script is provided as a Jupyter Notebook. 

1. Open `poa_upper_bound_gurobi.ipynb`.
2. **Quick Test**: The first code cell is a testing environment. You can use it to compute the PoA bound for any single combination of n, p, d, and rule.
3. **Full Experiment**: The second code cell runs the full parameter sweep. It applies the two step search and continuously outputs the results to CSV files in the current directory. 
4. The `time_limit_per` parameter controls the maximum Gurobi solver time (in seconds) for each fixed structural topology.

## Plotting

All data visualization and plotting codes are contained in `plot.ipynb`. Run the cells in that notebook to read the generated CSV data and reproduce the paper figures.
# misc
Container for short examples, ideas, comparisons

---

### Usage

After cloning the repo, with Python >= 3.8 do

```
python -m venv env
source env/activate/bin
pip install -r requirements.txt
```

Install the solver SHOT separately by following the [instructions](https://shotsolver.dev/shot/using-shot/getting-started).

Run the example:
```
python pyomo_unstable_ocp.py <solver>
<solver>: shot, scip, gurobi
```

**Note** to use SCIP or Gurobi for solving the unstable OCP example, AMPLpy is necessary. To install it do:
```
python -m pip install amplpy --upgrade
# Install solver modules -- SCIP and Gurobi
python -m amplpy.modules install scip gurobi
```




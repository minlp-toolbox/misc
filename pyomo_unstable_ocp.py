from pyomo.environ import *
from pyomo.opt import SolverFactory, ProblemFormat
from time import perf_counter

def tic():
    """Tic."""
    global perf_ti
    perf_ti = perf_counter()


def toc(reset=False):
    """Toc."""
    global perf_ti
    if perf_ti is None:
        tic()
    tim = perf_counter()
    dt = tim - perf_ti
    print(f"Elapsed time: {dt} s")
    if reset:
        perf_ti = tim
    return dt

def create_ocp_unstable_system_pyomo(p_val=[0.9, 0.7]):
    dt = 0.05
    N = 30
    min_uptime = 2  # NOTE: hard-coded!
    BigM = 1e2

    model = ConcreteModel()
    model.T = RangeSet(0, N)
    model.Tu = RangeSet(0, N-1)

    X0 = p_val[0]
    Xref = p_val[1]

    model.x = Var(model.T, bounds=(-BigM, BigM))
    model.u = Var(model.Tu, domain=Binary)

    model.x[0].fix(X0)

    def dynamics_rule(m, k):
        #  Explicit RK4 integrator
        X = m.x[k]
        fun = lambda x, u: x**3 - u
        k1 = fun(m.x[k], m.u[k])
        k2 = fun(m.x[k] + dt / 2 * k1, m.u[k])
        k3 = fun(m.x[k] + dt / 2 * k2, m.u[k])
        k4 = fun(m.x[k] + dt * k3, m.u[k])
        X += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return m.x[k+1] == X
        # return m.x[k+1] == m.x[k] + dt * (m.x[k]**3 - m.u[k])  # explicit Euler integrator
    model.dynamics = Constraint(model.Tu, rule=dynamics_rule)

    def objective_rule(m):
        return sum((m.x[k] - Xref)**2 for k in m.T)
    model.obj = Objective(rule=objective_rule, sense=minimize)

    def uptime_rule(m, k):
        if k==0:
            return Constraint.Skip
        else:
            idx_1 = k - 1
            idx_2 = k - 2
            if idx_1 >=0:
                b_1 = m.u[idx_1]
            else:
                b_1 = 0
            if idx_2 >= 0:
                b_2 = m.u[idx_2]
            else:
                b_2 = 0
            print(f"{-m.u[k]} + {b_1} - {b_2} <= 0")
            return -m.u[k] + b_1 - b_2 <= 0
    model.uptime_constraints = Constraint(model.Tu, rule=uptime_rule)

    return model


# --- MAIN ---
if __name__ == "__main__":

    print("\n ========================================")
    print(" =========== HARD CODED EXAMPLE !! ==========")
    print("========================================")


    model = create_ocp_unstable_system_pyomo()
    # Create the NL file
    model_filename = "ocp_model.nl"
    model.write(model_filename, format=ProblemFormat.nl)
    print(f"NL file written to {model_filename}")

    # Call SHOT solver via Pyomo
    opt = SolverFactory('shot')  # SHOT uses SCIP internally
    tic()
    results = opt.solve(model, tee=True)
    toc()

    # Create the MindtPy solver
    # mip_solver = 'gurobi'  # or cbc, glpk, cplex...
    # nlp_solver = 'ipopt'   # or conopt, knitro...
    # opt = SolverFactory('mindtpy')
    # tic()
    # # Solve using Outer Approximation
    # results = opt.solve(
    #     model,
    #     strategy='GOA',  # could also use 'GBD', 'ECP'
    #     mip_solver=mip_solver,
    #     nlp_solver=nlp_solver,
    #     tee=True        # prints solver logs
    # )
    # toc()

    # Print results
    print("\n--- Optimal Solution ---")
    print(f"Objective = {value(model.obj)}")
    for k in model.T:
        print(f"x[{k}] = {value(model.x[k]):.4f}")
    for k in model.Tu:
        print(f"u[{k}] = {value(model.u[k])}")

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    state = []
    for k, v in model.x.get_values().items():
        state += [v]
    state = np.array(state)
    control_b = []
    for k, v in model.u.get_values().items():
        control_b += [v]
    control_b += [control_b[-1]]
    control_b = np.array(control_b)

    matplotlib.rcParams.update({"lines.linewidth": 1})
    fig, axs = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
    time_array = np.arange(0, 31*0.05, 0.05)

    axs[0].plot(time_array, state, color="tab:blue",)
    axs[0].axhline(0.7, color="red", linestyle=":")
    axs[0].set_ylabel("$x$")
    axs[0].set_ylim(0.65, 0.95)
    axs[1].step(
        time_array,
        control_b,
        color="tab:orange",
        marker=".",
        where="post",
    )
    axs[1].set_ylabel("$u$")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim(0, 1.5)

    plt.tight_layout()
    plt.show()

    print("\n ========================================")
    print(" =========== HARD CODED EXAMPLE !! ==========")
    print("========================================")
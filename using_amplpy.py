from amplpy import AMPL
from sys import argv

solver = argv[1]
path = "/home/syscop/ghezzi/minlplib_mod/clay0304hfsg.mod"

verbosity = 3


ampl = AMPL()
ampl.eval(f"model {path};")

ampl.eval(f"option solver {solver};")
ampl.eval(f"option solver_msg {verbosity};")
ampl.eval("option show_stats 1;")

if solver == "gurobi":  # TODO gurobi gets slower if setting intfeastol even if the default its 1e-5
    ampl.eval(f'option {solver}_options "outlev 1" "feastol=1e-8" "mipgap=1e-2" "threads=1" "timelimit=300";')
elif solver == "scip":
    ampl.eval(f'option {solver}_options "outlev 1" "round_reptol=1e-3" "feastol=1e-8" "mipgap=1e-2" "maxnthreads=1" "timelimit=300";')
elif solver == "xpress":
    ampl.eval(f'option {solver}_options "outlev 1" "intfeastol=1e-3" "feastol=1e-8" "mipgap=1e-2" "threads=1" "timelimit=300";')


ampl.eval("solve;")

ampl.eval("display solve_result_num, solve_result;")

obj = ampl.getObjective('obj')
print(f"\nObjective value: {obj.value()}")
breakpoint()

# print("\nVariables and values:")
# for var in ampl.getVariables():
#     # `var.value()` returns a scalar for a scalar variable or a Python
#     # dictionary for indexed variables.
#     print(f"  {var[0]} = {var[1].value()}")


# 5c️  (Optional) write the solution back to an NL/sol file for external use
ampl.writeSolution('mysolution.sol')   # <-- AMPL's native solution format
ampl.write("solution.txt")              # <-- plain‑text dump

ampl.close()

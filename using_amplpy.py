from amplpy import AMPL
from sys import argv
import os
from time import time
from camino.utils.data import write_json, read_json



def do_write(overview_target, start, i, algorithm, total_stats):
    time_now = time() - start
    total_time = time_now / (i + 1) * total_to_compute
    write_json(
        {
            "time": time_now,
            "total": total_to_compute,
            "done": (i + 1),
            "progress": (i + 1) / total_to_compute,
            "time_remaining_est": total_time - time_now,
            "time_total_est": total_time,
            "algorithm": algorithm,
            "data": total_stats,
        },
        overview_target,
    )

# Convex problems
cvx_problems = [
    "batch.mod", "batch0812.mod", "batchdes.mod", "clay0203hfsg.mod", "clay0204hfsg.mod", "clay0205hfsg.mod", "clay0303hfsg.mod", "clay0304hfsg.mod", "clay0305hfsg.mod", "cvxnonsep_normcon20.mod", "cvxnonsep_normcon30.mod", "cvxnonsep_normcon40.mod", "cvxnonsep_nsig20.mod", "cvxnonsep_nsig20r.mod", "cvxnonsep_nsig30.mod", "cvxnonsep_nsig30r.mod", "cvxnonsep_nsig40.mod", "cvxnonsep_nsig40r.mod", "cvxnonsep_pcon20.mod", "cvxnonsep_pcon20r.mod", "enpro48pb.mod", "enpro56pb.mod", "ex1223.mod", "ex1223b.mod", "fac1.mod", "fac2.mod", "flay02h.mod", "flay02m.mod", "flay03h.mod", "flay03m.mod", "flay04h.mod", "flay04m.mod", "flay05h.mod", "flay05m.mod", "flay06h.mod", "flay06m.mod", "fo7.mod", "fo7_2.mod", "fo7_ar25_1.mod", "fo7_ar2_1.mod", "fo7_ar3_1.mod", "fo7_ar4_1.mod", "fo7_ar5_1.mod", "fo8.mod", "fo8_ar25_1.mod", "fo8_ar2_1.mod", "fo8_ar3_1.mod", "fo8_ar4_1.mod", "fo8_ar5_1.mod", "fo9.mod", "fo9_ar25_1.mod", "fo9_ar2_1.mod", "fo9_ar3_1.mod", "fo9_ar4_1.mod", "fo9_ar5_1.mod", "gear2.mod", "gear3.mod", "jit1.mod", "m3.mod", "m6.mod", "m7.mod", "m7_ar25_1.mod", "m7_ar2_1.mod", "m7_ar3_1.mod", "m7_ar4_1.mod", "m7_ar5_1.mod", "no7_ar25_1.mod", "no7_ar2_1.mod", "no7_ar3_1.mod", "no7_ar4_1.mod", "no7_ar5_1.mod", "nvs20.mod", "o7.mod", "o7_2.mod", "o7_ar25_1.mod", "o7_ar2_1.mod", "o7_ar3_1.mod", "o7_ar4_1.mod", "o7_ar5_1.mod", "o8_ar4_1.mod", "o9_ar4_1.mod", "p_ball_10b_5p_2d_h.mod", "p_ball_10b_5p_3d_h.mod", "p_ball_10b_5p_4d_h.mod", "p_ball_10b_7p_3d_h.mod", "p_ball_15b_5p_2d_h.mod", "p_ball_20b_5p_2d_h.mod", "p_ball_20b_5p_3d_h.mod", "p_ball_30b_5p_2d_h.mod", "portfol_buyin.mod", "portfol_card.mod", "portfol_roundlot.mod", "prob10.mod", "ravempb.mod", "sssd08-04.mod", "sssd12-05.mod", "sssd15-04.mod", "sssd15-06.mod", "sssd15-08.mod", "sssd16-07.mod", "sssd18-06.mod", "sssd18-08.mod", "sssd20-04.mod", "sssd20-08.mod", "sssd22-08.mod", "sssd25-04.mod", "sssd25-08.mod", "st_e14.mod", "stockcycle.mod", "synthes1.mod", "synthes2.mod", "synthes3.mod", "tls2.mod", "tls4.mod", "tls5.mod", "tls6.mod",
]

# Nonconvex problems -- removed because breaking "gastrans.mod",
noncvx_problems = [
    "4stufen.mod", "autocorr_bern20-05.mod", "autocorr_bern20-10.mod", "autocorr_bern20-15.mod", "autocorr_bern25-06.mod", "autocorr_bern25-13.mod", "autocorr_bern25-19.mod", "autocorr_bern25-25.mod", "autocorr_bern30-04.mod", "autocorr_bern30-08.mod", "autocorr_bern30-15.mod", "autocorr_bern30-23.mod", "autocorr_bern30-30.mod", "autocorr_bern35-04.mod", "autocorr_bern35-09.mod", "autocorr_bern35-18.mod", "autocorr_bern35-26.mod", "autocorr_bern35-35fix.mod", "autocorr_bern40-05.mod", "autocorr_bern40-10.mod", "autocorr_bern40-20.mod", "autocorr_bern40-30.mod", "autocorr_bern40-40.mod", "autocorr_bern45-05.mod", "autocorr_bern45-11.mod", "autocorr_bern45-23.mod", "autocorr_bern45-34.mod", "autocorr_bern45-45.mod", "autocorr_bern50-06.mod", "autocorr_bern50-13.mod", "autocorr_bern50-25.mod", "autocorr_bern55-06.mod", "autocorr_bern55-14.mod", "autocorr_bern60-08.mod", "autocorr_bern60-15.mod", "batch0812_nc.mod", "batch_nc.mod", "beuster.mod", "casctanks.mod", "contvar.mod", "csched1.mod", "csched1a.mod", "csched2.mod", "csched2a.mod", "eg_int_s.mod", "eniplac.mod", "ex1221.mod", "ex1222.mod", "ex1224.mod", "ex1225.mod", "ex1226.mod", "ex1233.mod", "ex1243.mod", "ex1244.mod", "ex1252.mod", "ex1252a.mod", "ex3pb.mod", "feedtray.mod", "gasnet.mod", "gastrans.mod", "gastrans040.mod", "gear2.mod", "gear3.mod", "gear4.mod", "ghg_1veh.mod", "ghg_2veh.mod", "ghg_3veh.mod", "gkocis.mod", "heatexch_gen1.mod", "heatexch_gen2.mod", "heatexch_gen3.mod", "heatexch_spec1.mod", "heatexch_spec2.mod", "heatexch_spec3.mod", "hybriddynamic_var.mod", "johnall.mod", "kport20.mod", "kport40.mod", "nvs01.mod", "nvs05.mod", "nvs08.mod", "nvs20.mod", "nvs21.mod", "nvs22.mod", "oaer.mod", "ortez.mod", "parallel.mod", "pooling_epa1.mod", "pooling_epa2.mod", "prob10.mod", "procsel.mod", "sfacloc1_2_90.mod", "sfacloc1_2_95.mod", "sfacloc1_3_90.mod", "sfacloc1_3_95.mod", "sfacloc1_4_90.mod", "sfacloc1_4_95.mod", "sfacloc2_2_90.mod", "sfacloc2_2_95.mod", "sfacloc2_3_90.mod", "sfacloc2_3_95.mod", "sfacloc2_4_90.mod", "sfacloc2_4_95.mod", "spring.mod", "st_e15.mod", "st_e29.mod", "st_e32.mod", "st_e35.mod", "st_e36.mod", "st_e38.mod", "st_e40.mod", "supplychainp1_020306.mod", "supplychainr1_020306.mod", "supplychainr1_030510.mod", "synheat.mod", "tanksize.mod", "transswitch0009p.mod", "transswitch0009r.mod", "transswitch0014p.mod", "transswitch0014r.mod", "transswitch0030p.mod", "transswitch0030r.mod", "transswitch0039p.mod", "transswitch0039r.mod", "tspn05.mod", "tspn08.mod", "tspn10.mod", "tspn12.mod", "tspn15.mod", "wager.mod", "wastepaper3.mod", "wastepaper4.mod", "wastepaper5.mod", "wastepaper6.mod", "water4.mod", "waternd1.mod", "waternd2.mod", "waterno2_01.mod", "waterno2_02.mod", "waterno2_03.mod", "watertreatnd_conc.mod", "watertreatnd_flow.mod", "waterx.mod", "waterz.mod", "windfac.mod",
]

if len(argv) != 5:
    print("Usage: python using_amplpy.py <problem type 'cvx', 'noncvx'> <solver 'gurobi', 'scip', 'xpress'> <root_folder_minlp> <results_folder>")
    exit()

verbosity = 3
problem_type = argv[1]
solver = argv[2]
root_folder_minlp = argv[3]
results_folder = argv[4]
results_folder = os.path.join(results_folder, problem_type + "_" + solver)

if problem_type == 'cvx':
    problems = cvx_problems
elif problem_type == 'noncvx':
    problems = noncvx_problems
else:
    raise ValueError("problem type must be either 'cvx' or 'noncvx'!")

overview_target = os.path.join(results_folder, "overview.json")
start = time()
total_to_compute = len(problems)
if os.path.exists(overview_target):
    data = read_json(overview_target)
    total_stats = data["data"]
    algorithm = data["algorithm"]
    i_start = data["done"]
    start -= data["time"]
else:
    os.makedirs(results_folder, exist_ok=True)
    total_stats = [
        [
            "id",
            "path",
            "obj",
            "dual_obj",
            "calc_time",
        ]
    ]
    i_start = 0

idx = i_start
for problem in problems[i_start:]:
    problem_path = os.path.join(root_folder_minlp, problem)
    ampl = AMPL()
    ampl.eval(f"model {problem_path};")

    ampl.eval(f"option solver {solver};")
    ampl.eval(f"option solver_msg {verbosity};")
    ampl.eval("option show_stats 1;")


    if solver == "gurobi":  # TODO gurobi gets slower if setting intfeastol even if the default its 1e-5
        ampl.eval(f'option {solver}_options "bestbound=1" "feastol=1e-8" "mipgap=1e-2" "threads=1" "timelimit=300";')  # "outlev 1"
    elif solver == "scip":
        ampl.eval(f'option {solver}_options "bestbound=1" "round_reptol=1e-3" "feastol=1e-8" "mipgap=1e-2" "maxnthreads=1" "timelimit=300";')  # "outlev 1"
    elif solver == "xpress":
        ampl.eval(f'option {solver}_options "bestbound=1" "intfeastol=1e-3" "feastol=1e-8" "mipgap=1e-2" "threads=1" "timelimit=300";')  # "outlev 1"

    ampl.eval("solve;")
    ampl.eval("display solve_result_num, solve_result;")

    # TODO check for optimal solution else return failed
    if ampl.getValue('solve_result') not in ["solved", "limit"]:
        total_stats.append([idx, problem_path, "FAILED", ampl.getValue('solve_result'), ampl.getValue('_solve_elapsed_time')])
    else:
        result = {}
        result["id"] = idx
        result["path"] = problem_path
        result["obj"] = ampl.getValue('obj')
        result["dual_obj"] = ampl.getValue('obj.bestbound')
        result["calc_time"] = ampl.getValue('_solve_elapsed_time')
        total_stats.append([idx, problem_path, result['obj'], result['dual_obj'], result['calc_time']])
    do_write(overview_target, start, idx, solver, total_stats)
    idx += 1




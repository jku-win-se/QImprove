import itertools
import sys

import click
import pathlib
import numpy as np

import logging

import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from src.utils import seeded_rng

from io import StringIO

log_stream = StringIO()
logging.basicConfig(
     stream=log_stream,  # log to variable
     level=logging.INFO,
     format= '[%(asctime)s] %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

handler = logging.StreamHandler(sys.stdout)
logging.getLogger("QRefactoring").addHandler(handler)  # log to sysout too

logger = logging.getLogger("QRefactoring")
logger.setLevel(logging.DEBUG)

from src import main, utils

objectives = ["overlap", "num_gates", "depth", "num_nonloc_gates", "num_parameters"]

"""
# only use circuits with overlap larger than this for calculation of Performance Indicators
These are calculated using this formula:
num_gates:  # number of gates of "ideal" solution (or the fixed version, if we have)
overlap = (1.0 - 0.007635)**(3*num_gates/2) * (1.0 - 0.0002023)**(3*num_gates/2) #device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error
"""
eval_data = {
 'AA2': [2, 1],
 'AA3': [3, 1],
 'AA4': [4, 1],
 'AA5': [5, 1],
 'GHZ2': [2, 2],
 'GHZ3': [3, 2],
 'GHZ4': [4, 2],
 'GHZ5': [5, 2],
 'GS3': [3, 2],
 'GS4': [4, 2],
 'GS5': [5, 2],
 'QFT2': [2, 1],
 'QFT3': [3, 1],
 'QFT4': [4, 1],
 'QFT5': [5, 1],
 'wstate2': [2, 2],
 'wstate3': [3, 2],
 'wstate4': [4, 2],
 'wstate5': [5, 2],
 'QSE2_2': [2, 1],
 'QSE2_3': [3, 1],
 'QSE2_4': [4, 1],
 'QSE2_5': [5, 1],
 'QG_8': [2, 2],
 'QSE_15': [4, 2],
 'QSE_3': [5, 2],
 'QSO_5': [3, 2],
 'QSO_6': [2, 2],
 'adder_n4': [4, 1],
 'bell_n4': [4, 2],
 'cat_state_n4': [4, 2],
 'fredkin_n3': [3, 1],
 'hs4_n4': [4, 1],
 'iswap_n2': [2, 1],
 'linearsolver_n3': [3, 2],
 'lpn_n5': [5, 2],
 'qec_en_n5': [5, 1],
 'qrng_n4': [4, 2],
 'quantum_walk': [2, 1],
 'teleportation_n3': [3, 1],
 'tofolli_n3': [3, 1],
 'wstate_n3': [3, 2],
 'hamiltonian_simulation_2': [2, 1],
 'hamiltonian_simulation_3': [3, 1],
 'hamiltonian_simulation_4': [4, 1],
 'hamiltonian_simulation_5': [5, 1],
 'quantum_mc_F': [3, 1]}

# [num_gates,depth,non_local]
reference_fitness_values = {
 'AA2': [11, 7, 1],
 'AA3': [29, 16, 6],
 'AA4': [33, 21, 13],
 'AA5': [53, 37, 29],
 'GHZ2': [2, 2, 1],
 'GHZ3': [3, 3, 2],
 'GHZ4': [4, 4, 3],
 'GHZ5': [5, 5, 4],
 'GS3': [6, 4, 3],
 'GS4': [8, 4, 4],
 'GS5': [10, 5, 5],
 'QFT2': [4, 4, 2],
 'QFT3': [7, 6, 4],
 'QFT4': [12, 8, 8],
 'QFT5': [17, 10, 12],
 'wstate2': [5, 4, 2],
 'wstate3': [9, 6, 4],
 'wstate4': [13, 8, 6],
 'wstate5': [17, 10, 8],
 'QSE2_2': [3, 3, 1],
 'QSE2_3': [6, 5, 3],
 'QSE2_4': [10, 7, 6],
 'QSE2_5': [15, 9, 10],
 'QG_8': [2, 2, 2],
 'QSE_15': [24, 17, 11],
 'QSE_3': [45, 31, 18],
 'QSO_5': [17, 12, 6],
 'QSO_6': [3, 2, 1],
 'adder_n4': [23, 11, 10],
 'bell_n4': [33, 13, 7],
 'cat_state_n4': [4, 4, 3],
 'fredkin_n3': [19, 11, 8],
 'hs4_n4': [28, 9, 4],
 'iswap_n2': [9, 7, 2],
 'linearsolver_n3': [19, 11, 4],
 'lpn_n5': [11, 4, 2],
 'qec_en_n5': [25, 17, 10],
 'qrng_n4': [4, 1, 0],
 'quantum_walk': [11, 7, 3],
 'teleportation_n3': [8, 6, 2],
 'tofolli_n3': [18, 12, 6],
 'wstate_n3': [30, 22, 9],
 'hamiltonian_simulation_2': [9, 6, 2],
 'hamiltonian_simulation_3': [15, 9, 4],
 'hamiltonian_simulation_4': [21, 12, 6],
 'hamiltonian_simulation_5': [27, 15, 8],
 'quantum_mc_F': [41, 32, 16]}

# [num_qubits, flag] where flag = 1 iff arbitrary, 2 = specific
experiment_props = {
'AA2': [2, 1],
 'AA3': [3, 1],
 'AA4': [4, 1],
 'AA5': [5, 1],
 'GHZ2': [2, 2],
 'GHZ3': [3, 2],
 'GHZ4': [4, 2],
 'GHZ5': [5, 2],
 'GS3': [3, 2],
 'GS4': [4, 2],
 'GS5': [5, 2],
 'QFT2': [2, 1],
 'QFT3': [3, 1],
 'QFT4': [4, 1],
 'QFT5': [5, 1],
 'wstate2': [2, 2],
 'wstate3': [3, 2],
 'wstate4': [4, 2],
 'wstate5': [5, 2],
 'QSE2_2': [2, 1],
 'QSE2_3': [3, 1],
 'QSE2_4': [4, 1],
 'QSE2_5': [5, 1],
 'QG_8': [2, 2],
 'QSE_15': [4, 2],
 'QSE_3': [5, 2],
 'QSO_5': [3, 2],
 'QSO_6': [2, 2],
 'adder_n4': [4, 1],
 'bell_n4': [4, 2],
 'cat_state_n4': [4, 2],
 'fredkin_n3': [3, 1],
 'hs4_n4': [4, 1],
 'iswap_n2': [2, 1],
 'linearsolver_n3': [3, 2],
 'lpn_n5': [5, 2],
 'qec_en_n5': [5, 1],
 'qrng_n4': [4, 2],
 'quantum_walk': [2, 1],
 'teleportation_n3': [3, 1],
 'tofolli_n3': [3, 1],
 'wstate_n3': [3, 2],
 'hamiltonian_simulation_2': [2, 1],
 'hamiltonian_simulation_3': [3, 1],
 'hamiltonian_simulation_4': [4, 1],
 'hamiltonian_simulation_5': [5, 1],
 'quantum_mc_F': [3, 1]}


def circ_gates(name):
    num_qubits = eval_data[name][0]
    alt = eval_data[name][1]
    if alt == 1: # arbitrary states
        gates_bell = 2*2*num_qubits  # h, cz per qubit; normal and inverse
        num_gates = gates_bell + 2*reference_fitness_values[name][0]
    else:
        num_gates = 2*reference_fitness_values[name][0]
    return num_gates

def _get_min_overlap_operator_only(problem_prefix: str):
    num_gates = reference_fitness_values[problem_prefix][0]
    err_rate_multi_qubit_gates = (1.0 - 0.007635) ** (3 * num_gates / 2)
    err_rate_single_qubit_gates = (1.0 - 0.0002023) ** (3 * num_gates / 2)
    return err_rate_multi_qubit_gates * err_rate_single_qubit_gates  # device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error

def _get_min_overlap_overall(problem_prefix: str):
    num_gates = circ_gates(problem_prefix)
    err_rate_multi_qubit_gates = (1.0 - 0.007635) ** (3 * num_gates / 2)
    err_rate_single_qubit_gates = (1.0 - 0.0002023) ** (3 * num_gates / 2)
    return err_rate_multi_qubit_gates * err_rate_single_qubit_gates  # device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error

ignore_suffix = [
    "",
    "_globalPareto.csv",
    "_HVrefpoint.csv",
    "_earliest_finish.csv",
    ".csv",
    "-PI.csv",
    "-normPI.csv",
    "-DCI.csv"
]

@click.command()
@click.argument("out_file", # help="Provide the path to the circuit file.",
                nargs=1, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
def run_analysis(out_file: pathlib.Path):
    results_dir = out_file.parent
    problem_prefix = out_file.stem.split("_seed")[0]

    print(f"Problem Identifier: {problem_prefix}")
    print("Analysing", out_file)

    # compute DCI for each generation
    # for file in all_problem_csv_files:
    calculate_IGD(out_file, problem_prefix, results_dir)
    calculate_norm_IGD(out_file, problem_prefix, results_dir)
    calculate_DCI(out_file, problem_prefix, results_dir)


def calculate_DCI(file: pathlib.Path, problem_prefix: str, results_dir: pathlib.Path):
    print("Calculating DCI for", file)
    dci_output_filename = file.name.replace(".csv", "-DCI.csv")

    if (results_dir / dci_output_filename).exists():
        print(results_dir / dci_output_filename, "exists already. Skipping DCI computation")
        return

    df = pd.read_csv(file)
    refpoint = pd.read_csv(results_dir / f"{problem_prefix}_HVrefpoint.csv")

    div = 10
    low = [0, 1, 1, 0, 0]
    # up = [1.01, 30, 30, 30, 50]
    up = [1.01] + refpoint.iloc[0].values[1:].tolist()

    d = [(up[i] - low[i]) / div for i in range(5)]  # hyperbox sizes

    gens_and_DCIs = []
    for ngen in list(df.ngen.unique())[:]:
        print(f"{ngen=}")
        timestamp = df[(df.ngen == ngen)].timestamp.max()
        gen_data = {"ngen": ngen, "timestamp": timestamp}
        for label, min_overlap in [("_all", 0.0),
                                   ("_min", _get_min_overlap_operator_only(problem_prefix)),
                                   ("_min_overall", _get_min_overlap_overall(problem_prefix))]:
            filtered = df[(df.overlap >= min_overlap) & (df.ngen == ngen)][objectives]  #.drop_duplicates()
            print(f"{label=}", len(filtered), "entries in generational Pareto front.")
            if len(filtered) == 0:
                gen_data[label] = None
            else:
                # gen_data[label] = calc_DCI_for_gen(filtered.to_numpy(), div, low, up, d)
                # print(gen_data[label])
                gen_data[label] = calc_DCI_for_gen(filtered.to_numpy(), div, low, up, d)
                print(gen_data[label])
        gens_and_DCIs.append(gen_data)

    df_DCI_full = pd.DataFrame(gens_and_DCIs)
    print("Writing DCI computationresults to", file.parent / dci_output_filename)
    df_DCI_full.to_csv(file.parent / dci_output_filename, index=False)


def calc_DCI_for_gen(pareto_np, div, low, up, d):
    # assign fitness values to grid coordinates
    f_gen_grid = []
    f_gen_grid = (pareto_np - low) / d

    grid = np.array(list(itertools.product(range(div), repeat=5)))
    dists = np.apply_along_axis(lambda coords: np.linalg.norm(f_gen_grid - coords, axis=1), 1, grid)
    dists = dists.min(axis=1)

    yes = dists < np.sqrt(5 + 1)
    dists[yes] = 1 - (dists[yes] ** 2) / (5 + 1)
    no = dists >= np.sqrt(5 + 1)
    dists[no] = 0

    return sum(dists) * 1. / (div ** 5)


def calculate_IGD(file: pathlib.Path, problem_prefix: str, results_dir: pathlib.Path):
    print("Calculating IGDs for", file)
    pi_output_filename = file.name.replace(".csv", "-PI.csv")

    if (results_dir / pi_output_filename).exists():
        print(results_dir / pi_output_filename, "exists already. Skipping PI computation")
        return

    df = pd.read_csv(file)

    gpf = pd.read_csv(results_dir / f"{problem_prefix}_globalPareto.csv")
    global_PF = gpf.to_numpy()
    global_PF[:,0] = 1 - global_PF[:,0]

    refpoint = pd.read_csv(results_dir / f"{problem_prefix}_HVrefpoint.csv")
    HV_refpoint = [1.0] + refpoint.iloc[0].values[1:].tolist()

    pis = dict(
        gd=GD(global_PF),
        gdplus=GDPlus(global_PF),
        igd=IGD(global_PF),
        igdplus=IGDPlus(global_PF),
        HV=HV(ref_point=HV_refpoint),
    )

    gens_and_DCIs = []
    for ngen in list(df.ngen.unique())[:]:
        print(f"{ngen=}")
        timestamp = df[(df.ngen == ngen)].timestamp.max()
        gen_data = {"ngen": ngen, "timestamp": timestamp}
        for overlap_label, min_overlap in [("_all", 0.0),
                                           ("_min", _get_min_overlap_operator_only(problem_prefix)),
                                           ("_min_overall", _get_min_overlap_overall(problem_prefix))]:
            filtered = df[(df.overlap >= min_overlap) & (df.ngen == ngen)][objectives]  #.drop_duplicates()
            print(f"{overlap_label=}", len(filtered), "entries in generational Pareto front.")
            for pi_name, pi in pis.items():
                label = f"{pi_name}{overlap_label}"
                if len(filtered) == 0:
                    gen_data[label] = None
                else:
                    minimize_gen_data = filtered.to_numpy()
                    minimize_gen_data[:, 0] = 1 - minimize_gen_data[:, 0]

                    gen_data[label] = pi(minimize_gen_data)
                    print(label, gen_data[label])
        gens_and_DCIs.append(gen_data)

    df_DCI_full = pd.DataFrame(gens_and_DCIs)
    print("Writing PI computation results to", file.parent / pi_output_filename)
    df_DCI_full.to_csv(file.parent / pi_output_filename, index=False)


def calculate_norm_IGD(file: pathlib.Path, problem_prefix: str, results_dir: pathlib.Path):
    print("Calculating normalized IGDs for", file)
    pi_output_filename = file.name.replace(".csv", "-normPI.csv")

    if (results_dir / pi_output_filename).exists():
        print(results_dir / pi_output_filename, "exists already. Skipping normalized PI computation")
        return

    df = pd.read_csv(file)

    refpoint = pd.read_csv(results_dir / f"{problem_prefix}_HVrefpoint.csv")
    HV_refpoint = [1.0] + refpoint.iloc[0].values[1:].tolist()

    gpf = pd.read_csv(results_dir / f"{problem_prefix}_globalPareto.csv")
    global_PF = gpf.to_numpy()
    global_PF[:,0] = 1 - global_PF[:,0]

    # normalize the PF
    norm_max = HV_refpoint
    norm_min = np.zeros(5)
    norm_span = norm_max - norm_min
    norm_PF = (global_PF - norm_min) / norm_span



    pis = dict(
        gd=GD(norm_PF),
        gdplus=GDPlus(norm_PF),
        igd=IGD(norm_PF),
        igdplus=IGDPlus(norm_PF),
        HV=HV(ref_point=np.ones(5)),
    )

    gens_and_DCIs = []
    for ngen in list(df.ngen.unique())[:]:
        print(f"{ngen=}")
        timestamp = df[(df.ngen == ngen)].timestamp.max()
        gen_data = {"ngen": ngen, "timestamp": timestamp}
        for overlap_label, min_overlap in [("_all", 0.0),
                                           ("_min", _get_min_overlap_operator_only(problem_prefix)),
                                           ("_min_overall", _get_min_overlap_overall(problem_prefix))]:
            filtered = df[(df.overlap >= min_overlap) & (df.ngen == ngen)][objectives]  # .drop_duplicates()
            print(f"{overlap_label=}", len(filtered), "entries in generational Pareto front.")
            for pi_name, pi in pis.items():
                label = f"{pi_name}{overlap_label}"
                if len(filtered) == 0:
                    gen_data[label] = None
                else:
                    minimize_gen_data = filtered.to_numpy()
                    minimize_gen_data[:, 0] = 1 - minimize_gen_data[:, 0]

                    # now normalize it using the HV refpoint as max and [0,0,0,0,0] as min
                    normalized = (minimize_gen_data - norm_min) / norm_span

                    gen_data[label] = pi(normalized)
                    print(label, gen_data[label])
        gens_and_DCIs.append(gen_data)

    df_DCI_full = pd.DataFrame(gens_and_DCIs)
    print("Writing PI computation results to", file.parent / pi_output_filename)
    df_DCI_full.to_csv(file.parent / pi_output_filename, index=False)


if __name__ == "__main__":
    run_analysis()

from pathlib import Path
import pandas as pd


HYBRID = ""
NONHYBRID = "use_numerical_optimizer_False"
FIXED = "gateset_fixed"

better_options = {'': "Hybrid", 
                  'Ngen_50': 'Hybrid_{NGen=50}', 
                  'Ngen_100': 'Hybrid_{NGen=100}', 
                  'N_100': 'Hybrid_{N=100}', 
                  'N_200': 'Hybrid_{N=200}', 
                  'gateset_fixed': "Fixed", 
                  'init_pop_20': 'Hybrid_{Init=20}',
                  'opt_within_Q2': 'Hybrid_{Q2}', 
                  'use_numerical_optimizer_False': "NonHybrid"}

# [num_qubits, flag] where flag = True iff arbitrary, False = specific
QUBITS_and_ARBITRARY = {
 'AA2': [2, True], 'AA3': [3, True], 'AA4': [4, True], 'AA5': [5, True], 
 'GHZ2': [2, False], 'GHZ3': [3, False], 'GHZ4': [4, False], 'GHZ5': [5, False], 
 'GS3': [3, False], 'GS4': [4, False], 'GS5': [5, False], 
 'QFT2': [2, True], 'QFT3': [3, True], 'QFT4': [4, True], 'QFT5': [5, True], 
 'wstate2': [2, False], 'wstate3': [3, False], 'wstate4': [4, False], 'wstate5': [5, False], 
 'QSE2_2': [2, True], 'QSE2_3': [3, True], 'QSE2_4': [4, True], 'QSE2_5': [5, True], 
 'QG_8': [2, False], 'QSE_15': [4, False], 'QSE_3': [5, False], 'QSO_5': [3, False], 'QSO_6': [2, False], 
 'adder_n4': [4, True], 'bell_n4': [4, False], 'cat_state_n4': [4, False], 'fredkin_n3': [3, True], 
 'hs4_n4': [4, True], 'iswap_n2': [2, True], 'linearsolver_n3': [3, False], 'lpn_n5': [5, False], 
 'qec_en_n5': [5, True], 'qrng_n4': [4, False], 'quantum_walk': [2, True], 'teleportation_n3': [3, True], 
 'tofolli_n3': [3, True], 'wstate_n3': [3, False], 
 'hamiltonian_simulation_2': [2, True], 'hamiltonian_simulation_3': [3, True], 'hamiltonian_simulation_4': [4, True], 'hamiltonian_simulation_5': [5, True], 
 'quantum_mc_F': [3, True]} 

repair_circuits = [f"QSE2_{qubits}" for qubits in range(2,6)] + ["QSE_15", "QSE_3", "QSO_5", "QSO_6", "QG_8"]
all_names = [k for k, v in QUBITS_and_ARBITRARY.items()]
perfect = [name for name in all_names if name not in repair_circuits]


def circ_gates(name):
    num_qubits, alt = QUBITS_and_ARBITRARY[name]
    if alt: # arbitrary states
        gates_bell = 2*2*num_qubits  # h, cz per qubit; normal and inverse
        num_gates = gates_bell + 2*reference_fitness_values[name][0]
    else:
        num_gates = 2*reference_fitness_values[name][0]
    return num_gates

def get_min_overlap_operator_only(problem_prefix: str):
    num_gates = reference_fitness_values[problem_prefix][0]
    err_rate_multi_qubit_gates = (1.0 - 0.007635) ** (3 * num_gates / 2)
    err_rate_single_qubit_gates = (1.0 - 0.0002023) ** (3 * num_gates / 2)
    return err_rate_multi_qubit_gates * err_rate_single_qubit_gates  # device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error

def get_min_overlap_overall(problem_prefix: str):
    num_gates = circ_gates(problem_prefix)
    err_rate_multi_qubit_gates = (1.0 - 0.007635) ** (3 * num_gates / 2)
    err_rate_single_qubit_gates = (1.0 - 0.0002023) ** (3 * num_gates / 2)
    return err_rate_multi_qubit_gates * err_rate_single_qubit_gates  # device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error

# Used in Results-analysis
def get_approx_error_rate(num_gates: int):
    err_rate_multi_qubit_gates = (1 - 0.007635) ** (3 * num_gates / 2)
    err_rate_single_qubit_gates = (1 - 0.0002023) ** (3 * num_gates / 2)
    return 1 - err_rate_multi_qubit_gates * err_rate_single_qubit_gates  # device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error

def get_actual_error_rate(num_gates: int, num_nonloc_gates: int):
    err_rate_multi_qubit_gates = (1 - 0.007635) ** (3 * (num_nonloc_gates))
    err_rate_single_qubit_gates = (1 - 0.0002023) ** (3 * (num_gates - num_nonloc_gates))
    return 1 - err_rate_multi_qubit_gates * err_rate_single_qubit_gates  # device with highest QV (IBMQ Mumbai), take median CNOT and Pauli-X error

# Used in PI analysis
def extract_info_from_file(filename):
    # print(filename)
    no_indicator = str(filename.stem).split("-")[0]
    
    problem = no_indicator.split("_seed")[0]
    seed = int(no_indicator.split("_seed")[1].split("_")[0])
    optstring = no_indicator.split("_seed")[1].split("_", 1)[1]
    
    option = ""
    possible_options = ["N_100", "N_200", "init_pop_20", "gateset_fixed", "opt_within_Q2", "use_numerical_optimizer_False"]
    for poss in possible_options:
        if poss in optstring:
            option = poss
    return problem, seed, option


def extract_values(files, earliest_finish=None):
    # earliest_finish = "25%"
    earliest_finish_times = {}
    if earliest_finish and earliest_finish != "last":
        for earliest_finish_file in list(results_path.glob("*earliest_finish.csv")):
            problem = earliest_finish_file.stem.replace("_earliest_finish", "")

            earliest_finish_df = pd.read_csv(earliest_finish_file)
            earliest_finish_time = earliest_finish_df.iloc[0][earliest_finish]
            earliest_finish_times[problem] = earliest_finish_time
    # print("Earliest finish times:")
    # pprint(earliest_finish_times)
    pi_rows = []
    # extract last gen value
    for pi_file in sorted(files):
        problem, seed, option = extract_info_from_file(pi_file)
        qubits, arbitrary = QUBITS_and_ARBITRARY[problem]
        
        last_row = dict(problem=problem, option=option, seed=seed, qubits=qubits, arbitrary=arbitrary)
        pi_file_df = pd.read_csv(pi_file)
        if earliest_finish and earliest_finish != "last":
            if problem not in earliest_finish_times:
                print("WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("No earliest finish time for problem", problem)
            else:
                pi_file_df = pi_file_df[pi_file_df.timestamp <= earliest_finish_df[earliest_finish].iloc[0]]

        last_row.update(pi_file_df.iloc[-1].to_dict())
        pi_rows.append(last_row)

    df = pd.DataFrame(pi_rows)
    return df.sort_values(by=["problem", "option", "seed"])





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


QISKIT_opt_results = {
 'AA2': [11, 7, 1],
 'AA3': [25, 14, 6],
 'AA4': [29, 17, 13],
 'AA5': [49, 33, 29],
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
 'adder_n4': [23, 11, 10],
 'bell_n4': [33, 13, 7],
 'cat_state_n4': [4, 4, 3],
 'fredkin_n3': [19, 11, 8],
 'hs4_n4': [16, 7, 4],
 'iswap_n2': [9, 7, 2],
 'linearsolver_n3': [15, 11, 4],
 'lpn_n5': [7, 4, 2],
 'qec_en_n5': [23, 15, 10],
 'qrng_n4': [4, 1, 0],
 'quantum_walk': [11, 7, 3],
 'teleportation_n3': [8, 6, 2],
 'tofolli_n3': [18, 12, 6],
 'wstate_n3': [29, 22, 9],
 'hamiltonian_simulation_2': [9, 6, 2],
 'hamiltonian_simulation_3': [15, 9, 4],
 'hamiltonian_simulation_4': [21, 12, 6],
 'hamiltonian_simulation_5': [27, 15, 8],
 'quantum_mc_F': [27, 15, 8]}

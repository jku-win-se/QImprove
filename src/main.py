# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:23:20 2022

@author: fege9
"""
import json
import pathlib
import logging
import pandas as pd
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


from . import helper_func as fun
# import helper_func as fun
import sys

logger = logging.getLogger("QRefactoring")

def settings_and_init(circuit_file: pathlib.Path,
                      settings_file: pathlib.Path,
                      buggy_file: pathlib.Path,
                      settings_override: dict = None):

    # read settings from file, then override with CLI params
    with open(settings_file, "r") as settingsf:
        settings = json.load(settingsf)
    settings.update(settings_override)

    # add the default settings, normalize, etc
    circ = _get_circ_from_file(circuit_file, settings)
    settings = adjust_settings(circ, settings)  # extend settings

    # use buggy operator as initial value (seeding...)
    if buggy_file is not None:
        logger.info(f"Importing buggy Circuit from {buggy_file}")
        buggy = QuantumCircuit.from_qasm_file(str(buggy_file))
        settings["init_operator"] = buggy
    else:
        settings["init_operator"] = circ[3]

    logger.info("Settings for this run are:")
    logger.info(settings)

    return settings


def run_search(settings) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    toolbox = fun.deap_init(settings)
    logger.info("Initialization done! Doing GP now...")

    start_time = time.time()
    pop, pareto, log, time_stamps, pareto_fitness_vals, evals, HVs = fun.nsga3(toolbox, settings)
    end_gp = time.time()

    if settings.get("reduce_pareto"):
        pareto = fun.reduce_pareto(pareto, settings)

    res = fun.get_pareto_final_qc(settings, pareto)
    end = time.time()

    # STORE RESULTS

    logger.info(f"The found oracle is/are: {res} ")
    logger.info(f"GP, Selection, Overall duration: {end_gp - start_time}, {end - end_gp}, {end - start_time}")

    for ind in pareto:
        logger.info(f"{ind} {ind.fitness.values}")

    if settings["sel_scheme"] == "Manual":
        solutions = [ind[0] for ind in res]
    else:
        solutions = [res[0]]

    # write log to CSV
    df_log = pd.DataFrame(log)

    gen_dfs = []
    for idx, (n_eval, timestamp, pareto, HV) in enumerate(zip(evals, time_stamps, pareto_fitness_vals, HVs)):
        gen_df = pd.DataFrame(pareto, columns=["overlap", "num_gates", "depth", "num_nonloc_gates", "num_parameters"])
        gen_df["ngen"] = idx + 1
        gen_df["neval"] = n_eval
        gen_df["timestamp"] = timestamp
        gen_df["HV"] = HV
        gen_dfs.append(gen_df)

    df = pd.concat(gen_dfs)
    return df, df_log, solutions


def main(circuit_file: pathlib.Path,
         settings_file: pathlib.Path,
         buggy_file: pathlib.Path,
         results_folder: pathlib.Path,
         run_id: str,
         settings_override: dict = None,
         log_stream=None):
    logger.debug(f"{circuit_file=}, {settings_file=}, {results_folder=}, {buggy_file=}, {run_id=}, {settings_override=}")

    # Define where the logs go
    results_folder.mkdir(exist_ok=True)

    set_override_string = "default-settings"

    if settings_override:
        set_override_strings = []
        for key, value in settings_override.items():
            if isinstance(value, list):
                set_override_strings.append(f"{key}_{''.join(value)}")
            else:
                set_override_strings.append(f"{key}_{value}")
        set_override_string = "_".join(set_override_strings)

    out_file_prefix = f"{circuit_file.stem}_{run_id}_{set_override_string}"
    search_log_datafile = results_folder / f"{out_file_prefix}.csv"
    solution_filename = results_folder / f"{out_file_prefix}.qasm"  # the file of the selected circuit (solution)
    log_csv_name = results_folder / f"{out_file_prefix}_logbook.csv"

    settings = settings_and_init(circuit_file, settings_file, buggy_file, settings_override)

    if log_csv_name.exists() and search_log_datafile.exists():  # if this already ran, then stop
        logger.warning("This job already ran. Exiting.")
        return

    if buggy_file is not None and buggy_file.exists() and settings_override.get("init_pop") == 20:
        logger.warning("This job already ran with pop_seed=20 as default. Exiting.")
        return

    df, df_log, solutions = run_search(settings)

    # Store all the results
    for count, ind in enumerate(solutions):
        with open(str(solution_filename).replace(".qasm", f"_sol_{count}.qasm"), "w") as CIRC_file:
            CIRC_file.write(ind.qasm())
            # print(ind[0].draw())

    # store the DEAP Logbook
    df_log.to_csv(log_csv_name, index=False)

    # Log our own search tracing
    df.to_csv(search_log_datafile, index=False)

    # write all logs to logfile
    with open(results_folder / f"{out_file_prefix}.out", "w") as logfile:
        all_text = log_stream.getvalue()

        lines = all_text.splitlines()
        keep = []
        for line in lines:
            if "] INFO - Pass:" in line:
                continue

            if "] INFO - Total Transpile Time" in line:
                # print(line)
                continue

            if line.strip() == "":
                continue

            # print(line)
            keep.append(line)

        logfile.write("\n".join(keep))

"""
Read and ajust circuit
"""

def _get_circ_from_file(circuit_file: pathlib.Path, settings: dict):
    logger.info(f"Importing QCircuit from {circuit_file}")
    sys.path.append(str(circuit_file.parent.absolute()))
    # init_circ = __import__(circ_file.stem)
    if settings.get("init_state") is None:
        circ = adjust_circuit(circuit_file)  # arbitrary initial states
    else:
        circ = adjust_circuit_with_initial_state(circuit_file, settings)  # specific initial state
    return circ


def adjust_circuit_with_initial_state(circ_file, settings):
    qc = QuantumCircuit.from_qasm_file(circ_file)
    gate = qc.to_gate()
    size = gate.num_qubits

    overall_QC = QuantumCircuit(size)
    oracle_index = []
    target_qubits_oracles = [range(size)]
    if settings["init_state"] == "zero":
        settings["init_state"] = np.zeros(2 ** size)
        settings["init_state"][0] = 1.0
    elif settings["init_state"] == "equal":
        settings["init_state"] = np.ones(2 ** size)
    init = np.array(settings["init_state"]) * (1. / np.linalg.norm(np.array(settings["init_state"])))
    overall_QC.initialize(init, overall_QC.qubits)
    dummy = Gate(name="oracle", num_qubits=len(target_qubits_oracles[0]), params=[])
    overall_QC.append(dummy, range(size))
    oracle_index.append(len(overall_QC) - 1)
    overall_QC.append(gate.inverse(), range(size))
    return overall_QC, target_qubits_oracles, oracle_index, qc


def adjust_circuit(circ_file):
    def bellstates(size, inverse=False):
        qc = QuantumCircuit(2 * size)
        for i in range(size):
            qc.h(i)
            qc.cx(i, i + size)
        gate = qc.to_gate()
        if inverse is True:
            gate = gate.inverse()
        return gate

    # parameter für Suche: zu optimierendes Gate
    qc = QuantumCircuit.from_qasm_file(circ_file)
    gate = qc.to_gate()
    size = gate.num_qubits

    overall_QC = QuantumCircuit(2 * gate.num_qubits)
    overall_QC.append(bellstates(size), range(2 * size))

    oracle_index = []
    target_qubits_oracles = [range(size)]
    dummy = Gate(name="oracle", num_qubits=len(target_qubits_oracles[0]), params=[])
    # dummy = gate
    overall_QC.append(dummy, range(size))
    # overall_QC.barrier()
    oracle_index.append(len(overall_QC) - 1)
    overall_QC.append(gate.inverse(), range(size))
    overall_QC.append(bellstates(size, inverse=True), range(2 * size))
    return overall_QC, target_qubits_oracles, oracle_index, qc


def adjust_settings(circ, customized_settings):
    # automatically generated

    # update default settings
    settings = {
        "N": 20,
        "use_numerical_optimizer": True,
        "overlap_const": False,
        "NGEN": 10,
        "CXPB": 1.0,
        "MUTPB": 1.0,
        "gateset": "variable",
        "numerical_optimizer": "COBYLA",
        "Sorted_order": [-1, 2, 3, 4, 5],
        "prob": None,
        "weights_gates": None,
        "weights_mut": None,
        "weights_cx": None,
        "opt_within": None,
        "opt_final": None,
        "opt_select": None,
        "sel_scheme": None,
        "opt_x_gen": None,
        "reduce_pareto": False
    }
    settings.update(customized_settings)

    if settings["gateset"] == "variable":
        from . import Gates_var as Gates
    elif settings["gateset"] == "fixed":
        from . import Gates_fix as Gates
    else:
        logger.info(f"Couldn't identify gateset '{settings['gateset']}', select 'variable' or 'fixed'")
    settings["elementary"] = Gates.gate_set

    if settings.get("weights_gates") is not None:
        assert isinstance(settings["weights_gates"], dict), "settings['weights_gates'] is not a dict"
        for gatename, weight in settings["weights_gates"].items():
            settings["elementary"][gatename.upper()].random_selection_weight = \
                weight / len(settings["weights_gates"].items())

    settings["min_max_selector"] = (1.0, -1.0, -1.0, -1.0, -1.0)  # define if we minimize or maximize in the search
    settings["Circuit"] = circ  # remember circuit in settings
    if settings.get("init_state") is None or settings.get("init_state") == "zero":  # if we don't have an init_state, then just use zeros as target
        settings["target"] = np.zeros(2 ** (circ[0].num_qubits))
        settings["target"][0] = 1.0
    else:
        settings["target"] = np.array(settings["init_state"]) * (1. / np.linalg.norm(np.array(settings["init_state"])))

    # normalize weights
    for weights_identifier in ["weights_mut", "weights_cx"]:
        if settings.get(weights_identifier, None) is not None:
            np_arr = np.array(settings[weights_identifier])
            settings[weights_identifier] = np_arr / np_arr.sum()

    return settings


# uncomment/comment for using "main" as function rather than right here for developing
# uncomment for use in console
"""if __name__ == "__main__":
    # circuit = sys.argv[1]
    # settings = sys.argv[2]

    circuit = f"{p.parents[1]}/examples/Grover/QC.py"
    settings = f"{p.parents[1]}/examples/Grover/settings.json"
    assert pathlib.Path(settings).exists()
    main(circuit, settings, seed)  # uncomment
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:28:56 2022

@author: fege9
"""
from deap import creator, base, tools, algorithms
from deap.benchmarks import tools as benchtools
import numpy as np
import scipy.special as scisp
import logging
import multiprocessing

# import pyzx as zx
import time
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGatesDecomposition
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
# from qiskit.circuit import ParameterVector, Parameter
# from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag

from src.utils import seeded_rng

logger = logging.getLogger("QRefactoring")

def deap_init(settings):
    # create classes and toolbox initialization
    # circ = settings["Circuit"]
    creator.create("FitnessMulti", base.Fitness, weights=settings["min_max_selector"])
    creator.create("Individual", list, fitness=creator.FitnessMulti, max_gates=None, operand=None, operator=None)
    toolbox = base.Toolbox()
    # pool = multiprocessing.Pool(5)
    # toolbox.register("map", pool.map)
    # toolbox.register("map", futures.map)
    toolbox.register("circuit", singleInd, settings)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.circuit)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutator, settings)
    toolbox.register("mate", crossover, weights_cx=settings["weights_cx"])
    toolbox.register("evaluate", fitness_function, settings)
    toolbox.decorate("mate", checkLength(settings))
    toolbox.decorate("mutate", checkLength(settings))

    P = get_P(len(settings["min_max_selector"]), settings["N"])
    ref_points = tools.uniform_reference_points(
        len(settings["min_max_selector"]), P
    )  # parameters: number of objective, P as defined in NSGA3-paper
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    return toolbox


def tupleConstruct(settings) -> tuple:
    """
    This function generates the tuple (gene) of the genetic programming
    chromosomes. Which are later appended in a list to form an individual
    """
    # Select an operator i.e. quantum gate
    qubits = settings["Circuit"][1][0]
    operator = [gate for gatename, gate in settings["elementary"].items()]
    gate_weights = [gate.random_selection_weight for gatename, gate in settings["elementary"].items()]
    gate_weights_norm = list(np.array(gate_weights / np.sum(gate_weights)))
    random_gate = seeded_rng().choice(operator, p=gate_weights_norm)

    if random_gate.non_func:
        if random_gate.params == 0:
            return (random_gate, seeded_rng().choice(qubits, random_gate.arity, False))
        else:
            p = list(seeded_rng().random(random_gate.params))
            return (random_gate, seeded_rng().choice(qubits, random_gate.arity, False), p)
    else:
        return (random_gate, seeded_rng().choice(qubits, random_gate.arity, False))

def singleInd(settings) -> list:
    """
    This function generates the single individual of genetic program
    """
    max_gates = settings["max_gates"]
    prob = settings["prob"]
    if settings.get("init_pop") is not None and (seeded_rng().random() <= float(settings["init_pop"]/100)):
        return gate_to_ind(settings["init_operator"], settings)

    if prob is None:
        N_CYCLES = seeded_rng().integers(2, max_gates + 1)
    # N_CYCLES = integers(2, max_gates+1)
    else:
        N_CYCLES = round(seeded_rng().normal(prob[0], prob[1]))
        N_CYCLES = min(max_gates, max(N_CYCLES, 2))  # assert N_CYCLES is in [2, max_gates

    # l = []
    # for i in range(N_CYCLES):
        # tpl = tupleConstruct(settings["operand"], settings["operator"], settings["weights_gates"], settings)
        # l.append(tpl)
    ind = [tupleConstruct(settings) for _ in range(N_CYCLES)]

    return ind


def is_similar(ind1, ind2) -> bool:
    if len(ind1) != len(ind2):  # check length of individuals
        return False

    for gate1, gate2 in zip(ind1, ind2):
        if len(gate1) != len(gate2):  # check length of gates
            return False

        if gate1[0] != gate2[0]:  # check gate-object of gates
            return False

        if (np.array(gate1[1]) != np.array(gate2[1])).any():  # check qubits of gates
            return False

    return True

def list_contains_element(individual_list, ind) -> bool:
    for ele in individual_list:
        if is_similar(ele, ind):
            return True
    return False


def remove_duplicates(pop) -> list:
    deduplicated = []
    for ind in pop:
        if not list_contains_element(deduplicated, ind):
            deduplicated.append(ind)
    return deduplicated


def var_form(settings, ind, parameters):
    circ = settings["Circuit"]
    qc = circ[0]

    for i in range(len(circ[2])):
        qc.append(ind_to_gate(settings, ind, parameters)[1], circ[1][i])
        qc.data[circ[2][i]] = qc.data[-1]
        del qc.data[-1]

    # position = []
    # for i in range(len(qc.data)):
    #    if "barrier" in str(qc.data[i]):
    #        position.append(i)
    # qc2 = qc.copy()
    # for j in range(len(qc2.data) - position[settings["target_pos"] - 1] - 1):
    #    del qc2.data[-1]
    out_state = Statevector.from_instruction(qc)
    # out_state = Statevector.from_instruction(qc)

    return out_state, qc


def get_difference(ind):
    def execute_circ(theta, settings, target=None):
        target = target or settings["target"]
        state = var_form(settings, ind, theta)[0]
        target = np.asarray(target)
        s = np.abs(np.vdot(state.data, target))  # taking real part ensures phase correct comparison
        diff = 1 - s
        return diff

    return execute_circ


def ind_to_gate(settings, ind, parameters):
    # opt = opt or settings["opt_within"]
    num_qubits = len(settings["Circuit"][1][0])

    qc = QuantumCircuit(num_qubits)
    k = 0
    for gate_tpl in ind:
        name = gate_tpl[0].qiskit_name.lower()
        if gate_tpl[0].non_func:
            if len(gate_tpl) == 2:
                getattr(qc, name)(*gate_tpl[1])

            elif len(gate_tpl) == 3:
                getattr(qc, name)(*parameters[k : k + len(gate_tpl[2])], *gate_tpl[1])  # *i[2] is parameters[k:k+len(i[2])]
                k += len(gate_tpl[2])
        else:
            if len(gate_tpl) == 2:
                getattr(qc, name)(*gate_tpl[0].params, *gate_tpl[1])
                # qc.append(getattr(settings["e_gate"], name), gate_tpl[1])
            else:
                logger.error("Parameters inside! ERROR!!")

    # simple circuit optimization
    new_qc = circuit_optimization(qc, settings["opt_within"])

    gate = new_qc.to_instruction()
    return new_qc, gate


# define fitness function
def quantum_state(settings, ind):
    circ = settings["Circuit"]
    # qubits = qubits or settings["qubits"]
    method = settings["numerical_optimizer"]
    # opt_within = opt_within or settings["opt_within"]
    var = 0
    for i in ind:
        if len(i) == 3:
            var = 1
    # numerical parameter optimization
    if var == 1:
        init_parameters = [i[2][j] for i in ind if len(i) == 3 for j in range(len(i[2]))]
        # init_parameters = []
        # for i in ind:
            # if len(i) == 3:
                # for j in range(len(i[2])):
                    # init_parameters.append(i[2][j])
        if settings["use_numerical_optimizer"]:
            difference = get_difference(ind)
            res = minimize(difference, init_parameters, args=settings, method=method)
            final_param = list(res.x)
        else:
            final_param = init_parameters.copy()
    else:
        final_param = None

    # write optimized parameters to ind
    if final_param is not None:
        k = 0
        for count, gate in enumerate(ind):
            if len(gate) == 3:
                gate = list(gate)
                gate[2] = final_param[k:k+len(gate[2])]  # *gate[2] is parameters[k:k+len(i[2])]
                k += len(gate[2])
                ind[count] = gate

    # create QC
    qc = circ[0]
    oracle = ind_to_gate(settings, ind, final_param)
    pos = len(qc.data)
    for i in range(len(circ[2])):
        qc.append(oracle[1], circ[1][i])
        qc.data[circ[2][i]] = qc.data[pos]
        del qc.data[pos]
    """
    position = []
    for i in range(len(qc.data)):
        if "barrier" in str(qc.data[i]):
            position.append(i)
    qc2 = qc.copy()
    for j in range(len(qc2.data) - position[settings["target_pos"] - 1] - 1):
        del qc2.data[len(qc2.data) - 1]"""

    state = Statevector.from_instruction(qc)
    # state = Statevector.from_instruction(qc)

    # qc.data.pop(0) #get rid of initialization of circuit

    return state, oracle, final_param, qc


# fitness function
def fitness_function(settings, ind):

    target = settings["target"]
    out_state, oracle_qc, params, overall_qc = quantum_state(settings, ind)

    # overlap
    target = np.asarray(target)
    s = round(np.abs(np.vdot(out_state.data, target)), 6)  # real part instead of abs?
    # number of gates
    num_gates = len(oracle_qc[0].data)
    for i in ind:
        if i[0] == "Identity":
            num_gates -= 1

    # depth
    d = oracle_qc[0].depth()
    for i in ind:
        if i[0] == "Identity":
            d -= 1

    # number of non-local gates
    nl = oracle_qc[0].num_nonlocal_gates()

    # number of parameters
    if params is None:
        p = 0
    else:
        p = len(params)

    return s, num_gates, d, nl, p


def crossover(circ1, circ2, weights_cx):
    cx = [0, 1, 2, 3]
    cxtype = seeded_rng().choice(cx, p=weights_cx)
    l1 = len(circ1)
    l2 = len(circ2)

    while (cxtype == 2 or cxtype == 3) and (l1 <= 2 or l2 <= 2):
        cxtype = seeded_rng().choice(cx, p=weights_cx)


    # One point crossover where length of individuals may change
    if cxtype == 0:
        cxpoint1 = seeded_rng().integers(1, len(circ1))
        cxpoint2 = seeded_rng().integers(1, len(circ2))
        circ1[cxpoint1:], circ2[cxpoint2:] = circ2[cxpoint2:], circ1[cxpoint1:]

    # One point crossover where length of individuals remains same
    elif cxtype == 1:
        smaller_size = min(l1, l2)
        cxpoint = seeded_rng().integers(1, smaller_size)
        circ1[cxpoint:], circ2[cxpoint:] = circ2[cxpoint:], circ1[cxpoint:]

    # two point crossover where length of individuals remains same
    elif cxtype == 2:
        size = min(l1, l2)
        cxpoint1 = seeded_rng().integers(1, size)
        cxpoint2 = seeded_rng().integers(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        circ1[cxpoint1:cxpoint2], circ2[cxpoint1:cxpoint2] = circ2[cxpoint1:cxpoint2], circ1[cxpoint1:cxpoint2]

    # two point crossover where length of individuals may change
    elif cxtype == 3:
        cxp11 = seeded_rng().integers(1, l1)
        cxp12 = seeded_rng().integers(1, l1 - 1)
        if cxp12 >= cxp11:
            cxp12 += 1
        else:
            cxp11, cxp12 = cxp12, cxp11

        cxp21 = seeded_rng().integers(1, l2)
        cxp22 = seeded_rng().integers(1, l2 - 1)
        if cxp22 >= cxp21:
            cxp22 += 1
        else:
            cxp21, cxp22 = cxp22, cxp21

        temp1 = circ1[cxp11:cxp12]
        for n in range(cxp12 - cxp11):
            del circ1[cxp11]
        for n in range(cxp22 - cxp21):
            circ1.insert(cxp11 + n, circ2[cxp21 + n])

        for n in range(cxp22 - cxp21):
            del circ2[cxp21]
        for n in range(cxp12 - cxp11):
            circ2.insert(cxp21 + n, temp1[n])

    return circ1, circ2


def mutator(settings, ind):
    qubits = settings["Circuit"][1][0]
    max_gates = settings["max_gates"]
    weights_mut = settings["weights_mut"]

    il = len(ind)
    mutators = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    if not settings["use_numerical_optimizer"]:
        mutators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    muttype = seeded_rng().choice(mutators, p=weights_mut)

    while (muttype == 6 and il <= 2) or (muttype == 7 and il <= 4) or (muttype == 8 and il <= 2):
        muttype = seeded_rng().choice(mutators, p=weights_mut)

    """while 1:
        if muttype == 6 and il <= 2:
            muttype = seeded_rng().choice(mutators, p=weights_mut)
        elif muttype == 7 and il <= 4:
            muttype = seeded_rng().choice(mutators, p=weights_mut)
        elif muttype == 8 and il <= 2:
            muttype = seeded_rng().choice(mutators, p=weights_mut)
        else:
            break"""

    # insert
    if muttype == 0:
        i = seeded_rng().integers(0, il)
        tpl = tupleConstruct(settings)
        ind.insert(i, tpl)

    # delete
    elif muttype == 1:
        i = seeded_rng().integers(0, il)
        del ind[i]

    # swap
    elif muttype == 2:
        i, j = seeded_rng().choice(range(il), 2, False)
        ind[i], ind[j] = ind[j], ind[i]

    # change whole gate
    elif muttype == 3:
        i = seeded_rng().integers(0, il)
        ind[i] = tupleConstruct(settings)

    # change only target/control-qubits as in Multi-objective (2018) paper
    elif muttype == 4:
        # i = seeded_rng().integers(0, il)
        # temp = list(ind[i])
        # if len(temp[1]) == 1:
            # qb = [seeded_rng().choice(operand)]
        random_gate_idx = seeded_rng().integers(0, il)
        new_qubits = seeded_rng().choice(qubits, ind[random_gate_idx][0].arity, False)
        # else:
            # qb = seeded_rng().choice(operand, len(ind[i][1]), False)
        # temp[1] = qb
        tmp = list(ind[random_gate_idx])
        tmp[1] = new_qubits
        ind[random_gate_idx] = tuple(tmp)
        #ind[i] = tuple(tmp)

    # move gate to different position in circuit
    elif muttype == 5:
        i = seeded_rng().integers(0, il)
        a = ind[i]
        del ind[i]
        j = seeded_rng().integers(0, il - 1)
        ind.insert(j, a)

    # replace sequence with random sequence of different size
    elif muttype == 6 and il > 2:
        m, n = seeded_rng().choice(range(il), 2, False)
        i = min(m, n)
        j = max(m, n)

        for k in range(j - i + 1):
            del ind[i]

        max_len = max_gates - len(ind)
        for n in range(seeded_rng().integers(max_len + 1)):
            tpl = tupleConstruct(settings)
            ind.insert(i + n, tpl)

    # swap two random sequences in chromosome
    elif muttype == 7 and il > 4:
        i, j, k, l1 = seeded_rng().choice(range(il), 4, False)
        lis = list([i, j, k, l1])
        lis.sort()
        a, b, c, d = lis[0], lis[1], lis[2], lis[3]

        temp1 = ind[a : b + 1]
        temp2 = ind[c : d + 1]
        for n in range(d + 1 - c):
            del ind[c]
        for n in range(d + 1 - c):
            ind.insert(b + 1 + n, temp2[n])
        for n in range(b + 1 - a):
            ind.insert(c + b + 1 - a + n, temp1[n])
        for n in range(a, b + 1):
            del ind[a]

    # random permutation of gates within a sequence
    elif muttype == 8 and il > 2:
        m, n = seeded_rng().choice(range(il), 2, False)
        i = min(m, n)
        j = max(m, n)

        temp = ind[i : j + 1]
        seeded_rng().shuffle(temp)
        temp = list(temp)

        for k in range(i, j + 1):
            del ind[i]
        for n in range(j + 1 - i):
            ind.insert(i + n, temp[n])

    # use the inverted of the gate, gate has to be added in "Gates.ipybn"
    # elif muttype==9:
    #   i = integers(0,l)
    #  circ[i][0]=circ[i][0]+'_inverted'

    # additional mutator for comparison with [42]
    elif muttype == 9:
        i = seeded_rng().integers(0, il)
        temp = list(ind[i])
        if len(temp) == 3:
            len_param = len(temp[2])
            p = seeded_rng().uniform(0, np.pi, len_param)
            temp[2] = p
        ind[i] = tuple(temp)

    return (ind,)


def checkLength(settings):
    min = 2
    max = settings["max_gates"]

    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                if len(child) < min:
                    for n in range(min - len(child)):
                        gate = tupleConstruct(settings)
                        child.insert(len(child) + n, gate)
                if len(child) > max:
                    for n in range(len(child) - settings["max_gates"]):
                        del child[0]
            return offspring

        return wrapper

    return decorator


def nsga3(toolbox, settings, seed=None):
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=settings["N"])
    pareto = tools.ParetoFront(similar=is_similar)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pareto.update(pop)

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    logger.info(logbook.stream)
    logger.info(pareto[0].fitness.values)

    # Begin the generational process
    time_stamps = []
    start_time = time.time()
    fitness_values_generations = []
    evals = []
    HVs = []
    temp = settings["opt_within"]

    for gen in range(1, settings["NGEN"]):
        cxpb = settings["CXPB"] or 1.0
        mutpb = settings["MUTPB"] or 1.0

        # CO only every x. generation: syntax ["opt_x_gen"]: [gen_x(int),["Q2", "CC", etc.]]
        if settings["opt_x_gen"] is not None:
            if gen % settings["opt_x_gen"][0] == 0:
                settings["opt_within"] = settings["opt_x_gen"][1]
            else:
                settings["opt_within"] = temp

        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        currenttime = time.time() - start_time
        time_stamps.append(currenttime)
        if (settings.get("time_limit") is not None) and (currenttime > settings["time_limit"]):
            break

        # remove duplicates and fill with new individuals
        offspring = remove_duplicates(offspring)
        x = settings["N"] - len(offspring)
        for i in range(x):
            ind_i = toolbox.individual()
            ind_i.fitness.values = toolbox.evaluate(ind_i)
            offspring.append(ind_i)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # keep only offspring individuals with certain overlap
        if settings["overlap_const"] == "yes":
            ov_c = 0.3 + 0.6 * gen / settings["NGEN"]
            for i in reversed(range(len(offspring))):
                if offspring[i].fitness.values[0] < ov_c:
                    del offspring[i]
            while (len(offspring) < settings["N"]):
                ind_i = toolbox.individual()
                ind_i.fitness.values = toolbox.evaluate(ind_i)
                if ind_i.fitness.values[0] > ov_c:
                    offspring.append(ind_i)

        pareto.update(offspring)

        fitness_values_curr_gen = []
        for i in range(len(pareto)):
            fitness_values_curr_gen.append(pareto[i].fitness.values)
        fitness_values_generations.append(fitness_values_curr_gen)

        # TODO: Stefan disabled this because it caused errors...
        # if len(pareto) > 1:
        #     try:
        #         HV = benchtools.hypervolume(pareto)
        #     except:
        #         breakpoint()
        #         pass
        # elif len(pareto) < 2:
        #     HV = 1000000000.0
        # HVs.append(HV)

        HVs.append(None)

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, settings["N"])

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        evals.append(len(invalid_ind))

        logger.info(logbook.stream)
        logger.info(pareto[0].fitness.values)

    return pop, pareto, logbook, time_stamps, fitness_values_generations, evals, HVs


def circuit_optimization(qc, opt):
    if (opt is not None) and (opt != "None"):
        # new_qc = 0
        count = 0
        for i in opt:
            if count > 0:
                qc = new_qc
            count += 1
            if i == "CC":
                pass_ = CommutativeCancellation()
                pm = PassManager(pass_)
                new_qc = pm.run(qc)
            elif i == "OQ1":
                pass_ = Optimize1qGatesDecomposition()
                pm = PassManager(pass_)
                new_qc = pm.run(qc)
            elif i == "Q1":
                new_qc = transpile(qc, optimization_level=1)
            elif i == "Q2":
                new_qc = transpile(qc, optimization_level=2)
            # elif i=="ZX":
            #   new_qc = zx_optimize(qc)
        dag_circuit = circuit_to_dag(new_qc)
        num_gates = dag_circuit.size()
        if num_gates >= 2:
            return new_qc

    return qc


"""def zx_optimize(circ):

    basis=['x','z','rz','u1','u2','h','cx','cz','ccx','ccz']
    new_circ=transpile(circ, basis_gates=basis, optimization_level=0)

    circ2=new_circ.qasm()
    #print(circ2)
    circ3=zx.qasm(circ2)
    circ4=zx.basic_optimization(circ3)
    circ5=circ4.to_qasm()
    #print(circ5)
    qc=QuantumCircuit.from_qasm_str(circ5)

    return qc
"""


def fitness_from_qc(qc2, settings):
    """
    #based on initial state:
    qc1=QuantumCircuit(qubits)
    qc1.initialize(init, qc1.qubits)
    qc=qc1+qc2
    out_state = Statevector.from_instruction(qc)"""
    circ = settings["Circuit"]
    target = settings["target"]

    qc2_gate = qc2.to_gate()
    qc = circ[0]
    pos = len(qc.data)
    for i in range(len(circ[2])):
        qc.append(qc2_gate, circ[1][i])
        qc.data[circ[2][i]] = qc.data[pos]
        del qc.data[pos]
    """
    position = []
    for i in range(len(qc.data)):
        if "barrier" in str(qc.data[i]):
            position.append(i)
    qc3 = qc.copy()
    for j in range(len(qc3.data) - position[settings["target_pos"] - 1] - 1):
        del qc3.data[len(qc3.data) - 1]"""

    out_state = Statevector.from_instruction(qc)
    # out_state = Statevector.from_instruction(qc)

    # overlap
    target = np.asarray(target)
    s = round(np.abs(np.vdot(out_state.data, target)), 6)  # real part ensures phase correct comparison

    # number of gates
    dag_circuit = circuit_to_dag(qc2)
    num_gates = dag_circuit.size()

    # depth
    d = qc2.depth()

    # number of non-local gates
    nl = qc2.num_nonlocal_gates()

    # number of parameters given in get_best_circ

    return s, num_gates, d, nl


def get_P(M, N):
    dis1 = N - 1
    for i in range(1, N):
        a = scisp.binom(i + M - 1, i)
        dis2 = abs(N - a)
        if dis2 < dis1:
            dis1 = dis2
        else:
            break

    return i - 1


# Möglich: o:overlap, g:number of gates, d: depth, nl: number of nl-gates, p: number of params
def reduce_pareto(pareto, settings):
    reduce = settings["reduce_pareto"]
    fit_vals = {"o": 0, "g": 1, "d": 2, "nl": 3, "p": 4}

    for element in reduce:
        s = element.strip().split()
        f_val = fit_vals.get(s[0])
        """if s[0] == "o":
            f_val = 0
        elif s[0] == "g":
            f_val = 1
        elif s[0] == "d":
            f_val = 2
        elif s[0] == "nl":
            f_val = 3
        elif s[0] == "p":
            f_val = 4"""
        if fit_vals is None:
            print("Warning: Fitness values must be one of 'o','g', 'd', 'nl', 'p'!")
        # ineq = s[1]
        # constr = float(s[2])

        for i in reversed(range(len(pareto))):
            if eval(s[2] + s[1] + str(pareto[i].fitness.values[f_val])):
                del pareto[i]

            else:
                print("Warning: inequality operator must be either '<' or '>'!")
    return pareto


def _get_pareto_final_qc_manual(settings, popu):
    pareto_front = []
    for indi in popu:
        out_state, oracle, params, _ = quantum_state(settings, indi)
        qc = oracle[0]
        if settings["opt_select"] is not None:
            qc = circuit_optimization(qc, settings["opt_select"])
        fitness = list(fitness_from_qc(qc, settings))
        if params is None:
            p = 0
        else:
            p = len(params)
        fitness.append(p)
        ind = [qc, fitness, out_state]
        pareto_front.append(ind)

    return pareto_front

def get_order(settings, popu, fitness_values=None):
    order = settings["Sorted_order"]
    np.array(order)
    order_sign = list(np.sign(np.array(order)))
    order = list(np.abs(order))
    if fitness_values is None:
        liste = sorted(
            range(len(popu)),
            key=lambda i: (
                order_sign[0] * popu[i].fitness.values[order[0] - 1],
                order_sign[1] * popu[i].fitness.values[order[1] - 1],
                order_sign[2] * popu[i].fitness.values[order[2] - 1],
                order_sign[3] * popu[i].fitness.values[order[3] - 1],
                order_sign[4] * popu[i].fitness.values[order[4] - 1],
            )
        )
    else:
        liste = sorted(
            range(len(popu)),
            key=lambda i: (
                order_sign[0] * fitness_vals[order[0] - 1][i],
                order_sign[1] * fitness_vals[order[1] - 1][i],
                order_sign[2] * fitness_vals[order[2] - 1][i],
                order_sign[3] * fitness_vals[order[3] - 1][i],
                order_sign[4] * fitness_vals[order[4] - 1][i],
            ),
        )

    return liste

def _get_pareto_final_qc_sorted_no_CO(settings, popu):
    liste = get_order(settings, popu)
    out_state, qc_gate, params, _ = quantum_state(settings, popu[liste[0]])
    qc1 = qc_gate[0]
    if params is None:
        p1 = 0
    else:
        p1 = len(params)
    return qc1, p1, out_state

def _get_pareto_final_qc_sorted_with_CO(settings, popu):
    pareto_front = f1 = f2 = f3 = f4 = f5 = []
    for indi in popu:
        state, oracle, params, _ = quantum_state(settings, indi)
        qctmp = oracle[0]
        qc = circuit_optimization(qctmp, settings["opt_select"])
        fitness = list(fitness_from_qc(qc, settings))
        if params is None:
            p = 0
        else:
            p = len(params)
        fitness.append(p)
        ind = [qc, fitness, state]
        pareto_front.append(ind)
        f1.append(fitness[0])
        f2.append(fitness[1])
        f3.append(fitness[2])
        f4.append(fitness[3])
        f5.append(fitness[4])
    fitness_vals = [f1, f2, f3, f4, f5]

    liste = get_order(settings, pareto_front, fitness_vals)
    out_state = pareto_front[liste[0]][2]
    qc1 = pareto_front[liste[0]][0]
    p1 = pareto_front[liste[0]][1][4]

    return qc1, p1, out_state

def _get_pareto_final_qc_weighted(settings, popu):
    if settings["opt_select"] is None:
        m1 = -1000
        for i in popu:
            m2 = np.dot(np.array(settings["weights2"]), np.array(i.fitness.values))
            if m2 > m1:
                m1 = m2
                ind = i
                p1 = i.fitness.values[4]
        out_state, qc_gate, params, _ = quantum_state(settings, ind)
        qc1 = qc_gate[0]
    else:
        m1 = -1000
        for indi in popu:
            state, qc_gate, params, _ = quantum_state(settings, indi)
            qc = qc_gate[0]
            new_qc = circuit_optimization(qc, settings["opt_select"])
            fitness = list(fitness_from_qc(new_qc, settings))
            if params is None:
                p = 0
            else:
                p = len(params)
            fitness.append(p)
            m2 = np.dot(np.array(settings["weights2"]), np.array(fitness))
            if m2 > m1:
                m1 = m2
                qc1 = new_qc
                ind = indi
                p1 = p
                out_state = state
    return qc1, p1, out_state

def get_pareto_final_qc(settings, popu):
    if settings["sel_scheme"] == "Manual":
        return _get_pareto_final_qc_manual(settings, popu)

    if settings["sel_scheme"] is None or settings["sel_scheme"] == "Sorted":
        if settings["opt_select"] is None:
            qc1, p1, out_state = _get_pareto_final_qc_sorted_no_CO(settings, popu)
        else:
            qc1, pq, out_state = _get_pareto_final_qc_sorted_with_CO(settings, popu)

    if settings["sel_scheme"] == "Weighted":
        qc1, pq, out_state = _get_pareto_final_qc_weighted(settings, popu)

    # optimization of final QC
    final_qc = circuit_optimization(qc1, settings["opt_final"])
    f = list(fitness_from_qc(final_qc, settings))
    f.append(p1)

    return final_qc, f, out_state


def gate_to_ind(qc, settings):

    ind = []
    gates_var = [gate for gatename, gate in settings["elementary"].items()]

    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

    dag = circuit_to_dag(qc)
    for gate_qc in dag.gate_nodes():
        for gate_var in gates_var:
            if gate_qc.name == gate_var.qiskit_name.lower():
                optr = gate_var
        # params = settings["par_nums"].get(optr)
        if optr.non_func:
            if optr.params == 0:
                ind.append((optr, np.array([int(qubit.index) for qubit in gate_qc.qargs])))
            else:
                p = list(seeded_rng().random(optr.params))
                ind.append((optr, np.array([int(qubit.index) for qubit in gate_qc.qargs]), p))
        else:
            ind.append((optr, np.array([int(qubit.index) for qubit in gate_qc.qargs])))

    return ind


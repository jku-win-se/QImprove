# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:25:28 2022

@author: fege9
"""

from dataclasses import dataclass

# elementary Quantum Gates

Non_func_gates = ["x", "y", "z", "cx", "cy", "cz", "swap", "h", "rz", "rx", "ry", "u", "rzz", "ryy", "rxx", "crx",
                  "cry", "crz", "cp", "cu", "tdg", "t", "ch", "csx", "iswap"]

@dataclass
class Elementary():
    qiskit_name: str
    name: str
    arity: int
    params: int
    non_func: bool
    random_selection_weight: float = 1

gate_set = dict(
    H=Elementary("h", "Hadamard", arity=1, params=0, non_func=True),
    X=Elementary("x", "Not", arity=1, params=0, non_func=True),
    Y=Elementary("y", "Pauli Y", arity=1, params=0, non_func=True),
    Z=Elementary("z", "Pauli Z", arity=1, params=0, non_func=True),
    CX=Elementary("cx", "Controlled Not", arity=2, params=0, non_func=True),
    CY=Elementary("cy", "Controlled Y", arity=2, params=0, non_func=True),
    CZ=Elementary("cz", "Controlled Z", arity=2, params=0, non_func=True),
    RZ=Elementary("rz", "Rotate Z", arity=1, params=1, non_func=True),
    RX=Elementary("rx", "Rotate X", arity=1, params=1, non_func=True),
    RY=Elementary("ry", "Rotate Y", arity=1, params=1, non_func=True),
    U=Elementary("u", "U", arity=1, params=3, non_func=True),
    RZZ=Elementary("rzz", "Rotate ZZ", arity=2, params=1, non_func=True),
    RXX=Elementary("rxx", "Rotate XX", arity=2, params=1, non_func=True),
    RYY=Elementary("ryy", "Rotate YY", arity=2, params=1, non_func=True),
    SWAP=Elementary("swap", "Swap", arity=2, params=0, non_func=True),
    CRX=Elementary("crx", "Controlled RX", arity=2, params=1, non_func=True),
    CRY=Elementary("cry", "Controlled RY", arity=2, params=1, non_func=True),
    CRZ=Elementary("crz", "Controlled RZ", arity=2, params=1, non_func=True),
    CP=Elementary("cp", "Controlled P", arity=2, params=1, non_func=True),
    CU=Elementary("cu", "Controlled U", arity=2, params=4, non_func=True),
    TDG=Elementary("tdg", "TDG", arity=1, params=0, non_func=True),
    T=Elementary("t", "T", arity=1, params=0, non_func=True),
    CH=Elementary("ch", "CH", arity=2, params=0, non_func=True),
    CSX=Elementary("csx", "CSX", arity=2, params=0, non_func=True),
    ISWAP=Elementary("iswap", "ISWAP", arity=2, params=0, non_func=True)
)

"""
H = "Hadamard"
X = "Not"
Y = "Pauli Y"
Z = "Pauli Z"
CX = "Controlled Not"
CY = "Controlled Y"
CZ = "Controlled Z"
RZ = "Rotate Z"
RX = "Rotate X"
RY = "Rotate Y"
U = "U"
RZZ = "Rotate ZZ"
RXX = "Rotate XX"
RYY = "Rotate YY"
SWAP = "Swap"
CRX = "Controlled RX"
CRY = "Controlled RY"
CRZ = "Controlled RZ"
CP = "Controlled P"
CU = "Controlled U"
TDG = "TDG"
T = "T"
CH = "CH"
CSX = "CSX"
ISWAP = "ISWAP"


gateArity = {
    H: 1,
    X: 1,
    Y: 1,
    Z: 1,
    CX: 2,
    CY: 2,
    CZ: 2,
    SWAP: 2,
    RZ: 1,
    RX: 1,
    RY: 1,
    U: 1,
    RZZ: 2,
    RXX: 2,
    RYY: 2,
    CRX: 2,
    CRY: 2,
    CRZ: 2,
    CP: 2,
    CU: 2,
    TDG: 1,
    T: 1,
    CH: 2,
    CSX: 2,
    ISWAP: 2
}

gateName = {
    H: "h",
    X: "x",
    Y: "y",
    Z: "z",
    CX: "cx",
    CY: "cy",
    CZ: "cz",
    SWAP: "swap",
    RZ: "rz",
    RX: "rx",
    RY: "ry",
    U: "u",
    RZZ: "rzz",
    RXX: "rxx",
    RYY: "ryy",
    CRX: "crx",
    CRY: "cry",
    CRZ: "crz",
    CP: "cp",
    CU: "cu",
    TDG: "tdg",
    T: "t",
    CH: "ch",
    CSX: "csx",
    ISWAP: "iswap"
}

# give number of parameters for each gate
gateParams = {H: 0, X: 0, Y: 0, Z: 0, CX: 0, CY: 0, CZ: 0, SWAP: 0, RZ: 1, RX: 1, RY: 1, U: 3, RZZ: 1, RXX: 1, RYY: 1,
              CRX: 1, CRY: 1, CRZ: 1, CP: 1, CU: 4, TDG: 0, T: 0, CH: 0, CSX: 0, ISWAP: 0}


class ElementaryGate:
    def __init__(self):
        self.type = "ElementaryGate"
        
"""

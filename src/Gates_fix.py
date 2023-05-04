# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:28:25 2022

@author: fege9
"""

# elementary Quantum Gates
from qiskit import QuantumCircuit
from dataclasses import dataclass
import numpy as np

"RX, RY, and RZ rotations with fixed discrete angles 2πk/8 with k between 1-7"

Non_func_gates = ["x", "y", "z", "cx", "cy", "cz", "swap", "h"]

@dataclass
class Elementary():
    qiskit_name: str
    name: str
    arity: int
    params: list
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
    RZ1=Elementary("rz", "Rotate Z 1/8", arity=1, params=[(1 / 8) * 2 * np.pi], non_func=False),
    RZ2=Elementary("rz", "Rotate Z 2/8", arity=1, params=[(2 / 8) * 2 * np.pi], non_func=False),
    RZ3=Elementary("rz", "Rotate Z 3/8", arity=1, params=[(3 / 8) * 2 * np.pi], non_func=False),
    RZ4=Elementary("rz", "Rotate Z 4/8", arity=1, params=[(4 / 8) * 2 * np.pi], non_func=False),
    RZ5=Elementary("rz", "Rotate Z 5/8", arity=1, params=[(5 / 8) * 2 * np.pi], non_func=False),
    RZ6=Elementary("rz", "Rotate Z 6/8", arity=1, params=[(6 / 8) * 2 * np.pi], non_func=False),
    RZ7=Elementary("rz", "Rotate Z 7/8", arity=1, params=[(7 / 8) * 2 * np.pi], non_func=False),
    RX1=Elementary("rx", "Rotate X 1/8", arity=1, params=[(1 / 8) * 2 * np.pi], non_func=False),
    RX2=Elementary("rx", "Rotate X 2/8", arity=1, params=[(2 / 8) * 2 * np.pi], non_func=False),
    RX3=Elementary("rx", "Rotate X 3/8", arity=1, params=[(3 / 8) * 2 * np.pi], non_func=False),
    RX4=Elementary("rx", "Rotate X 4/8", arity=1, params=[(4 / 8) * 2 * np.pi], non_func=False),
    RX5=Elementary("rx", "Rotate X 5/8", arity=1, params=[(5 / 8) * 2 * np.pi], non_func=False),
    RX6=Elementary("rx", "Rotate X 6/8", arity=1, params=[(6 / 8) * 2 * np.pi], non_func=False),
    RX7=Elementary("rx", "Rotate X 7/8", arity=1, params=[(7 / 8) * 2 * np.pi], non_func=False),
    RY1=Elementary("ry", "Rotate Y 1/8", arity=1, params=[(1 / 8) * 2 * np.pi], non_func=False),
    RY2=Elementary("ry", "Rotate Y 2/8", arity=1, params=[(2 / 8) * 2 * np.pi], non_func=False),
    RY3=Elementary("ry", "Rotate Y 3/8", arity=1, params=[(3 / 8) * 2 * np.pi], non_func=False),
    RY4=Elementary("ry", "Rotate Y 4/8", arity=1, params=[(4 / 8) * 2 * np.pi], non_func=False),
    RY5=Elementary("ry", "Rotate Y 5/8", arity=1, params=[(5 / 8) * 2 * np.pi], non_func=False),
    RY6=Elementary("ry", "Rotate Y 6/8", arity=1, params=[(6 / 8) * 2 * np.pi], non_func=False),
    RY7=Elementary("ry", "Rotate Y 7/8", arity=1, params=[(7 / 8) * 2 * np.pi], non_func=False),
    SWAP=Elementary("swap", "Swap", arity=2, params=0, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    # U=Elementary("u", "U", arity=1, params=3, non_func=True),
    RZZ1=Elementary("rzz", "Rotate ZZ 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    RZZ2=Elementary("rzz", "Rotate ZZ 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    RZZ3=Elementary("rzz", "Rotate ZZ 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    RZZ4=Elementary("rzz", "Rotate ZZ 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    RZZ5=Elementary("rzz", "Rotate ZZ 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    RZZ6=Elementary("rzz", "Rotate ZZ 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    RZZ7=Elementary("rzz", "Rotate ZZ 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    RXX1=Elementary("rxx", "Rotate XX 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    RXX2=Elementary("rxx", "Rotate XX 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    RXX3=Elementary("rxx", "Rotate XX 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    RXX4=Elementary("rxx", "Rotate XX 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    RXX5=Elementary("rxx", "Rotate XX 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    RXX6=Elementary("rxx", "Rotate XX 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    RXX7=Elementary("rxx", "Rotate XX 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    RYY1=Elementary("ryy", "Rotate YY 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    RYY2=Elementary("ryy", "Rotate YY 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    RYY3=Elementary("ryy", "Rotate YY 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    RYY4=Elementary("ryy", "Rotate YY 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    RYY5=Elementary("ryy", "Rotate YY 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    RYY6=Elementary("ryy", "Rotate YY 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    RYY7=Elementary("ryy", "Rotate YY 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    CRX1=Elementary("crx", "Controlled RX 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    CRX2=Elementary("crx", "Controlled RX 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    CRX3=Elementary("crx", "Controlled RX 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    CRX4=Elementary("crx", "Controlled RX 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    CRX5=Elementary("crx", "Controlled RX 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    CRX6=Elementary("crx", "Controlled RX 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    CRX7=Elementary("crx", "Controlled RX 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    CRY1=Elementary("cry", "Controlled RY 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    CRY2=Elementary("cry", "Controlled RY 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    CRY3=Elementary("cry", "Controlled RY 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    CRY4=Elementary("cry", "Controlled RY 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    CRY5=Elementary("cry", "Controlled RY 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    CRY6=Elementary("cry", "Controlled RY 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    CRY7=Elementary("cry", "Controlled RY 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    CRZ1=Elementary("crz", "Controlled RZ 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    CRZ2=Elementary("crz", "Controlled RZ 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    CRZ3=Elementary("crz", "Controlled RZ 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    CRZ4=Elementary("crz", "Controlled RZ 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    CRZ5=Elementary("crz", "Controlled RZ 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    CRZ6=Elementary("crz", "Controlled RZ 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    CRZ7=Elementary("crz", "Controlled RZ 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    CP1=Elementary("cp", "Controlled P 1/8", arity=2, params=[(1 / 8) * 2 * np.pi], non_func=False),
    CP2=Elementary("cp", "Controlled P 2/8", arity=2, params=[(2 / 8) * 2 * np.pi], non_func=False),
    CP3=Elementary("cp", "Controlled P 3/8", arity=2, params=[(3 / 8) * 2 * np.pi], non_func=False),
    CP4=Elementary("cp", "Controlled P 4/8", arity=2, params=[(4 / 8) * 2 * np.pi], non_func=False),
    CP5=Elementary("cp", "Controlled P 5/8", arity=2, params=[(5 / 8) * 2 * np.pi], non_func=False),
    CP6=Elementary("cp", "Controlled P 6/8", arity=2, params=[(6 / 8) * 2 * np.pi], non_func=False),
    CP7=Elementary("cp", "Controlled P 7/8", arity=2, params=[(7 / 8) * 2 * np.pi], non_func=False),
    # CU=Elementary("cu", "Controlled U", arity=2, params=4, non_func=True),
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
# I='Identity'

# CH='Controlled Hadamard'
CX = "Controlled Not"
CY = "Controlled Y"
CZ = "Controlled Z"
RZ1 = "Rotate Z 1/8"
RZ2 = "Rotate Z 2/8"
RZ3 = "Rotate Z 3/8"
RZ4 = "Rotate Z 4/8"
RZ5 = "Rotate Z 5/8"
RZ6 = "Rotate Z 6/8"
RZ7 = "Rotate Z 7/8"
RX1 = "Rotate X 1/8"
RX2 = "Rotate X 2/8"
RX3 = "Rotate X 3/8"
RX4 = "Rotate X 4/8"
RX5 = "Rotate X 5/8"
RX6 = "Rotate X 6/8"
RX7 = "Rotate X 7/8"
RY1 = "Rotate Y 1/8"
RY2 = "Rotate Y 2/8"
RY3 = "Rotate Y 3/8"
RY4 = "Rotate Y 4/8"
RY5 = "Rotate Y 5/8"
RY6 = "Rotate Y 6/8"
RY7 = "Rotate Y 7/8"
# CU='Controlled U'


# S='Clifford S'
# SC='Clifford S Conjugate'

# T='SquareRoot S'
# TC='T Conjugate'

SWAP = "Swap"


gateArity = {
    H: 1,
    X: 1,
    Y: 1,
    Z: 1,
    #        I:1,
    #
    # CH:2,
    CX: 2,
    CY: 2,
    CZ: 2,
    #
    # S:1,
    # SC:1,
    #
    # T:1,
    # TC:1,
    #
    SWAP: 2,
    RZ1: 1,
    RZ2: 1,
    RZ3: 1,
    RZ4: 1,
    RZ5: 1,
    RZ6: 1,
    RZ7: 1,
    RX1: 1,
    RX2: 1,
    RX3: 1,
    RX4: 1,
    RX5: 1,
    RX6: 1,
    RX7: 1,
    RY1: 1,
    RY2: 1,
    RY3: 1,
    RY4: 1,
    RY5: 1,
    RY6: 1,
    RY7: 1,
    #        CU:2
}

gateName = {
    H: "h",
    X: "x",
    Y: "y",
    Z: "z",
    #        I:'i',
    # CH:'ch',
    CX: "cx",
    CY: "cy",
    CZ: "cz",
    # S:'s',
    # SC:'sdg',
    # T:'t',
    # TC:'tdg',
    SWAP: "swap",
    RZ1: "rz1",
    RZ2: "rz2",
    RZ3: "rz3",
    RZ4: "rz4",
    RZ5: "rz5",
    RZ6: "rz6",
    RZ7: "rz7",
    RX1: "rx1",
    RX2: "rx2",
    RX3: "rx3",
    RX4: "rx4",
    RX5: "rx5",
    RX6: "rx6",
    RX7: "rx7",
    RY1: "ry1",
    RY2: "ry2",
    RY3: "ry3",
    RY4: "ry4",
    RY5: "ry5",
    RY6: "ry6",
    RY7: "ry7",
    #       CU:'cu'
}

# give number of parameters for each gate
gateParams = {
    H: 0,
    X: 0,
    Y: 0,
    Z: 0,
    #     I:0,
    #
    # CH:2,
    CX: 0,
    CY: 0,
    CZ: 0,
    #
    # S:1,
    # SC:1,
    #
    # T:1,
    # TC:1,
    #
    SWAP: 0,
    RZ1: 0,
    RZ2: 0,
    RZ3: 0,
    RZ4: 0,
    RZ5: 0,
    RZ6: 0,
    RZ7: 0,
    RX1: 0,
    RX2: 0,
    RX3: 0,
    RX4: 0,
    RX5: 0,
    RX6: 0,
    RX7: 0,
    RY1: 0,
    RY2: 0,
    RY3: 0,
    RY4: 0,
    RY5: 0,
    RY6: 0,
    RY7: 0,
    #  CU:0
}


class ElementaryGate:
    def __init__(self):
        self.type = "ElementaryGate"

    def _rXYZ_generic(self, k: int, xyz: str, target_qubits=None, control_qubits=None, inverse=False):
        # Pass a k and one letter of [x,y,z] as arguments
        qc = QuantumCircuit(1)
        if xyz.lower() == "x":
            qc.rx((k / 8) * 2 * np.pi, 0)
        elif xyz.lower() == "y":
            qc.ry((k / 8) * 2 * np.pi, 0)
        elif xyz.lower() == "z":
            qc.rz((k / 8) * 2 * np.pi, 0)

        gate = qc.to_gate()
        gate.label = f"R{xyz.upper()}{k}"
        return gate

    # RZ Gates

    def rz1(self, *args, **kwargs):
        return self._rXYZ_generic(1, "x", *args, **kwargs)

    def rz2(self, *args, **kwargs):
        return self._rXYZ_generic(2, "x", *args, **kwargs)

    def rz3(self, *args, **kwargs):
        return self._rXYZ_generic(3, "x", *args, **kwargs)

    def rz4(self, *args, **kwargs):
        return self._rXYZ_generic(4, "x", *args, **kwargs)

    def rz5(self, *args, **kwargs):
        return self._rXYZ_generic(5, "x", *args, **kwargs)

    def rz6(self, *args, **kwargs):
        return self._rXYZ_generic(6, "x", *args, **kwargs)

    def rz7(self, *args, **kwargs):
        return self._rXYZ_generic(7, "x", *args, **kwargs)

    # RX Gates

    def rx1(self, *args, **kwargs):
        return self._rXYZ_generic(1, "x", *args, **kwargs)

    def rx2(self, *args, **kwargs):
        return self._rXYZ_generic(2, "x", *args, **kwargs)

    def rx3(self, *args, **kwargs):
        return self._rXYZ_generic(3, "x", *args, **kwargs)

    def rx4(self, *args, **kwargs):
        return self._rXYZ_generic(4, "x", *args, **kwargs)

    def rx5(self, *args, **kwargs):
        return self._rXYZ_generic(5, "x", *args, **kwargs)

    def rx6(self, *args, **kwargs):
        return self._rXYZ_generic(6, "x", *args, **kwargs)

    def rx7(self, *args, **kwargs):
        return self._rXYZ_generic(7, "x", *args, **kwargs)

    # RY Gates

    def ry1(self, *args, **kwargs):
        return self._rXYZ_generic(1, "y", *args, **kwargs)

    def ry2(self, *args, **kwargs):
        return self._rXYZ_generic(2, "y", *args, **kwargs)

    def ry3(self, *args, **kwargs):
        return self._rXYZ_generic(3, "y", *args, **kwargs)

    def ry4(self, *args, **kwargs):
        return self._rXYZ_generic(4, "y", *args, **kwargs)

    def ry5(self, *args, **kwargs):
        return self._rXYZ_generic(5, "y", *args, **kwargs)

    def ry6(self, *args, **kwargs):
        return self._rXYZ_generic(6, "y", *args, **kwargs)

    def ry7(self, *args, **kwargs):
        return self._rXYZ_generic(7, "y", *args, **kwargs)

"""
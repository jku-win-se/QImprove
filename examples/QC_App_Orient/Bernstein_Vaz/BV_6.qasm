OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
x q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0],q[5];
cx q[1],q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[5];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
x q[0];
x q[2];
h q[1];
h q[3];
cx q[0],q[1];
cx q[2],q[3];
h q[1];
h q[3];
x q[0];
x q[2];
h q[0];
h q[1];
h q[2];
h q[3];
h q[1];
h q[3];
cx q[0],q[1];
cx q[2],q[3];
h q[1];
h q[3];
h q[0];
h q[1];
h q[2];
h q[3];
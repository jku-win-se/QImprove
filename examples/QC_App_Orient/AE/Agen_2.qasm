OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry(0.72273425) q[1];
x q[1];
x q[1];
cx q[1],q[0];
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ry(0.72273425) q[2];
x q[2];
x q[2];
cx q[2],q[0];
cx q[2],q[1];
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
ry(0.72273425) q[5];
x q[5];
x q[5];
cx q[5],q[0];
cx q[5],q[1];
cx q[5],q[2];
cx q[5],q[3];
cx q[5],q[4];
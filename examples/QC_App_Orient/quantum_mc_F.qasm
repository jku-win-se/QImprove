OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ry(-1.0881116) q[2];
ry(-0.056799389) q[2];
cx q[0],q[2];
ry(0.056799389) q[2];
cx q[0],q[2];
ry(-0.21769431) q[2];
cx q[1],q[2];
ry(0.21769431) q[2];
cx q[1],q[2];
ry(-0.20819107) q[2];
h q[2];
cx q[1],q[2];
tdg q[2];
cx q[0],q[2];
t q[2];
cx q[1],q[2];
t q[1];
tdg q[2];
cx q[0],q[2];
cx q[0],q[1];
t q[0];
tdg q[1];
cx q[0],q[1];
t q[2];
h q[2];
ry(0.20819107) q[2];
h q[2];
cx q[1],q[2];
tdg q[2];
cx q[0],q[2];
t q[2];
cx q[1],q[2];
t q[1];
tdg q[2];
cx q[0],q[2];
cx q[0],q[1];
t q[0];
tdg q[1];
cx q[0],q[1];
t q[2];
h q[2];
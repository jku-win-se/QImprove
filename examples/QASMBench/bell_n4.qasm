OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[3];
cx q[0],q[2];
rx(pi*-0.25) q[0];
ry(pi*-0.5) q[2];
u(pi*0.5,0,pi*0.75) q[3];
u(pi*0.5,0,pi*0.25) q[2];
rx(pi*0.5) q[3];
cx q[3],q[2];
rx(pi*0.25) q[3];
ry(pi*0.5) q[2];
cx q[2],q[3];
rx(pi*-0.5) q[2];
rz(pi*0.5) q[2];
cx q[3],q[2];
u(pi*0.5,pi*0.5,pi*1.0) q[3];
u(pi*0.5,pi*1.0,pi*1.0) q[2];
ry(pi*0.5) q[2];
ry(pi*-0.5) q[0];
u(pi*0.5,0,pi*0.75) q[1];
u(pi*0.5,0,pi*0.25) q[0];
rx(pi*0.5) q[1];
cx q[1],q[0];
rx(pi*0.25) q[1];
ry(pi*0.5) q[0];
cx q[0],q[1];
rx(pi*-0.5) q[0];
rz(pi*0.5) q[0];
cx q[1],q[0];
u(pi*0.5,pi*0.5,pi*1.0) q[1];
u(pi*0.5,pi*1.0,pi*1.0) q[0];
ry(pi*0.5) q[0];
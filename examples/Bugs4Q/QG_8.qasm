OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
crz(pi/2) q[2],q[0];
crz(pi/4) q[2],q[1];
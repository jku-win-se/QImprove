OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
h q[0];
rz(-2.3540515) q[0];
h q[0];
h q[1];
rz(-2.3540515) q[1];
h q[1];
h q[2];
rz(-2.3540515) q[2];
h q[2];
h q[3];
rz(-2.3540515) q[3];
h q[3];
h q[4];
rz(-2.3540515) q[4];
h q[4];
h q[5];
rz(-2.3540515) q[5];
h q[5];
h q[6];
rz(-2.3540515) q[6];
h q[6];
cx q[0],q[1];
rz(-pi/2) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-pi/2) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-pi/2) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-pi/2) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-pi/2) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-pi/2) q[6];
cx q[5],q[6];


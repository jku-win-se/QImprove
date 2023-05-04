OPENQASM 2.0;
include "qelib1.inc";
qreg q7[2];
swap q7[0],q7[1];
h q7[1];
cp(-pi/2) q7[1],q7[0];
h q7[0];

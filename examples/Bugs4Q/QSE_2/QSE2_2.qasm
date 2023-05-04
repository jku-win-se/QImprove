OPENQASM 2.0;
include "qelib1.inc";
qreg q5[2];
h q5[1];
cp(-pi/2) q5[1],q5[0];
h q5[0];

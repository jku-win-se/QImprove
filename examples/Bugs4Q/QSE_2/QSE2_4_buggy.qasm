OPENQASM 2.0;
include "qelib1.inc";
qreg q27[4];
swap q27[0],q27[3];
swap q27[1],q27[2];
h q27[3];
cp(-pi/2) q27[3],q27[2];
cp(-pi/4) q27[3],q27[1];
cp(-pi/8) q27[3],q27[0];
h q27[2];
cp(-pi/2) q27[2],q27[1];
cp(-pi/4) q27[2],q27[0];
h q27[1];
cp(-pi/2) q27[1],q27[0];
h q27[0];
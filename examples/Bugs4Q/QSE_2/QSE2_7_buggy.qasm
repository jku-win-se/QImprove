OPENQASM 2.0;
include "qelib1.inc";
qreg q30[7];
swap q30[0],q30[6];
swap q30[1],q30[5];
swap q30[2],q30[4];
h q30[6];
cp(-pi/2) q30[6],q30[5];
cp(-pi/4) q30[6],q30[4];
cp(-pi/8) q30[6],q30[3];
cp(-pi/16) q30[6],q30[2];
cp(-pi/32) q30[6],q30[1];
cp(-pi/64) q30[6],q30[0];
h q30[5];
cp(-pi/2) q30[5],q30[4];
cp(-pi/4) q30[5],q30[3];
cp(-pi/8) q30[5],q30[2];
cp(-pi/16) q30[5],q30[1];
cp(-pi/32) q30[5],q30[0];
h q30[4];
cp(-pi/2) q30[4],q30[3];
cp(-pi/4) q30[4],q30[2];
cp(-pi/8) q30[4],q30[1];
cp(-pi/16) q30[4],q30[0];
h q30[3];
cp(-pi/2) q30[3],q30[2];
cp(-pi/4) q30[3],q30[1];
cp(-pi/8) q30[3],q30[0];
h q30[2];
cp(-pi/2) q30[2],q30[1];
cp(-pi/4) q30[2],q30[0];
h q30[1];
cp(-pi/2) q30[1],q30[0];
h q30[0];
OPENQASM 2.0;
include "qelib1.inc";
qreg q36[6];
h q36[5];
cp(-pi/2) q36[5],q36[4];
h q36[4];
cp(-pi/4) q36[5],q36[3];
cp(-pi/2) q36[4],q36[3];
h q36[3];
cp(-pi/8) q36[5],q36[2];
cp(-pi/4) q36[4],q36[2];
cp(-pi/2) q36[3],q36[2];
h q36[2];
cp(-pi/16) q36[5],q36[1];
cp(-pi/8) q36[4],q36[1];
cp(-pi/4) q36[3],q36[1];
cp(-pi/2) q36[2],q36[1];
h q36[1];
cp(-pi/32) q36[5],q36[0];
cp(-pi/16) q36[4],q36[0];
cp(-pi/8) q36[3],q36[0];
cp(-pi/4) q36[2],q36[0];
cp(-pi/2) q36[1],q36[0];
h q36[0];

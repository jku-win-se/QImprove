// Benchmark was created by MQT Bench on 2022-12-15
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.2.2
// Qiskit version: {'qiskit-terra': '0.22.3', 'qiskit-aer': '0.11.1', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.19.2', 'qiskit': '0.39.3', 'qiskit-nature': '0.5.1', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.4.0', 'qiskit-machine-learning': '0.5.0'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
ry(-pi/4) q[0];
ry(-0.95531662) q[1];
ry(-pi/3) q[2];
ry(-1.1071487) q[3];
x q[4];
cz q[4],q[3];
ry(1.1071487) q[3];
cz q[3],q[2];
ry(pi/3) q[2];
cz q[2],q[1];
ry(0.95531662) q[1];
cz q[1],q[0];
ry(pi/4) q[0];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
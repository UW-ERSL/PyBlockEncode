from pyblockencode.qiskit_encoding import PoissonCircuit, ElasticityCircuit
print(PoissonCircuit(m=3, dim=1, disc='fdm').verify()['block_encoding_rel_err'])
print(ElasticityCircuit(m=1, nu=0.3).verify()['block_encoding_rel_err'])

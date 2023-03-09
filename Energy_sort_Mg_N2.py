from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter, ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit.algorithms.optimizers import SLSQP, COBYLA, SPSA
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.properties import AngularMomentum
from qiskit.circuit.library import EvolvedOperatorAnsatz
import math
import sys
import time
import copy

# -----------------------input parameters------------------------
activate = 8
reduce_qubit = True
maxiter = 50

start_time = time.time()
driver = PySCFDriver(
    atom='''N    0.00000000    0.00000000   -0.65403500
            N    0.00000000    0.00000000   -1.77306600
            Mg   0.00000000    0.00000000    1.34596500''',
    basis="sto3g",
    charge=2,
    spin=0,
    unit=DistanceUnit.ANGSTROM)
full_problem = driver.run()
HF_energy = full_problem.reference_energy

# active sapce
if (activate == 4):
    as_transformer = ActiveSpaceTransformer(4, 4)
elif (activate == 6):
    as_transformer = ActiveSpaceTransformer(6, 6)
elif (activate == 8):
    as_transformer = ActiveSpaceTransformer(8, 8)  #---------------------------------
else:
    sys.exit()
as_problem = as_transformer.transform(full_problem)
fermionic_op = as_problem.hamiltonian.second_q_op()

# mapping
if (reduce_qubit):
    converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    qubit_op = converter.convert(fermionic_op, num_particles=as_problem.num_particles)
else:
    converter = QubitConverter(JordanWignerMapper())
    qubit_op = converter.convert(fermionic_op)

# use uccsd ansatz
uccsd = UCC(num_particles=as_problem.num_particles,
            num_spatial_orbitals=as_problem.num_spatial_orbitals,
            excitations='sd',
            qubit_converter=converter)
# build uccsd operator pools
op_pools = uccsd.operators
results = []
delta_results = []
test = []
# calculate Energy contribution compared with HF energy
for i in range(0, len(op_pools), 1):
    test = []
    test.append(op_pools[i])
    
    evo = EvolvedOperatorAnsatz(test)
    estimator = Estimator()
    initial_point = len(evo.operators) * [0]
    vqe_solver = VQEUCCFactory(estimator=estimator,
                                ansatz=evo,
                                optimizer=SLSQP(maxiter=maxiter),
                                initial_point=initial_point)

    vqe_calc = GroundStateEigensolver(converter, vqe_solver)
    vqe_result = vqe_calc.solve(as_problem)
    vqe_value = vqe_result.total_energies[0]
    results.append(vqe_value)

    print("##############################################################")
    print("Reduce qubit: ", reduce_qubit)
    print("Qubit: ", qubit_op.num_qubits)
    print("uccsd items: ", len(uccsd.operators))
    print("evo items: ", len(evo.operators))
    print("Theta number: ", len(vqe_result.raw_result.optimal_parameters))
    print("Cost function evals: ", vqe_result.raw_result.cost_function_evals)
    print("HF energy: ", HF_energy)
    print(f"VQE value: {vqe_value:.6f}")
    print(f"Delta: {(HF_energy-vqe_value):.6f}")

end_time = time.time()
run_time = end_time - start_time
for i in range(0, len(op_pools), 1):
    delta_results.append((HF_energy - results[i]))
print("---------------------------------------------------")
print("Sort from large to small according to E(HF) - vqe_value")
sort_index = []
sort_delta_results = []
if (len(op_pools) != len(delta_results)):
    sys.exit()
for i in range(0, len(delta_results), 1):
    tmp = -100000  # a temporary variable
    index = -1
    for j in range(len(delta_results)):
        if (delta_results[j] > tmp):
            tmp = delta_results[j]
            index = j
            
    delta_results[index] = -100001  # a temporary variable
    sort_index.append(index)
    sort_delta_results.append(tmp)

if (len(sort_delta_results) != len(delta_results)):
    sys.exit()
# show the sorted results from large to small
for i in range(0, len(delta_results), 1):
    print("i = ", sort_index[i], "  %.6f" % sort_delta_results[i])

print("Time: %.2f s" % run_time)
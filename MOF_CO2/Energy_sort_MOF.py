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
    atom='''Mg                 0.23581400    0.01400200    0.08601700
            C                  3.13653000    1.27945000    0.09157700
            C                 -1.58836500    2.61719000    0.21147500
            C                 -2.74256100   -1.29394400   -0.27756100
            O                 -0.60188700    1.86590600    0.23758000
            O                 -1.50909500   -1.05825700   -0.19035500
            O                 -0.49436300   -0.54178200    1.90290800
            C                 -1.53117600    4.08853200    0.45254700
            H                 -1.88330000    4.63865100   -0.42335400
            H                 -2.16659600    4.36963700    1.29552300
            H                 -0.51238100    4.39022100    0.66807300
            O                 -2.80068300    2.05128500   -0.05755600
            H                 -3.54895500    2.67791100   -0.06905200
            O                 -3.36348900   -1.65706100    0.84957000
            H                 -4.31598000   -1.86052400    0.77323500
            O                  3.32062900    1.43209600   -1.24082400
            H                  4.14642200    1.87826500   -1.50966300
            C                 -0.62659100   -1.39739800    2.98898000
            H                 -0.25335200   -0.83578200    3.87707200
            H                 -1.68443800   -1.56645200    3.22977700
            H                 -0.06487700   -2.32661000    2.90017600
            C                  4.18083700    1.77678500    1.03275300
            H                  5.13880300    1.28619700    0.84424900
            H                  4.33437800    2.85111600    0.90619800
            H                  3.88366600    1.58435800    2.05761900
            O                  2.08551200    0.71463900    0.45171800
            O                  0.02455400   -0.01690600   -1.92242000
            C                  0.50719000   -0.54643700   -3.11638800
            H                 -0.26667500   -0.58272900   -3.88953700
            H                  1.25745500    0.18143400   -3.50556000
            H                  1.02832900   -1.49975900   -2.99885200
            C                 -3.48378700   -1.23209000   -1.57009000
            H                 -2.95511500   -0.59151400   -2.26775300
            H                 -3.55199400   -2.23544600   -2.00114400
            H                 -4.49738400   -0.85679000   -1.42951400
            O                  1.07232307   -1.57710089   -0.00695709
            C                  1.72339307   -2.57265189    0.09016191
            O                  2.36045007   -3.55143089    0.18030091''',
    basis="sto3g",
    charge=2,
    spin=0,
    unit=DistanceUnit.ANGSTROM)
full_problem = driver.run()
HF_energy = full_problem.reference_energy
# active sapce
if (activate == 8):
    as_transformer = ActiveSpaceTransformer(8, 8)
elif (activate == 10):
    as_transformer = ActiveSpaceTransformer(10, 10)
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
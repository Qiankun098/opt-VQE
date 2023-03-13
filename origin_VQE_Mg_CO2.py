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
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.opflow import I, X, Y, Z
import math
import sys
import time
import numpy as np
import argparse

# ----------------------------Parameters-----------------------------
prefix = "Origin_VQE_Mg_Co2_active_"
basis = "sto3g"
charge = 2
spin = 0
maxiter = 100
active_space = 8
optimizer = SLSQP(maxiter=maxiter, eps=1e-4, ftol=1e-4)

# set file
results_file = prefix + str(active_space) + "_" + str(active_space) + \
                "_" + "SLSQP" + ".csv"
log_file = prefix + str(active_space) + "_" + str(active_space) + \
                "_" + "SLSQP" + "_info.log"

# ------------------Coordiantes and Reference values------------------------
ref_coord = 0.11036800 # O atom coordinate
atom_Mg = ['''Mg  1.11036755    0.00031369   -0.00000093''',
           '''Mg  1.31036746    0.00012343   -0.00000072''',
           '''Mg  1.51036737   -0.00006683   -0.00000051''',
           '''Mg  1.71036728   -0.00025711   -0.00000028''',
           '''Mg  1.81036723   -0.00035224   -0.00000017''',
           '''Mg  1.91036719   -0.00044739   -0.00000008''',
           '''Mg  2.01036714   -0.00054252    0.00000003''',
           '''Mg  2.11036709   -0.00063765    0.00000013''',
           '''Mg  2.21036705   -0.00073278    0.00000024''',
           '''Mg  2.31036700   -0.00082791    0.00000035''',
           '''Mg  2.41036696   -0.00092304    0.00000046''',
           '''Mg  2.51036691   -0.00101817    0.00000057''',
           '''Mg  2.61036687   -0.00111330    0.00000068''',
           '''Mg  2.71036682   -0.00120843    0.00000079''',
           '''Mg  2.91036673   -0.00139866    0.00000098''',
           '''Mg  3.11036664   -0.00158892    0.00000119''',
           '''Mg  4.11036619   -0.00254023    0.00000225''',
           '''Mg  5.11036574   -0.00349154    0.00000331''',
           ]

atom_Co2 = '''\
C  -1.11138400    0.00037800    0.00000500
O  -2.26002300   -0.00076600   -0.00000200
O   0.11036800    0.00126500   -0.00000200
'''

MgCo2_ref_values_4_4 = [-380.5408631537266,
                        -381.33281266220143,
                        -381.63270516013034,
                        -381.728351920418,
                        -381.741174713052,
                        -381.741341518169,
                        -381.733062881581,
                        -381.718987083122,
                        -381.70260204652,
                        -381.687665703042,
                        -381.674002327378,
                        -381.661275067291,
                        -381.649707503786,
                        -381.639481251555,
                        -381.6232013636178,
                        -381.6118399339273,
                        -381.590156373803,
                        -381.58366759082844,
                        ]

MgCo2_ref_values_6_6 = [-380.59088387036496,
                        -381.373662247625,
                        -381.6696648204708,
                        -381.762507478014,
                        -381.773529540638,
                        -381.770894587111,
                        -381.757665660517,
                        -381.734950615417,
                        -381.708409277188,
                        -381.688784944612,
                        -381.674214885824,
                        -381.661339482055,
                        -381.649739836906,
                        -381.639481251555,
                        -381.62321140370426,
                        -381.6118438669504,
                        -381.59015637550993,
                        -381.5836675908288,
                        ]

MgCo2_ref_values_8_8 = [-380.59767326755014,
                        -381.40406173987685,
                        -381.6927665600249,
                        -381.787571800522,
                        -381.799800031996,
                        -381.798965928168,
                        -381.788716741267,
                        -381.770458147556,
                        -381.746584096613,
                        -381.724459779659,
                        -381.67456407079,
                        -381.661513763651,
                        -381.649843995651,
                        -381.639481251555,
                        -381.6232375591813,
                        -381.6118526888231,
                        -381.59015637888535,
                        -381.5836675908291,
                        ]

# ------------------------Run Simulation----------------------------
start_total_time = time.perf_counter()
results = np.empty((len(atom_Mg), 5))

# set active space
if (active_space == 4):
    ref_values = MgCo2_ref_values_4_4
elif (active_space == 6):
    ref_values = MgCo2_ref_values_6_6
elif (active_space == 8):
    ref_values = MgCo2_ref_values_8_8
else:
    print("Error, bad active para")
    sys.exit()
    
if (len(ref_values) != len(atom_Mg)):
    print("Bad atom parameters!")
    sys.exit()

with open(file=log_file, mode='w') as filename :
    time_tuple = time.localtime(time.time())
    filename.write(str("run time: {}/{}/{} {}:{}:{}".format(time_tuple[0],time_tuple[1],time_tuple[2],
                                                            time_tuple[3],time_tuple[4],time_tuple[5]))
                                                            + "\n")
    # N points on PES
    for cnt in range(len(atom_Mg)):
        start_time = time.perf_counter()
        bond_length = math.fabs(float(atom_Mg[cnt].split()[1]) - (ref_coord))
        # set molecular parameters
        atom = atom_Co2 + atom_Mg[cnt]
        driver = PySCFDriver(
                    atom=atom,
                    basis=basis,
                    charge=charge,
                    spin=spin,
                    unit=DistanceUnit.ANGSTROM)
        
        # set as_problem
        full_problem = driver.run()
        if (active_space == 4):
            as_transformer = ActiveSpaceTransformer(4, 4)
        elif (active_space == 6):
            as_transformer = ActiveSpaceTransformer(6, 6)
        elif (active_space == 8):
            as_transformer = ActiveSpaceTransformer(8, 8)
        else:
            print("Error, bad active para")
            sys.exit()
        as_problem = as_transformer.transform(full_problem)
        fermionic_op = as_problem.hamiltonian.second_q_op()
        
        # set mapping
        converter = QubitConverter(JordanWignerMapper())
        qubit_op = converter.convert(fermionic_op)
        # set ansatz
        uccsd = UCC(num_particles=as_problem.num_particles,
            num_spatial_orbitals= as_problem.num_spatial_orbitals,
            excitations='sd',
            qubit_converter=converter)
        
        estimator = Estimator()  # use exact, statevector Estimator
        # set initial point = 0
        initial_point = len(uccsd.operators) * [0]
        # use VQEUCCFactory
        vqe_solver = VQEUCCFactory(estimator=estimator,
                                    ansatz=uccsd,
                                    optimizer=optimizer,
                                    initial_point=initial_point)

        vqe_calc = GroundStateEigensolver(converter, vqe_solver)
        vqe_result = vqe_calc.solve(as_problem)
        vqe_value = vqe_result.total_energies[0]
        
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        filename.write('-----------------------------------------------------------------------------------\n')
        filename.write("points: " + str(cnt) + "\n") 
        filename.write("bond_length: " + str(('%.3f' % bond_length)) + "\n")
        filename.write(atom)
        filename.write("\n********************************************************\n")
        filename.write("full_ele_num_part: " + str(full_problem.num_particles) + "\n")
        filename.write("full_ele_num_mo: "+ str(full_problem.num_spatial_orbitals) + "\n")
        filename.write("********************************************************\n")
        filename.write("act_ele_num_part: " +  str(as_problem.num_particles) + "\n")
        filename.write("act_ele_num_mo: " + str(as_problem.num_spatial_orbitals) + "\n")
        filename.write("********************************************************\n")
        filename.write("ele_h_cons: " + str(as_problem.hamiltonian.constants) + "\n")
        filename.write("fermionic_H_op: " + str(len(fermionic_op)) + "\n")
        filename.write("qubit_H_op: " + str(len(qubit_op)) + "\n")
        filename.write("Qubit: " + str(qubit_op.num_qubits) + "\n")
        filename.write("********************************************************\n")
        filename.write("Theta: " + str(len(vqe_result.raw_result.optimal_parameters)) + "\n")
        filename.write("Cost function evals: " + str(vqe_result.raw_result.cost_function_evals) + "\n")
        filename.write("********************************************************\n")
        filename.write("Ref value: " + str(('%.6f' % ref_values[cnt])) + "\n")
        filename.write("VQE value: " + str(('%.6f' % vqe_value)) + "\n")
        filename.write("vqe_value - ref_value: " + str(('%.6f' % (vqe_value-ref_values[cnt]))) + "\n")
        filename.write("********************************************************\n")
        filename.write(str(vqe_result))
        filename.write("\n********************************************************\n")
        filename.write("Time: " + str(('%.1f' % run_time)) + " s\n")
        
        results[cnt][0] = bond_length
        results[cnt][1] = ref_values[cnt]
        results[cnt][2] = vqe_value
        results[cnt][3] = math.fabs(vqe_value - ref_values[cnt])
        results[cnt][4] = run_time

    np.savetxt(results_file, results, fmt="%.3f\t%.6f\t%.6f\t%.6f\t%.1f",
            delimiter=",", comments='#',
            header="length(A) ref_energy(Ha) vqe_energy(Ha) |Delta|(Ha) time(s)")

    end_total_time = time.perf_counter()
    run_total_time = end_total_time - start_total_time
    filename.write("\nProgram finished!\n")
    filename.write("--------------------------input paramters----------------------------\n")
    filename.write("basis: " + str(basis) + "\n")
    filename.write("charge: "+  str(charge) + "\n")
    filename.write("spin: "+  str(spin) + "\n")
    filename.write("active_space: " + str(active_space) + "," + str(active_space) + "\n")
    filename.write("opt: " + "SLSQP" + "\n")
    filename.write("maxiter: " + str(maxiter) + "\n")
    filename.write("results_file: "+ str(results_file)  + "\n")
    filename.write("log_file: "+ str(log_file)  + "\n")
    filename.write("Total time: " +   str(('%.1f' % run_total_time)) + " s\n")

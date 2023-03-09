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
prefix = "Origin_VQE_Mg_N2_active_"
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
atom_Mg = ['''Mg  0.00000000    0.00000000    0.34596500''',
           '''Mg  0.00000000    0.00000000    0.54596500''',
           '''Mg  0.00000000    0.00000000    0.74596500''',
           '''Mg  0.00000000    0.00000000    0.94596500''',
           '''Mg  0.00000000    0.00000000    1.14596500''',
           '''Mg  0.00000000    0.00000000    1.34596500''',
           '''Mg  0.00000000    0.00000000    1.54596500''',
           '''Mg  0.00000000    0.00000000    1.74596500''',
           '''Mg  0.00000000    0.00000000    1.94596500''',
           '''Mg  0.00000000    0.00000000    2.14596500''',
           '''Mg  0.00000000    0.00000000    2.34596500''',
           '''Mg  0.00000000    0.00000000    2.54596500''',
           '''Mg  0.00000000    0.00000000    2.94596500''',
           '''Mg  0.00000000    0.00000000    3.94596500''',
           ]

atom_N2 = '''\
N    0.00000000    0.00000000   -0.65403500
N    0.00000000    0.00000000   -1.77306600
'''
ref_coord = -0.65403500

MgN2_ref_values_4_4 = [-302.701494367086,
                        -303.542104491019,
                        -303.916084933413,
                        -304.040757014667,
                        -304.09590760989,
                        -304.128668755457,
                        -304.117973418922,
                        -304.082289061899,
                        -304.065628435442,
                        -304.051861113417,
                        -304.041355199302,
                        -304.033848141791,
                        -304.025196899043,
                        -304.017734928519,
                       ]

MgN2_ref_values_6_6 = [-302.752368166777,
                        -303.593970391132,
                        -303.968076510816,
                        -304.121443532331,
                        -304.17620348011,
                        -304.184848108677,
                        -304.16823692732,
                        -304.125082453373,
                        -304.069604306596,
                        -304.051961016065,
                        -304.041360519033,
                        -304.033848701684,
                        -304.025196909,
                        -304.01773492852,
                       ]

MgN2_ref_values_8_8 = [-302.772161759599,
                        -303.611338122793,
                        -303.985205057842,
                        -304.139313918564,
                        -304.195122352979,
                        -304.205639306076,
                        -304.193538216563,
                        -304.163282737799,
                        -304.11915194905,
                        -304.052404377919,
                        -304.041514227816,
                        -304.033895714716,
                        -304.025199699681,
                        -304.017734928819,
                       ]

# ------------------------Run Simulation----------------------------
start_total_time = time.perf_counter()
results = np.empty((len(atom_Mg), 5))

# set active space
if (active_space == 4):
    ref_values = MgN2_ref_values_4_4
elif (active_space == 6):
    ref_values = MgN2_ref_values_6_6
elif (active_space == 8):
    ref_values = MgN2_ref_values_8_8
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
        bond_length = math.fabs(float(atom_Mg[cnt].split()[3]) - (ref_coord))
        # set molecular parameters
        atom = atom_N2 + atom_Mg[cnt]
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

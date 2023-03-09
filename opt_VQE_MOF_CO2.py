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
prefix = "MOF_Co2_active_"
basis = "sto3g"
charge = 2
spin = 0
maxiter = 100
reduce_qubit = True
active_space = -1
shots = -1
use_shots = False
opt = "opt"

parser = argparse.ArgumentParser(description='Calculate ground-state energy by VQE;\n\
                                 1. if use shots, the optimizer is SLSQP/COBYLA, elif no use shots, the optimizer is SLSQP;\n\
                                 2. if use shots, must set --shots, elif no use shots, the --shots is invalid')

parser.add_argument("--use_shots", type=int, default=-1, help='-1/1, default -1')
parser.add_argument("--active", type=int, default=-1, help='active(n,n), n=4/6/8, no default')
parser.add_argument("--shots", type=int, default=-1, help='shots, int, no default')
parser.add_argument("--opt", type=str, default="opt", help='optimizer SLSQP/COBYLA, no default')
args = parser.parse_args()

if args.use_shots == -1:
    use_shots = False
elif args.use_shots == 1:
    use_shots = True
else:
    print("Error, bad use_shots para")
    sys.exit()
        
if not args.active == -1:
    active_space = args.active
if not args.shots == -1:
    shots = args.shots
if not args.opt == "opt":
    opt = args.opt
# set opt
if (use_shots):
    if (shots == -1):
        print("Error, please input suitable shots value!")
        sys.exit()
    if (opt == "SLSQP"):
        optimizer = SLSQP(maxiter=maxiter, eps=1e-2, ftol=1e-3)
    elif (opt == "COBYLA"):
        optimizer = COBYLA(maxiter=maxiter, tol=1e-3)
    else:
        print("Error, bad opt para")
        sys.exit()
else:
    if (opt == "SLSQP"):
        optimizer = SLSQP(maxiter=maxiter, eps=1e-4, ftol=1e-4)
    else:
        print("Error, bad opt para")
        sys.exit()
    
# set file
if (use_shots):
    results_file = prefix + str(active_space) + "_" + str(active_space) + \
                   "_shots_" + str(shots) + "_" + opt + ".csv"
    log_file = prefix + str(active_space) + "_" + str(active_space) + \
                   "_shots_" + str(shots) + "_" + opt + "_info.log"
else:
    results_file = prefix + str(active_space) + "_" + str(active_space) + \
                   "_" + opt + ".csv"
    log_file = prefix + str(active_space) + "_" + str(active_space) + \
                   "_" + opt + "_info.log"

# ------------------Coordiantes and Reference values------------------------
atom_O1 = ['''O                  0.88643217   -1.22352247    0.01370382''', #1.4
            '''O                  0.97937762   -1.40031168    0.00337337''',  #1.6
            '''O                  1.07232307   -1.57710089   -0.00695709''',   #1.8
            '''O                  1.16526852   -1.75389010   -0.01728754''',   #2.0
            '''O                  1.29404100   -1.99882500   -0.03160000''', #2.27
            '''O                  1.35115942   -2.10746852   -0.03794845''',   #2.4
            '''O                  1.44410487   -2.28425773   -0.04827810''',  #2.6
            '''O                  1.53705032   -2.46104694   -0.05860935''',   #2.8
            '''O                  1.62999577   -2.63783615   -0.06893980''',  #3.0
            '''O                  1.72294122   -2.81462536   -0.07927025''', #3.2    
            '''O                  2.09472304   -3.52178219   -0.12059209''',#4
            '''O                  2.55945030   -4.40572824   -0.17224436''',#5
            '''O                  3.02417756   -5.28967429   -0.22389663''',#6
            '''O                  3.48890482   -6.17362034   -0.27554890''',#7
            ]            

atom_C =  [
            '''C                  1.53750217   -2.21907347    0.11082282''',  #1.4
            '''C                  1.63044762   -2.39586268    0.10049237''',  #1.6
            '''C                  1.72339307   -2.57265189    0.09016191''', #1.8
            '''C                  1.81633852   -2.74944110    0.07983146''' , #2.0
            '''C                  1.94511100   -2.99437600    0.06551900 ''' , #2.27
            '''C                  2.00222942   -3.10301952    0.05917055''' ,  #2.4
            '''C                  2.09517487   -3.27980873    0.04884010''' ,#2.6
            '''C                  2.18812032   -3.45659794    0.03850965''',  #2.8
            '''C                  2.28106577   -3.63338715    0.02817920''',  #3.0
            '''C                  2.37401122   -3.81017636    0.01784875 ''',  #3.2
            '''C                  2.74579304   -4.51733319   -0.02347309''',#4
            '''C                  3.21052030   -5.40127924   -0.07512536  ''',#5
            '''C                  3.67524756   -6.28522529   -0.12677763  ''',#6
            '''C                  4.13997482   -7.16917134   -0.17842990 ''',#7
            ]

atom_O2 = [         
            '''O                  2.17455917   -3.19785247    0.20096182''',   #1.4
            '''O                  2.26750462   -3.37464168    0.19063137''',  #1.6
            '''O                  2.36045007   -3.55143089    0.18030091''', #1.8
            '''O                  2.45339552   -3.72822010    0.16997046''',  #2.0
            '''O                  2.58216800   -3.97315500    0.15565800''' , #2.27
            '''O                  2.63928642   -4.08179852    0.14930955''' , #2.4
            '''O                  2.73223187   -4.25858773    0.13897910''',   #2.6
            '''O                  2.82517732   -4.43537694    0.12864865''',  #2.8
            '''O                  2.91812277   -4.61216615    0.11831820''',   #3.0
            '''O                  3.01106822   -4.78895536    0.10798775''',  #3.2
            '''O                  3.38285004   -5.49611219    0.06666591''',#4
            '''O                  3.84757730   -6.38005824    0.01501364  ''',#5
            '''O                  4.31230456   -7.26400429   -0.03663863  ''',#6
            '''O                  4.77703182   -8.14795034   -0.08829090  ''',#7
            ]

atom_xx = '''\
Mg                 0.23581400    0.01400200    0.08601700
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
'''
len_atom_O1 = len(atom_O1)   
np_Mg = np.array([0.23581400, 0.01400200, 0.08601700])   
np_O1 = np.zeros((len_atom_O1, 3))

for i in range(len_atom_O1):   
    O1_coordinate = atom_O1[i].split()
    C_coordinate = atom_C[i].split()
    O2_coordinate = atom_O2[i].split()
    
    np_O1[i, 0] = float(O1_coordinate[1])
    np_O1[i, 1] = float(O1_coordinate[2])
    np_O1[i, 2] = float(O1_coordinate[3])

MOFCo2_ref_values_8_8 = [
                        -1282.36334336587 ,
                        -1282.48401366294,
                        -1282.52656094663  ,
                        -1282.53270753464 ,
                        -1282.52026564336,
                        -1282.51403530539,
                        -1282.50145185416,
                        -1282.49498815722 ,
                        -1282.49049930837,
                        -1282.48750128309 ,
                        -1282.482276489,
                        -1282.479945471,
                        -1282.478606962,
                        -1282.477888849,
                        ]

MOFCo2_ref_values_10_10 = [-1282.37150209594,
                        -1282.49314956303,
                        -1282.53425332444,
                        -1282.54806986345,
                        -1282.53655271405,
                        -1282.52972725183,
                        -1282.5240332444,
                        -1282.51751769672,
                        -1282.51008225867,
                        -1282.48955153679,
                        -1282.48384112756,
                        -1282.48140938392,
                        -1282.48012739907,
                        -1282.47934740009,
                        ]

# ------------------------Run Simulation----------------------------
start_total_time = time.perf_counter()
results = np.empty((len(atom_O1), 5))

# set active space
if (active_space == 8):
    ref_values = MOFCo2_ref_values_8_8
elif (active_space == 10):
    ref_values = MOFCo2_ref_values_10_10
else:
    print("Error, bad active para")
    sys.exit()
    
if (len(ref_values) != len(atom_O1)):
    print("Bad atom parameters!")
    sys.exit()

# use opt-VQE
def simplify_circuit(tmp_basis, tmp_charge, tmp_spin, tmp_active, reduce_qubit):
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
                        O                  2.36045007   -3.55143089    0.18030091''', # 1.8 angstrom
                basis=tmp_basis,
                charge=tmp_charge,
                spin=tmp_spin,
                unit=DistanceUnit.ANGSTROM)
    tmp_full_problem = driver.run()
    if (tmp_active == 8):
        tmp_as_transformer = ActiveSpaceTransformer(8, 8)
    elif (tmp_active == 10):
        tmp_as_transformer = ActiveSpaceTransformer(10, 10)
    else:
        print("Error, bad active para")
        sys.exit()
    # use active space
    tmp_as_problem = tmp_as_transformer.transform(tmp_full_problem)
    tmp_fermionic_op = tmp_as_problem.hamiltonian.second_q_op()
    # mapping
    if (reduce_qubit):
        tmp_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        tmp_op = tmp_converter.convert(tmp_fermionic_op, num_particles=tmp_as_problem.num_particles)
    else:
        tmp_converter = QubitConverter(JordanWignerMapper())
        tmp_op = tmp_converter.convert(tmp_fermionic_op)
    
    # use UCC ansatz
    uccsd = UCC(num_particles=tmp_as_problem.num_particles,
            num_spatial_orbitals= tmp_as_problem.num_spatial_orbitals,
            excitations='sd',
            qubit_converter=tmp_converter)
    # build excitation operator pool
    uccsd_op = uccsd.operators
    len_uccsd_op = len(uccsd_op)

    uccsd_cuicuit_items = 0 
    for i in range(len_uccsd_op):
        uccsd_cuicuit_items += len(uccsd_op[i])
    # select important excitation items in UCCSD
    test = []
    for i in range(len_uccsd_op):
        if (tmp_active == 8):  # 29 parameters
            if (i == 272 or i == 206 or i == 284 or i == 84 or
                i == 268 or i == 205 or i == 145 or i == 201 or
                i == 148 or i == 198 or i == 114 or i == 342 or
                i == 88 or i == 160 or i == 220 or i == 68 or
                i == 141 or i == 65 or i == 83 or i == 132 or
                i == 264 or i == 149 or i == 216 or i == 39 or
                i == 331 or i == 94 or i == 262 or i == 194 or
                i == 71):
                test.append(uccsd_op[i])
        else:
            print("Error, bad active para")
            sys.exit()
    # reduce evolved Pauli operators
    test_01 = []
    for i in range(len(test)):
        test_01.append(str(test[i][0]).split(" * "))

    coeffs = []
    for i in range(len(test_01)):
        coeffs.append(float(test_01[i][0]))
    # transform the pauli string to pauli tensor product
    simplified_operators = []
    for i in range(len(coeffs)):
        if (len(test_01[i]) == 0):
            sys.exit()
        cnt  = 0
        for op in test_01[i][1]:
            if (cnt == 0):     
                if (op == 'X'):
                    tmp = coeffs[i] * X
                elif (op == 'Y'):
                    tmp = coeffs[i] * Y
                elif (op == 'Z'):
                    tmp = coeffs[i] * Z
                elif (op == 'I'):
                    tmp = coeffs[i] * I
                else:
                    sys.exit()
            else:
                if (op == 'X'):
                    tmp = tmp ^ X
                elif (op == 'Y'):
                    tmp = tmp ^ Y
                elif (op == 'Z'):
                    tmp = tmp ^ Z
                elif (op == 'I'):
                    tmp = tmp ^ I
                else:
                    sys.exit()
            cnt += 1
            
        simplified_operators.append(tmp)
    return [simplified_operators, uccsd_cuicuit_items]

with open(file=log_file, mode='w') as filename :
    time_tuple = time.localtime(time.time())
    filename.write(str("run time: {}/{}/{} {}:{}:{}".format(time_tuple[0],time_tuple[1],time_tuple[2],
                                                            time_tuple[3],time_tuple[4],time_tuple[5]))
                                                            + "\n")
    # N points on PES
    for cnt in range(len_atom_O1):
        start_time = time.perf_counter()
        bond_length = np.linalg.norm(np_O1[cnt, :] - np_Mg)
        # set molecular parameters
        atom = atom_xx + atom_O1[cnt] +"\n" + atom_C[cnt] + "\n" + atom_O2[cnt]
        driver = PySCFDriver(
                    atom=atom,
                    basis=basis,
                    charge=charge,
                    spin=spin,
                    unit=DistanceUnit.ANGSTROM)
        
        # set as_problem
        full_problem = driver.run()
        if (active_space == 8):
            as_transformer = ActiveSpaceTransformer(8, 8)
        else:
            print("Error, bad active para")
            sys.exit()
        as_problem = as_transformer.transform(full_problem)
        fermionic_op = as_problem.hamiltonian.second_q_op()
        
        # set mapping
        if (reduce_qubit):
            converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
            qubit_op = converter.convert(fermionic_op, num_particles=as_problem.num_particles)
        else:
            converter = QubitConverter(JordanWignerMapper())
            qubit_op = converter.convert(fermionic_op)

        simplified_evo_operators = simplify_circuit(basis, charge, spin, active_space, reduce_qubit)
        simplified_anastz = EvolvedOperatorAnsatz(simplified_evo_operators[0])
        if (use_shots):
            estimator = AerEstimator(run_options={"shots": shots}) # use based-shots AerEstimator
        else:
            estimator = Estimator()  # use exact, statevector Estimator
        # set initial point = 0
        initial_point = len(simplified_anastz.operators) * [0]
        # use VQEUCCFactory
        vqe_solver = VQEUCCFactory(estimator=estimator,
                                    ansatz=simplified_anastz,
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
        filename.write("uccsd circuit items: " + str(simplified_evo_operators[1]) + "\n")
        filename.write("simplified circuit items: " + str(len(simplified_evo_operators[0])) + "\n")
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
    filename.write("reduce_qubit: " + str(reduce_qubit) + "\n")
    filename.write("opt: " + str(opt) + "\n")
    filename.write("maxiter: " + str(maxiter) + "\n")
    filename.write("use_shots: " + str(use_shots) + "\n")
    if (use_shots):
        filename.write("shots: " + str(shots) + "\n")
    filename.write("results_file: "+ str(results_file)  + "\n")
    filename.write("log_file: "+ str(log_file)  + "\n")
    filename.write("Total time: " +   str(('%.1f' % run_total_time)) + " s\n")

import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf, scf
starttime = datetime.datetime.now()
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

for i in range(len(atom_Mg)):
    print('\n')
    print("point = ", i)
    atom = atom_N2 + atom_Mg[i]
    print(atom)
    print("###############################################################################")
    mol = gto.M(atom=atom,
            #verbose=4,
	    #symmetry=True,
            basis='sto-3g',
            charge=2,
            spin=0)  # 定义原子各种属性
    myhf = mol.HF().run() 
    mycas = mcscf.CASCI(myhf, 4, 4)
    #mycas.kernel()
    e_casci = mycas.kernel()[0]

endtime = datetime.datetime.now()
seconds = (endtime - starttime).seconds
print("Time:", seconds)

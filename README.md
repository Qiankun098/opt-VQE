The repository is used to finish the Deloitte’s Quantum Climate Challenge 2023.
The opt-VQE algorithm is the major content, which includes the optimization: reducing qubits, variable parameters, circuit depth and Hamiltonian items.
The repository is writen based on python by Qiskit software.

To run opt-VQE, for example Mg_CO2 system
1. python Energy_sort_Mg_CO2.py (set active = 4 or 6 or 8)
2. python opt_VQE_Mg_CO2.py --active 4/6/8 --opt SLSQP (use statevector simulator)
3. python opt_VQE_Mg_CO2.py --use_shots 1 --shots 8000 --active 4/6/8 --opt SLSQP/COBYLA (use shot-based simulator)

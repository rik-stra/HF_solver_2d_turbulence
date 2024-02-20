# Input files
The solver takes a .json file as argument. This input file should be used to specify the input variables. The input files are structured as 2 dictionaries. When a flag in dict 2 is not specified it defaults to the green value. All flags in Dict 1 must be specified.

Example input files are given in this folder.

## input HF_solver.py
* Header

Dict 1:
 * **"N_HF":257,**  number of resolved modes in high fidelity simulation (must be odd).
 * **"remove_alias":{true/false},**  if true, aliasing errors are removed using the 2/3 rule. This means that Fourier transforms are taken over N_HF*3/2 modes.
 * **"dt_HF":0.001** , time step high fidelity model in days.
 * **"t_start":300,** first day of the simulation (only relevant when restarting).
 * **"simulation_time":2000**, give length of simulation in days.

Dict 2:
 * **"sim_ID": "training_data"**, will be used to store samples in output directory.
 * **"store_final_state": {true/<span style="color:green">false</span>},** choose to store the state at the end of the simulation for a possible restart (continuation) from this point.
 * **"restart": {true/<span style="color:green">false</span>}**, if true, the below provided restart file is used for the initial conditions of the simulation.
 * **"restart_file_name": "./restart/..",**
 * **"plot_frame_rate":{<span style="color:green">1000</span>}**, store the solution field every $x$-th time step. If plot_frame_rate = 1000 and dt_HF=0.001 once per day a solution is stored.
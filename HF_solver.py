"""
========================================================================
python script:
Script to compute reference data with a high fidelity simulation.
R. Hoekstra

========================================================================
"""

from functions_for_solver import *
from plot_store_functions import store_samples_hdf5, store_state
from filters import Filters, Grids
from initial_conditions import get_initial_conditions

###########################
# M A I N   P R O G R A M #
###########################

import numpy as np
import torch
import os
import h5py
import sys
import json
import time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HOME = os.path.abspath(os.path.dirname(__file__))

if os.path.exists(HOME + '/output') == False:
        os.makedirs(HOME + '/output')

#read flags from input file
fpath = sys.argv[1]
#fpath = HOME+"/inputs/input_ref.json"
fp = open(fpath, 'r')

#print the desription of the input file
print(fp.readline())
varia = json.loads(fp.readline())

################################ set up grid and projection operators ################################################
remove_alias = varia["remove_alias"]
N = varia["N_HF"]

print("GRID:")
print("reference: N=", N)
if remove_alias:
    N_a = (N-1)*3//2+1
    
else:
    N_a=N; 

grid = Grids(N_a, N_HF_resolved=N)
filters = Filters(grid, remove_alias)
######################################################################################################################


##############  Time integrations settings  ##############
#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega


#start, end time, time step
dt = varia["dt_HF"]*day
t = varia["t_start"]*day
t_end = t + varia["simulation_time"]*day
n_steps = int(np.round((t_end-t)/dt))
print("TIME SETTINGS: (in days)")
print("T start:", varia["t_start"], "T end:", varia["t_start"]+varia["simulation_time"], "dt:", varia["dt_HF"], "(",dt,")")




#########################
# Read simulation flags #
#########################
##### to prevent warnings define all possible flags  ##########################
sim_ID = -99; 
restart = False; 
store_final_state=False; 
restart_file_name = "none";
method = "AB/BDI2" ;
plot_frame_rate = 1000;
###############################################################################
##### read from json file  ####################################################
flags_line = fp.readline()[:-1]
while flags_line[-1] != '}':
    flags_line = flags_line+fp.readline()[:-1]
flags = json.loads(flags_line)
print('*********************')
print('Simulation flags')
for key in flags.keys():
    vars()[key] = flags[key]
    print(key, '=', flags[key])

fp.close()
print('*********************')
#####################################################################################

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*(256//3)**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

if method == "AB/BDI2":
    #constant factor that appears in AB/BDI2 time stepping scheme   
    norm_factor = 1.0/(3.0/(2.0*dt) - nu*grid.k_squared_HF + mu)        #for reference solution
    
elif method == "AB/CN":
    #constant factor that appears in AB/CN time stepping scheme   
    norm_factor = 1.0/(2.0/dt - nu*grid.k_squared_HF + mu)        #for reference solution
    
else:
    print("integration scheme not implemented: ", method)
    sys.exit()

#framerate of storing data, plotting results (1 = every integration time step)
S_fields = np.floor(n_steps/plot_frame_rate).astype('int')+2

    
###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#TRAINING DATA SET

QoI = ['w_hat_n_HF'] 

Q = len(QoI)

#allocate memory
samples = {}

QoI_fields = []
QoI_scalars = []

for q in range(Q):
    #assume a field contains the string '_hat_'
    if 'w_hat_n_HF' in QoI[q]:
        QoI_fields.append(QoI[q])
        samples[QoI[q]] = np.zeros([S_fields, N, N]) + 0.0j



#######  forcing term   ########################
F_HF = 2**1.5*np.cos(5*grid.x_HF)*np.cos(5*grid.y_HF)
#F_LF = 0*x_LF;
F_hat_HF = np.fft.fft2(F_HF)


##################################################


#########  inital conditions ######################
w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, psi_hat_n_HF, psi_hat_nm1_HF = get_initial_conditions(restart,filters, grid,
                                                                                                    restart_file_name, USE_HDF5=True, filter_restart_state=False,transpose=False)

####################################################

######### put everything on the GPU if it is there ######
# constants
#dt = torch.Tensor([dt]).to(device=device)
#mu = torch.Tensor([mu]).to(device=device)
# fields
w_hat_n_HF = torch.from_numpy(w_hat_n_HF).to(device=device)
psi_hat_n_HF= torch.from_numpy(psi_hat_n_HF).to(device=device)
w_hat_nm1_HF= torch.from_numpy(w_hat_nm1_HF).to(device=device)
VgradW_hat_nm1_HF= torch.from_numpy(VgradW_hat_nm1_HF).to(device=device)
F_hat_HF= torch.from_numpy(F_hat_HF).to(device=device)
norm_factor = torch.from_numpy(norm_factor).to(device=device)


# objects
filters.P_HF = torch.from_numpy(filters.P_HF).to(device=device)

grid.k_x_HF = torch.from_numpy(grid.k_x_HF).to(device=device)
grid.k_y_HF = torch.from_numpy(grid.k_y_HF).to(device=device)
grid.k_squared_nonzero_HF = torch.from_numpy(grid.k_squared_nonzero_HF).to(device=device)
grid.k_squared_HF = torch.from_numpy(grid.k_squared_HF).to(device=device)




print('*********************')
print('Solving forced dissipative vorticity equations on '+ str(device))
print('Ref grid = ', N, 'x', N)
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')
print('*********************')

t0 = time.time()


############ time loop  ########################
#some counters
j2=plot_frame_rate
j3=0

for n in range(n_steps):
    
    if np.mod(n, int(day/dt)) == 0:
        print(f'day =  {n//int(day/dt)} of  {n_steps//int(day/dt)}')
    

    ########run the HF model#####################################################################    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(method,dt,mu,nu,w_hat_n_HF, psi_hat_n_HF, w_hat_nm1_HF, 
                                                      VgradW_hat_nm1_HF, filters.P_HF, 
                                                      norm_factor,
                                                      grid.k_x_HF, grid.k_y_HF, grid.k_squared_HF,
                                                      F_hat_HF)
    psi_hat_np1_HF = get_psi_hat(w_hat_np1_HF, grid.k_squared_nonzero_HF)
            
    

    for qoi in QoI_scalars:
        samples[qoi][n] = eval(qoi)
    # store fields (once every plot_frame_rate)
    if j2 == plot_frame_rate:
        j2 = 0
        samples['w_hat_n_HF'][j3] = filters.filter_HF2resolved(w_hat_n_HF).cpu()
        j3 += 1
    
    
    t += dt
    j2 += 1
      
    #update variables
    
    w_hat_nm1_HF = torch.clone(w_hat_n_HF)
    w_hat_n_HF = torch.clone(w_hat_np1_HF)
    VgradW_hat_nm1_HF = torch.clone(VgradW_hat_n_HF)
    psi_hat_n_HF = torch.clone(psi_hat_np1_HF)
    
    if n==n_steps-10:
        w_hat_nm10_HF=filters.filter_HF2resolved(w_hat_nm1_HF).cpu().numpy(force=True)
        VgradW_hat_nm10_HF = filters.filter_HF2resolved(VgradW_hat_nm1_HF).cpu().numpy(force=True)
    
    ## check for nans ###########
    if torch.isnan(torch.sum(w_hat_n_HF)):
        break
# end of time loop
    
t1 = time.time()
print('Simulation time =', t1 - t0, 'seconds')
#store last samples
#calculate the QoI after applying the SGS term (every time step)

for qoi in QoI_scalars:
    samples[qoi][n] = eval(qoi)
# store fields (once every plot_frame_rate)
samples['w_hat_n_HF'][j3] = filters.filter_HF2resolved(w_hat_n_HF).cpu()
j3 += 1


    
####################################

#store the state of the system to allow for a simulation restart at t > 0
if store_final_state == True:
    restart_dic = {'w_hat_nm1_HF': filters.filter_HF2resolved(w_hat_nm1_HF).cpu().numpy(force=True),
                    'w_hat_n_HF': filters.filter_HF2resolved(w_hat_n_HF).cpu().numpy(force=True), 
                    'VgradW_hat_nm1_HF': filters.filter_HF2resolved(VgradW_hat_nm1_HF).cpu().numpy(force=True),
                    'w_hat_nm10_HF':w_hat_nm10_HF, 'VgradW_hat_nm10_HF':VgradW_hat_nm10_HF}
    store_state(restart_dic, HOME, sim_ID, t/day)


#store the samples

store_samples_hdf5(HOME,sim_ID,t/day, QoI, samples)
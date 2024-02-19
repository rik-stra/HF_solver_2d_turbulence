import numpy as np
import os
import h5py

def store_state(dict, HOME, sim_ID, t_in_days, USE_HDF5=True):
    if USE_HDF5:
        store_state_hdf5(dict, HOME, sim_ID, t_in_days)
    else:
        store_state_pickle(dict, HOME, sim_ID, t_in_days)

def store_state_hdf5(dict, HOME, sim_ID, t_in_days):
    import h5py
    if os.path.exists(HOME + '/output/restart') == False:
        os.makedirs(HOME + '/output/restart')
    
    fname = HOME + '/output/restart/' + sim_ID + '_t_' + str(np.around(t_in_days,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in dict:
        h5f.create_dataset(key, data = dict[key])
        
    h5f.close()   

def store_state_pickle(dict, HOME, sim_ID, t_in_days):
    import pickle
    if os.path.exists(HOME + '/output/restart') == False:
        os.makedirs(HOME + '/output/restart')
    
    fname = HOME + '/output/restart/' + sim_ID + '_t_' + str(np.around(t_in_days,1)) + '.dat'
    
    #create HDF5 file
    with open(fname,'wb') as file:
        pickle.dump(dict,file, protocol=pickle.HIGHEST_PROTOCOL)
    

def store_samples(HOME,sim_ID,t_by_day, QoI, samples, USE_HDF5):
    if USE_HDF5:
        store_samples_hdf5(HOME,sim_ID,t_by_day, QoI, samples)
    else:
        store_samples_pickle(HOME,sim_ID,t_by_day, QoI, samples)

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5(HOME,sim_ID,t_by_day, QoI, samples):
    
    fname = HOME + '/output/samples/' + sim_ID + '_t_' + str(np.around(t_by_day, 1)) + '.hdf5'
    
    print('Storing samples in ', fname)
    
    if os.path.exists(HOME + '/output/samples') == False:
        os.makedirs(HOME + '/output/samples')
    
    #create HDF5 file
    h5f_store = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f_store.create_dataset(q, data = samples[q])
        
    h5f_store.close()



def store_samples_pickle(HOME,sim_ID,t_by_day, QoI, samples):
    import pickle
    fname = HOME + '/output/samples/' + sim_ID + '_t_' + str(np.around(t_by_day, 1)) + '.hdf5'
    
    print('Storing samples in ', fname)
    
    if os.path.exists(HOME + '/output/samples') == False:
        os.makedirs(HOME + '/output/samples')

    dic={}
    for q in QoI:
        dic[q] = samples[q]
    #create HDF5 file
    with open(fname, 'wb') as file:
        pickle.dump(dic,file, protocol=pickle.HIGHEST_PROTOCOL)

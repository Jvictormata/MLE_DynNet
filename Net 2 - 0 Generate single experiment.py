
from network2 import *
from scipy.io import savemat


from jax import config
config.update("jax_enable_x64", True)

########## System parameters

n_ab = np.array([[2,2,2],[2,2,2]]) #n_ab = [[na1,na2,na3],[nb1,nb2,nb3]]
M = n_ab.shape[1] #number of systems
N = 500 #number of samples per experiment;
sigma = 0.1  #noise standard deviation



### Systems:
sys1 = signal.zpk2tf([],[-1, -1], 2)
sys2 = signal.zpk2tf([-2],[-0.5, -1.5], 1)
sys3 = signal.zpk2tf([],[-0.5, -0.5], 0.5)


sys1 = signal.cont2discrete(sys1,1)
sys2 = signal.cont2discrete(sys2,1)
sys3 = signal.cont2discrete(sys3,1)


### Estimtion data:

e_single_exp = sigma*np.random.randn(N,M)
r_single_exp = np.random.choice([-1, 1], size=(N,M), p=[1./2, 1./2])
x_single_exp = gen_states(sys1,sys2,sys3,r_single_exp,e_single_exp)

np.save("exp_2/e_single_exp2",np.array(e_single_exp))
np.save("exp_2/r_single_exp2",np.array(r_single_exp))
np.save("exp_2/x_single_exp2",np.array(x_single_exp))


savemat('exp_2/x_2.mat', {"x":x_single_exp})
savemat('exp_2/r_2.mat', {"r":r_single_exp})
savemat('exp_2/e_2.mat', {"e":e_single_exp})


### Validation data:

e_validation = sigma*np.random.randn(N,M)
r_validation = np.random.choice([-1, 1], size=(N,M), p=[1./2, 1./2])
x_validation = gen_states(sys1,sys2,sys3,r_validation,e_validation)


np.save("exp_2/e_validation2",np.array(e_validation))
np.save("exp_2/r_validation2",np.array(r_validation))
np.save("exp_2/x_validation2",np.array(x_validation))


savemat('exp_2/x_validation2.mat', {"x_val":x_validation})
savemat('exp_2/r_validation2.mat', {"r_val":r_validation})
savemat('exp_2/e_validation2.mat', {"e_val":e_validation})



initial_thetas = []
while len(initial_thetas)<100:
    new_theta = 2*np.random.rand(12)-1
    if check_stability(new_theta,n_ab,dt=1):
        initial_thetas.append(new_theta)
    print("\n")
    print("total of stable initial points:", len(initial_thetas))

initial_thetas = np.array(initial_thetas)
np.save("exp_2/initial_thetas2",initial_thetas)


print("Data generated!")

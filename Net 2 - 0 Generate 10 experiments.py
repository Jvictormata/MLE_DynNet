
from network2 import *


from jax import config
config.update("jax_enable_x64", True)

########## System parameters

n_ab = np.array([[2,2,2],[2,2,2]]) #n_ab = [[na1,na2,na3],[nb1,nb2,nb3]]
M = n_ab.shape[1] #number of systems
N = 50 #number of samples per experiment;
n_exp = 10 #number of experiments
sigma = 0.1  #noise standard deviation



### Systems:
sys1 = signal.zpk2tf([],[-1, -1], 2)
sys2 = signal.zpk2tf([-2],[-0.5, -1.5], 1)
sys3 = signal.zpk2tf([],[-0.5, -0.5], 0.5)


sys1 = signal.cont2discrete(sys1,1)
sys2 = signal.cont2discrete(sys2,1)
sys3 = signal.cont2discrete(sys3,1)


e_10_exp = []
r_10_exp = []
x_10_exp = []

for i in range(n_exp):
    e = sigma*np.random.randn(N,M)
    r = np.random.choice([-1, 1], size=(N,M), p=[1./2, 1./2])
    states = gen_states(sys1,sys2,sys3,r,e)

    e_10_exp.append(e)
    r_10_exp.append(r)
    x_10_exp.append(states)


np.save("exp_2/e_10_exp2",np.array(e_10_exp))
np.save("exp_2/r_10_exp2",np.array(r_10_exp))
np.save("exp_2/x_10_exp2",np.array(x_10_exp))
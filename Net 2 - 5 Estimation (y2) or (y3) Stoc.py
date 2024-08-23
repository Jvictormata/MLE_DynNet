from network2 import *

from jax import config
config.update("jax_enable_x64", True)


########## System parameters

n_ab = np.array([[2,2,2],[2,2,2]]) #n_ab = [[na1,na2,na3],[nb1,nb2,nb3]]

M = n_ab.shape[1] #number of systems
N = 50 #number of samples;
n_exp = 10 #number of experiments
sigma = 0.1  #noise standard deviation

lam = 0.1


#Interconections
Lamb = jnp.array([[0,0,0],[1,0,1],[1,0,0]])
Delta = jnp.eye(3)


########## Loading data:

e_10 = np.load("exp_2/e_10_exp2.npy")
r_10 = np.load("exp_2/r_10_exp2.npy")
x_10 = np.load("exp_2/x_10_exp2.npy")

#initial_thetas = np.load("exp_1/initial_thetas.npy")
initial_thetas = np.load("exp_2/initial_thetas2.npy")



########## Optimizing:


def load_data(i,Lamb,Delta,M,N):

    if i >= 10 :
        j = i%10
    else:
        j = i

    if i%2 == 0:
        n_obs = 1
        obs = [2]
    else:
        n_obs = 1
        obs = [3]


    r = r_10[j,:,:]
    x = x_10[j,:,:]
    r_sig = jnp.concatenate([r[:,0],r[:,1],r[:,2]]).reshape(-1,1)

    xo,_,Permut = get_xoxm(x,obs,N,M)

    A2,B2 = gen_A2B2(Lamb,Delta,Permut,M,N,r_sig)
    A2o = A2[:,:n_obs*N]
    A2m = A2[:,n_obs*N:]
    To_phi, Tm_phi, T_gamma, W2, V2 = get_transform_matrices(A2o,A2m,B2)

    return To_phi, Tm_phi, T_gamma, W2, xo.reshape(-1,1), Permut




def optimize_stoc(theta, n_ab, N, M, n_iter):
    max_eta = 1
    
    for i in range(n_iter):
        To_phi, Tm_phi, T_gamma, W2, xo, Permut = load_data(i,Lamb,Delta,M,N)

        g_theta = grad_theta(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo)
        d_theta = g_theta

        current_cost = eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)

        if i == 0:
            print("Initial cost: ",current_cost)
            print("\n\n\n")    
        
        eta,new_cost = bcktrck_linsearch(theta, n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2,xo, max_eta, g_theta, d_theta, current_cost)    

        theta -= eta*d_theta

        print("Iteration: ",i+1,"  -  Cost function value: ",new_cost)
        print("\n")
        if eta<0.0000001 or new_cost>1000000 or np.isnan(new_cost):
            new_cost = eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
            return theta,new_cost
        
        if (d_theta.T@g_theta)<0.000001:
            new_cost = eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
            return theta,new_cost
        
    new_cost = eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
    return theta,new_cost



theta = []
costs = []

for i in range(100):
    print("\n\n\n")
    print("Initial Theta: ",i+1)
    current_theta,cost_temp = optimize_stoc(initial_thetas[i,:].reshape(-1,1), n_ab, N, M, 150)
    theta.append(current_theta)
    costs.append(cost_temp)
    np.save("exp_2/theta_opt2_y2_or_y3_stoc_GD_Newton",np.array(theta))
    np.save("exp_2/costs_opt2_y2_or_y3_stoc_GD_Newton",np.array(costs))

np.save("exp_2/theta_opt2_y2_or_y3_stoc_GD_Newton",np.array(theta))
np.save("exp_2/costs_opt2_y2_or_y3_stoc_GD_Newton",np.array(costs))
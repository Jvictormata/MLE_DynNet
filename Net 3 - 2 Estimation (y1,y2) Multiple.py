from network3 import *
import time

from jax import config
config.update("jax_enable_x64", True)


########## System parameters

n_ab = np.array([[2,2,3],[2,2,3]]) #n_ab = [[na1,na2,na3],[nb1,nb2,nb3]]

M = n_ab.shape[1] #number of systems
N = 50 #number of samples;
n_exp = 10 #number of experiments
sigma = 0.1  #noise standard deviation

lam = 0.1


#Interconections
Lamb = jnp.array([[0,1,0],[1,0,1],[0,1,0]])
Delta = jnp.eye(3)


########## Loading data:

e_10 = np.load("exp_3/e_10_exp3.npy")
r_10 = np.load("exp_3/r_10_exp3.npy")
x_10 = np.load("exp_3/x_10_exp3.npy")

initial_thetas = np.load("exp_3/initial_thetas3.npy")



########## Optimizing:


def load_all_data(Lamb,Delta,M,N):
    n_obs = 2
    n_exp = 10
    obs = [1,2]

    for i in range(n_exp):
        if i >= 10 :
            j = i%10
        else:
            j = i

        if i%2 == 0:
            obs = [1,2]
        else:
            obs = [1,2]

        if i == 0:
            r = r_10[j,:,:]
            x = x_10[j,:,:]
            r_sig = jnp.concatenate([r[:,0],r[:,1],r[:,2]]).reshape(-1,1)
            obs_def = obs
        else:
            r = r_10[j,:,:]
            x = jnp.block([[x],[x_10[j,:,:]]])
            r_sig = jnp.block([[r_sig],[jnp.concatenate([r[:,0],r[:,1],r[:,2]]).reshape(-1,1)]])
            obs_def = jnp.concatenate([jnp.array(obs_def),jnp.array(obs)+i*M*2])
            
   
    xo,_,Permut = get_xoxm_multiple_exp(x,obs_def,N,M,n_exp)

    A2,B2 = gen_A2B2_multiple_exp(n_exp,Lamb,Delta,Permut,M,N,r_sig)
    A2o = A2[:,:n_obs*n_exp*N]
    A2m = A2[:,n_obs*n_exp*N:]
    To_phi, Tm_phi, T_gamma, W2, V2 = get_transform_matrices(A2o,A2m,B2)
    return To_phi, Tm_phi, T_gamma, W2, xo.reshape(-1,1), Permut




def optimize_multiple(theta, n_ab, N, M, n_iter):
    max_eta = 1

    for i in range(n_iter):
        
        g_theta = grad_theta_multiple_exp(theta,n_ab, Permut, N, M,n_exp, To_phi, Tm_phi, T_gamma, W2, xo)

        if i<=100:
            h_comput = 0
            d_theta = g_theta
        else:
            H_theta = Hessian_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])
            try:
                L_theta = np.linalg.cholesky(H_theta)
                d_temp = jnp.linalg.solve(L_theta,g_theta)
                d_theta = jnp.linalg.solve(L_theta.T,d_temp)
                h_comput = 1
                print("PSD hessian!")
            except:
                h_comput = 0
                d_theta = g_theta
                print("Comming back to GD!")


        current_cost = eval_cost_func_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo)

        if i == 0:
            print("Initial cost: ",current_cost)
            print("\n\n\n")    
        
        eta,new_cost = bcktrck_linsearch_multiple_exp(theta, n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2,xo, max_eta, g_theta, d_theta, current_cost)    

        theta -= eta*d_theta

        print("Iteration: ",i+1,"  -  Cost function value: ",new_cost)
        print("\n")
        if eta<0.0000001 or new_cost>1000000 or np.isnan(new_cost):
            new_cost = eval_cost_func_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo)
            return theta,new_cost,2
        
        if h_comput:
            if (d_theta.T@g_theta)<0.000001:
                new_cost = eval_cost_func_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo)
                return theta,new_cost,1
            
    new_cost = eval_cost_func_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo)
    return theta,new_cost,0




theta = []
costs = []
accuracy = []

To_phi, Tm_phi, T_gamma, W2, xo, Permut = load_all_data(Lamb,Delta,M,N)

initial_time = time.time()

for i in range(100):
    print("\n\n\n")
    print("Initial Theta: ",i+1)
    current_theta,cost_temp,acc = optimize_multiple(initial_thetas[i,:].reshape(-1,1), n_ab, N, M, 150)
    theta.append(current_theta)
    costs.append(cost_temp)
    accuracy.append(acc)
    np.save("exp_3/theta_opt3_y1y2_multiple_GD_Newton",np.array(theta))
    np.save("exp_3/costs_opt3_y1y2_multiple_GD_Newton",np.array(costs))
    np.save("exp_3/accuracy3_y1y2_multiple_GD_Newton",np.array(accuracy))

total_time = initial_time-time.time()
print("Total time:")
print(total_time)
print("========")

np.save("exp_3/theta_opt3_y1y2_multiple_GD_Newton",np.array(theta))
np.save("exp_3/costs_opt3_y1y2_multiple_GD_Newton",np.array(costs))
np.save("exp_3/accuracy3_y1y2_multiple_GD_Newton",np.array(accuracy))
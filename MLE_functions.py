import numpy as np
from scipy import signal


from jax import grad,jacfwd
import jax.numpy as jnp
from jax.scipy import linalg

from jax import config
config.update("jax_enable_x64", True)



def series_sys(sys1,sys2):
    num1 = sys1[0].reshape(-1,1)
    den1 = sys1[1].reshape(-1,1)
    num2 = sys2[0].reshape(-1,1)
    den2 = sys2[1].reshape(-1,1)
    
    num = signal.convolve(num1, num2)
    den = signal.convolve(den1, den2)
    dt = sys1[2]
    return num.reshape(-1),den.reshape(-1),dt



def feedback_sys(sys1,sys2):
    num1 = sys1[0].reshape(-1,1)
    den1 = sys1[1].reshape(-1,1)
    num2 = sys2[0].reshape(-1,1)
    den2 = sys2[1].reshape(-1,1)
    
    num = signal.convolve(num1, den2)
    den = signal.convolve(den1, den2) - signal.convolve(num1, num2)
    dt = sys2[2]
    return num.reshape(-1),den.reshape(-1),dt


def add_sys(sys1,sys2):
    num1 = sys1[0].reshape(-1,1)
    den1 = sys1[1].reshape(-1,1)
    num2 = sys2[0].reshape(-1,1)
    den2 = sys2[1].reshape(-1,1)

    num = np.polyadd(signal.convolve(num1, den2), signal.convolve(num2, den1))
    den = signal.convolve(den1, den2)
    dt = sys1[2]
    return num.reshape(-1),den.reshape(-1),dt


def permutation_mat(p):
    size_P = len(p)
    temp = np.eye(size_P)
    P = temp[:,p[0]-1].reshape(-1,1)
    for index in p[1:]:
        P = np.block([P,temp[:,index-1].reshape(-1,1)])
    return P




def gen_Ty(a,N):
    n_zeros = N-len(a)-1
    col = jnp.append(1,a)
    col = jnp.append(col,np.linspace(0,0,n_zeros))
    Ty = linalg.toeplitz(col,r=np.zeros(N))
    return Ty


def gen_Tu(b,N,dif_ab):
    n_zeros = N-len(b)-dif_ab
    col = jnp.append(np.linspace(0,0,dif_ab),b)
    col = jnp.append(col,np.linspace(0,0,n_zeros))
    Tu = linalg.toeplitz(col,r=np.zeros(N))
    return Tu


def gen_A1(theta,n_ab,Permut,N):
    pos = 0
    for i in range(len(n_ab[0])):
        na = n_ab[0][i]
        nb = n_ab[1][i]
        if pos==0:
            blk1 = gen_Ty(theta[pos:pos+na],N)
            blk2 = gen_Tu(theta[pos+na:pos+na+nb],N,na+1-nb)
        else:
            blk1 = linalg.block_diag(blk1,gen_Ty(theta[pos:pos+na],N))
            blk2 = linalg.block_diag(blk2,gen_Tu(theta[pos+na:pos+na+nb],N,na+1-nb))
        pos += (na+nb)
    return jnp.block([blk1,-blk2])@jnp.kron(Permut,jnp.eye(N))


def gen_A2B2(Lamb,Delta,Permut,M,N,r):
    A2 = jnp.block([-jnp.kron(Lamb,np.eye(N)), jnp.eye(M*N)])@jnp.kron(Permut,jnp.eye(N))
    B2 = -jnp.kron(Delta,np.eye(N))@r
    return A2,B2



def get_transform_matrices(A2o,A2m,B2):
    m2,no = A2o.shape
    _,nm = A2m.shape
    U,S,V = np.linalg.svd(A2m,full_matrices=True)
    V = V.T
    m21 = len(S)
    nm1 = m21
    m22 = m2-m21
    nm2 = nm - nm1
    U1 = U[:,:m21]
    U2 = U[:,m21:]
    V1 = V[:,:nm1]
    V2 = V[:,nm1:]
    
    Sigma1 = np.diag(S)
    barA2o2 = U2.T@A2o
    
    W,R = np.linalg.qr(barA2o2.T,mode="complete")
    try:
        nW1 = np.linalg.matrix_rank(R)
    except:
        nW1 = 0
    nW2 = no - nW1
    
    W1 = W[:,:nW1]
    W2 = W[:,nW1:]
    
    hatA2m = V1@np.linalg.pinv(Sigma1)@U1.T
    
    Y1 = W1@np.linalg.pinv(U2.T@A2o@W1)@U2.T
    
    To_phi = np.block([[np.eye(no)],[-hatA2m@A2o]])@W2
    Tm_phi = np.block([[np.zeros((no,nm2))],[V2]])
    T_gamma = np.block([[-Y1@B2], [hatA2m@(A2o@Y1 - np.eye(m2))@B2]]);
    
    return To_phi,Tm_phi,T_gamma,W2,V2

    

def eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo):

    #m = (N*M)
    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])
        
    bar_xo2 = W2.T@xo
    m = bar_xo2.shape[0]
    

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    L_PhiSquared = linalg.cholesky(Phi.T@Phi)
    ln_det_PhiSquared = 2*(jnp.log(jnp.diag(L_PhiSquared))).sum()


    Z = Phi_m.T@Phi_m
    #P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T                                                   #previous code with inverse computation
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    
    
    L_Z = linalg.cholesky(Z)
    ln_det_Z = 2*(jnp.log(jnp.diag(L_Z))).sum()
    #return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@P@Phi_xo_Gamma)+(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)      #previous code with inverse computation
    return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma)-(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)

    
    
    
def get_ab(theta,n_ab):
    a = []
    b = []
    pos = 0
    for i in range(len(n_ab[0])):
        na = n_ab[0][i]
        nb = n_ab[1][i]
        a.append(theta[pos:pos+na])
        b.append(theta[pos+na:pos+na+nb])
        pos += (na+nb)
    return a,b



def get_xoxm(states,list_xo,N,M):
    p = np.arange(2*M)+1
    p = np.concatenate([list_xo,p])
    p_new = []
    for i in p:
        if not i in p_new:
            p_new.append(i)
    p = p_new
    Permut = permutation_mat(p)
    sorted_states = jnp.kron(Permut,jnp.eye(N)).T@states
    xo = sorted_states[:len(list_xo)*N]
    xm = sorted_states[len(list_xo)*N:]
    return xo,xm,Permut





def bcktrck_linsearch(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo,max_learning_rate,d_theta, direction,current_cost):    
    eta = max_learning_rate
    
    next_cost = eval_cost_func(theta-eta*direction,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2,xo)
    
    while np.isnan(next_cost) or np.isinf(next_cost) or (not np.isfinite(next_cost)) or  next_cost > current_cost - 0.1*eta*d_theta.T@direction:
        eta = eta/2
        next_cost = eval_cost_func(theta-eta*direction,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2,xo)
        print("bktk lin -  Next cost: ",next_cost)
        if eta < 0.0000000000001:
            break
    print("bktk lin -  learning rate: ",eta)
    return eta,next_cost







def optimize(theta, n_ab, Permut, N, M,To_phi, Tm_phi, T_gamma, W2, xo, n_iter):
    

    current_cost = eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
    #if not check_stability(theta,n_ab,dt=1):
    #    print("Initial theta infeasible!")
    #    return 

    print("Initial cost: ",current_cost)
    print("\n")


    for i in range(n_iter):
        if i == 0:
            max_eta = 0.0001
        else:
            max_eta = 0.001

        d_theta = grad_theta(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo)
        eta,current_cost = bcktrck_linsearch(theta, n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2,xo, max_eta, d_theta, d_theta, current_cost)    

        theta -= eta*d_theta
        print("Iteration: ",i+1,"  -  Cost function value: ",current_cost)
        print("\n")
        if eta<0.0000001 or current_cost>1000000 or np.isnan(current_cost):
            return theta,current_cost
    return theta,current_cost

grad_theta = grad(eval_cost_func, argnums=0)
Hessian_theta = jacfwd(lambda theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo : grad_theta(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo),argnums=0)




###### Fisher information matrix


def C_matrix(theta,lamb,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo):
    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])

    bar_xo2 = W2.T@xo
    

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    Z = Phi_m.T@Phi_m
    
    m = bar_xo2.shape[0]

    #PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T   
    #lamb = (1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma
    return lamb*jnp.linalg.inv(Phi_o.T@P@Phi_o)



def mu_o(theta,lamb,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo):
    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])
    
    mu = -jnp.linalg.inv(Phi)@Gamma
    bar_xo2 = W2.T@xo
    
    return mu[:bar_xo2.shape[0],:]



dC_dtheta = jacfwd(C_matrix,argnums=0)
dmuo_dtheta = jacfwd(mu_o,argnums=0)

    


def fisher_matrix(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo):
    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])

    Z = Phi_m.T@Phi_m

    bar_xo2 = W2.T@xo

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    m = bar_xo2.shape[0]


    lamb = (1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma

    dC = dC_dtheta(theta,lamb,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo)
    dmu_o = dmuo_dtheta(theta,lamb,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo)

    C_inv = jnp.linalg.inv(C_matrix(theta,lamb,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo))
    
    fisher_mat = np.zeros((theta.shape[0],theta.shape[0]))

    for i in range(theta.shape[0]):
        for j in range(theta.shape[0]):
            fisher_mat[i,j] = dmu_o[:,:,i,0].T@C_inv@dmu_o[:,:,j,0]+(1/2)*jnp.trace(C_inv@dC[:,:,i,0]@C_inv@dC[:,:,j,0])

    return fisher_mat










######################### Considering multiple experiments at once:

def gen_A1_multiple_exp(n_exp,theta,n_ab,Permut,N):
    M = n_ab.shape[1]
    A1_block = gen_A1(theta,n_ab,jnp.eye(2*M),N)
    for i in range(n_exp):
        if i ==0:
            A1 = A1_block
        else:
            A1 = linalg.block_diag(A1,A1_block)
    return A1@jnp.kron(Permut,jnp.eye(N))


def gen_A2B2_multiple_exp(n_exp,Lamb,Delta,Permut,M,N,r):
    for i in range(n_exp):
        if i ==0:
            A2,B2 = gen_A2B2(Lamb,Delta,np.eye(2*M),M,N,r[0:M*N,:].reshape(-1,1))
        else:
            A2_temp,B2_temp = gen_A2B2(Lamb,Delta,np.eye(2*M),M,N,r[i*M*N:(i+1)*M*N,:].reshape(-1,1))
            A2 = linalg.block_diag(A2,A2_temp)
            B2 = jnp.block([[B2],[B2_temp]])

    return A2@jnp.kron(Permut,jnp.eye(N)),B2


def eval_cost_func_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo):

    #m = (N*M*n_exp)
    A1 = gen_A1_multiple_exp(n_exp,theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])
        
    bar_xo2 = W2.T@xo
    m = bar_xo2.shape[0]
    

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    L_PhiSquared = linalg.cholesky(Phi.T@Phi)
    ln_det_PhiSquared = 2*(jnp.log(jnp.diag(L_PhiSquared))).sum()


    Z = Phi_m.T@Phi_m
    #P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T                                                   #previous code with inverse computation
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    
    
    L_Z = linalg.cholesky(Z)
    ln_det_Z = 2*(jnp.log(jnp.diag(L_Z))).sum()
    #return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@P@Phi_xo_Gamma)+(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)      #previous code with inverse computation
    return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma)-(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)



def bcktrck_linsearch_multiple_exp(theta,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2, xo,max_learning_rate,d_theta, direction,current_cost):    
    eta = max_learning_rate
    
    next_cost = eval_cost_func_multiple_exp(theta-eta*direction,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2,xo)
    
    while np.isnan(next_cost) or np.isinf(next_cost) or (not np.isfinite(next_cost)) or  next_cost > current_cost - 0.1*eta*d_theta.T@direction:
        eta = eta/2
        next_cost = eval_cost_func_multiple_exp(theta-eta*direction,n_ab,Permut,N,M,n_exp,To_phi, Tm_phi, T_gamma, W2,xo)
        print("bktk lin -  Next cost: ",next_cost)
        if eta < 0.0000000000001:
            break
    print("bktk lin -  learning rate: ",eta)
    return eta,next_cost



def get_xoxm_multiple_exp(states,list_xo,N,M,n_exp):
    p = np.arange(2*M*n_exp)+1
    p = np.concatenate([list_xo,p])
    p_new = []
    for i in p:
        if not i in p_new:
            p_new.append(i)
    p = p_new
    Permut = permutation_mat(p)
    sorted_states = jnp.kron(Permut,jnp.eye(N)).T@states
    xo = sorted_states[:len(list_xo)*N]
    xm = sorted_states[len(list_xo)*N:]
    return xo,xm,Permut



grad_theta_multiple_exp = grad(eval_cost_func_multiple_exp, argnums=0)
Hessian_multiple_exp = jacfwd(lambda theta,n_ab, Permut, N, M,n_exp, To_phi, Tm_phi, T_gamma, W2, xo : grad_theta_multiple_exp(theta,n_ab, Permut, N, M,n_exp, To_phi, Tm_phi, T_gamma, W2, xo),argnums=0)




def C_matrix_multi(theta,lamb,n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2, xo):
    A1 = gen_A1_multiple_exp(n_exp,theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])

    bar_xo2 = W2.T@xo
    

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    Z = Phi_m.T@Phi_m
    
    m = bar_xo2.shape[0]

    #PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T   
    #lamb = (1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma
    return lamb*jnp.linalg.inv(Phi_o.T@P@Phi_o)



def mu_o_multi(theta,lamb,n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2, xo):
    A1 = gen_A1_multiple_exp(n_exp,theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])
    
    mu = -jnp.linalg.inv(Phi)@Gamma
    bar_xo2 = W2.T@xo
    
    return mu[:bar_xo2.shape[0],:]



dC_dtheta_multi = jacfwd(C_matrix_multi,argnums=0)
dmuo_dtheta_multi = jacfwd(mu_o_multi,argnums=0)

    


def fisher_matrix_multi(theta,n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2, xo):
    A1 = gen_A1_multiple_exp(n_exp,theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])

    Z = Phi_m.T@Phi_m

    bar_xo2 = W2.T@xo

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    m = bar_xo2.shape[0]


    lamb = (1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma

    dC = dC_dtheta_multi(theta,lamb,n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2, xo)
    dmu_o = dmuo_dtheta_multi(theta,lamb,n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2, xo)

    C_inv = jnp.linalg.inv(C_matrix_multi(theta,lamb,n_ab, Permut, N, M, n_exp, To_phi, Tm_phi, T_gamma, W2, xo))
    
    fisher_mat = np.zeros((theta.shape[0],theta.shape[0]))

    for i in range(theta.shape[0]):
        for j in range(theta.shape[0]):
            fisher_mat[i,j] = dmu_o[:,:,i,0].T@C_inv@dmu_o[:,:,j,0]+(1/2)*jnp.trace(C_inv@dC[:,:,i,0]@C_inv@dC[:,:,j,0])

    return fisher_mat



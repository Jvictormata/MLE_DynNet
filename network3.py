from MLE_functions import *


def check_stability(theta,n_ab,dt=1):
    a,b = get_ab(theta,n_ab)
    sys1 = (b[0],np.append([1],a[0]),1)
    sys2 = (b[1],np.append([1],a[1]),1)
    sys3 = (b[2],np.append([1],a[2]),1)    

    num1 = sys1[0].reshape(-1)
    den1 = sys1[1].reshape(-1)
    num2 = sys2[0].reshape(-1)
    den2 = sys2[1].reshape(-1)
    num3 = sys3[0].reshape(-1)
    den3 = sys3[1].reshape(-1)
    
    num = signal.convolve(signal.convolve(den1,den2),den3)
    den = np.polyadd(np.polyadd(signal.convolve(signal.convolve(den1,den2),den3),-signal.convolve(signal.convolve(num1,num2),den3)),-signal.convolve(signal.convolve(num2,num3),den1))
    sys = (num,den,sys1[2])

    if np.abs(np.roots(sys[1].reshape(-1))).max() > 1:
        print("WARNING: Closed loop System instable!")
        return 0
    else: 
        print("Closed loop System stable!")
    
    for i in range(len(a)):
        if np.abs(np.roots(np.append([1],a[i]))).max() > 1:
            print("WARNING: System ",i+1," instable!")
            return 0
        else:
            print("System ",i+1," stable!")
    return 1

    


def gen_states(sys1,sys2,sys3,r,e=[]):

    num1 = sys1[0].reshape(-1)
    den1 = sys1[1].reshape(-1)
    num2 = sys2[0].reshape(-1)
    den2 = sys2[1].reshape(-1)
    num3 = sys3[0].reshape(-1)
    den3 = sys3[1].reshape(-1)
    
    num = signal.convolve(signal.convolve(den1,den2),den3)
    den = np.polyadd(np.polyadd(signal.convolve(signal.convolve(den1,den2),den3),-signal.convolve(signal.convolve(num1,num2),den3)),-signal.convolve(signal.convolve(num2,num3),den1))
    base_Tf = (num.reshape(-1),den.reshape(-1),sys1[2])
    
    num_temp = np.polyadd(signal.convolve(den2,den3),-signal.convolve(num2,num3))
    den_temp = signal.convolve(den2,den3)
    
    Tu1_r1 = series_sys((num_temp.reshape(-1),den_temp.reshape(-1),sys1[2]),base_Tf)
    
    Tu1_e2 = base_Tf
    Tu1_e1 = series_sys(sys2,base_Tf)
    Tu1_r2 = Tu1_e1
    Tu1_e3 = Tu1_e1
    Tu1_r3 = series_sys(sys3,Tu1_e1)

    u1 = signal.dlsim(Tu1_r1,r[:,0])[1] + signal.dlsim(Tu1_r2,r[:,1])[1] + signal.dlsim(Tu1_r3,r[:,2])[1]
    if len(e):
        u1 = u1 + signal.dlsim(Tu1_e1,e[:,0])[1] + signal.dlsim(Tu1_e2,e[:,1])[1] + signal.dlsim(Tu1_e3,e[:,2])[1]
    
    y2 = u1 - r[:,0].reshape(-1,1)

    y1 = signal.dlsim(sys1,u1)[1]
    if len(e):
        y1 = y1 + e[:,0].reshape(-1,1)

    u3 = y2 + r[:,2].reshape(-1,1)
    y3 = signal.dlsim(sys3,u3)[1]
    if len(e):
        y3 = y3 + e[:,2].reshape(-1,1)
    
    u2 = y1 + y3 + r[:,1].reshape(-1,1)
    
    return np.concatenate([y1,y2,y3,u1,u2,u3])





def gen_xoxm_from_theta(theta,n_ab,obs,N,M,r,e=[]):
    a,b = get_ab(theta,n_ab)
    
    sys1_est = (b[0].reshape(-1),np.append([1],a[0]),1)
    sys2_est = (b[1].reshape(-1),np.append([1],a[1]),1)
    sys3_est = (b[2].reshape(-1),np.append([1],a[2]),1)

    state_est = gen_states(sys1_est,sys2_est,sys3_est,r,e)

    xo_est,xm_est,_ = get_xoxm(state_est,obs,N,M)
    return xo_est,xm_est
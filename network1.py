from MLE_functions import *


def check_stability(theta,n_ab,dt=1):
    a,b = get_ab(theta,n_ab)
    sys1 = (np.append([0],b[0]),np.append([1],a[0]),1)
    sys3 = (np.append([0],b[2]),np.append([1],a[2]),1)    
    sys = feedback_sys(sys1,sys3)
    if np.abs(np.roots(sys[1])).max() > 1:
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
    Tu3_r1 = feedback_sys(sys1,sys3)
    Tu3_r2 = series_sys(feedback_sys(sys1,sys3),sys2)
    Tu3_r3 = feedback_sys((np.array([1]),np.array([1])),series_sys(sys1,sys3))


    Tu3_e1 = feedback_sys((np.array([1]),np.array([1])),series_sys(sys1,sys3))
    Tu3_e2 = feedback_sys(sys1,sys3)
    Tu3_e3 = feedback_sys(sys1,sys3)
    
    u3_r = signal.dlsim(Tu3_r1,r[:,0])[1]+signal.dlsim(Tu3_r2,r[:,1])[1]+signal.dlsim(Tu3_r3,r[:,2])[1]
    if len(e):
        u3_e = signal.dlsim(Tu3_e1,e[:,0])[1]+signal.dlsim(Tu3_e2,e[:,1])[1]+signal.dlsim(Tu3_e3,e[:,2])[1]
        u3 = u3_r + u3_e 
    else:
        u3 = u3_r
        
    y1 = u3 - r[:,2].reshape(-1,1)
    y2 = signal.dlsim(sys2,r[:,1])[1]
    y3 = signal.dlsim(sys3,u3)[1]
    u1 = y2 + y3 + r[:,0].reshape(-1,1)
    u2 = r[:,1].reshape(-1,1)
    
    if len(e):
        y2 = y2+e[:,1].reshape(-1,1)
        y3 = y3+e[:,2].reshape(-1,1)
        
    return np.concatenate([y1,y2,y3,u1,u2,u3])




def gen_xoxm_from_theta(theta,n_ab,obs,N,M,r,e=[]):
    a,b = get_ab(theta,n_ab)
    
    sys1_est = (b[0].reshape(-1),np.append([1],a[0]),1)
    sys2_est = (b[1].reshape(-1),np.append([1],a[1]),1)
    sys3_est = (b[2].reshape(-1),np.append([1],a[2]),1)

    state_est = gen_states(sys1_est,sys2_est,sys3_est,r,e)

    xo_est,xm_est,_ = get_xoxm(state_est,obs,N,M)
    return xo_est,xm_est

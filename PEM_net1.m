clear all
clc

% script generating data for interconnection of three ARX models
M = 3;          % number of interconnected systems
N = 500;        % number of samples (lenght of time interval)
sigma = 0.1;   % noise standard deviation


load('exp_1/x_1.mat');
load('exp_1/r_1.mat');
load('exp_1/e_1.mat');

load('exp_1/x_validation.mat');
load('exp_1/r_validation.mat');
load('exp_1/e_validation.mat');


u1 = x(3*N+1:4*N);
u2 = x(4*N+1:5*N);
u3 = x(5*N+1:6*N);
r1 = double(r(:,1));
r2 = double(r(:,2));
r3 = double(r(:,3));

Z1 = iddata(u1-r1,[u2 u3]);
nf = [2 2]; 
nb = [2 2];
nk = [1 1];
est23 = oe(Z1, [nb nf nk])

Z2 = iddata(u3-r3,[u1]);
nf2 = [2]; 
nb2 = [2];
nk2 = [1];
est1 = oe(Z2, [nb2 nf2 nk2])

cov23 = getcov(est23);
cov1 = getcov(est1);


%% Computing Fit value

u1_val = x_val(3*N+1:4*N);
u2_val = x_val(4*N+1:5*N);
u3_val = x_val(5*N+1:6*N);
r1_val = double(r_val(:,1));
r2_val = double(r_val(:,2));
r3_val = double(r_val(:,3));

G1_hat = tf(est1.B,est1.F,1);
G2_hat = tf(est23.B{1},est23.F{1},1);
G3_hat = tf(est23.B{2},est23.F{2},1);

Tu3_r1 = feedback(G1_hat,G3_hat);
Tu3_r2 = series(Tu3_r1,G2_hat);
Tu3_r3 = feedback(1,series(G1_hat,G3_hat));



u3_hat = lsim(Tu3_r1,r1_val)+lsim(Tu3_r2,r2_val)+lsim(Tu3_r3,r3_val);
u1_hat = r1_val + lsim(G2_hat,r2_val)+lsim(G3_hat,u3_hat);
u2_hat = r2_val;
y1_hat = lsim(G1_hat,u1_hat);
y2_hat = lsim(G2_hat,r2_val);
y3_hat = lsim(G3_hat,u3_hat);


%fit_1 = 1 - norm(u3_hat-u3_val)/norm(u3_hat-mean(u3_val))
%fit_2 = 1 - norm(u1_hat-u1_val)/norm(u1_hat-mean(u1_val))


xo_hat = [u1_hat; u2_hat; u3_hat];
xo_val = x_val(3*N+1:6*N);

xm_hat = [y1_hat; y2_hat; y3_hat];
xm_val = x_val(0*N+1:3*N);

fit_xo = 1 - norm(xo_hat-xo_val)/norm(xo_hat-mean(xo_val))
fit_xm = 1 - norm(xm_hat-xm_val)/norm(xm_hat-mean(xm_val))


%[trace(cov1),max(eig(cov1)),det(cov1)]
%[trace(cov23),max(eig(cov23)),det(cov23)]

cov = blkdiag(cov1,cov23);
[trace(cov),max(eig(cov)),det(cov)]




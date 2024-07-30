clear all
clc

% script generating data for interconnection of three ARX models
M = 3;          % number of interconnected systems
N = 500;        % number of samples (lenght of time interval)
sigma = 0.1;   % noise standard deviation

load('exp_2/x_2.mat');
load('exp_2/r_2.mat');
load('exp_2/e_2.mat');

load('exp_2/x_validation2.mat');
load('exp_2/r_validation2.mat');
load('exp_2/e_validation2.mat');


y2 = x(1*N+1:2*N);
y3 = x(2*N+1:3*N);
r1 = double(r(:,1));
r2 = double(r(:,2));
r3 = double(r(:,3));


%%

Z1 = iddata(y2,[(y3+r2) r1]);
nf = [2 4]; 
nb = [2 4];
nc = [2];
nd = [2];
nk = [1 1];
est2 = bj(Z1, [nb nc nd nf nk])
%est2 = oe(Z1, [nb nf nk])


Z2 = iddata(y3,[r3 r1]);
nf = [2 4]; 
nb = [2 4];
nc = [2];
nd = [2];
nk = [1 1];
est3= bj(Z2, [nb nc nd nf nk])
%est3= oe(Z2, [nb nf nk])


cov2 = getcov(est2);
cov3 = getcov(est3);


%% Computing Fit value

y2_val = x_val(1*N+1:2*N);
y3_val = x_val(2*N+1:3*N);
r1_val = double(r_val(:,1));
r2_val = double(r_val(:,2));
r3_val = double(r_val(:,3));


y3_hat = lsim(tf(est3.B{1},est3.F{1},1),r3_val) + lsim(tf(est3.B{2},est3.F{2},1),r1_val);
y2_hat = lsim(tf(est2.B{1},est2.F{1},1),y3_hat+r2_val) + lsim(tf(est2.B{2},est2.F{2},1),r1_val);

xo_hat = [y2_hat;y3_hat];
xo_val = x_val(1*N+1:3*N);

%fit_2 = 1 - norm(y2_hat-y2_val)/norm(y2_hat-mean(y2_val))
%fit_3 = 1 - norm(y3_hat-y3_val)/norm(y3_hat-mean(y3_val))

fit_xo = 1 - norm(xo_hat-xo_val)/norm(xo_hat-mean(xo_val))


%[trace(cov2),max(eig(cov2)),det(cov2)]
%[trace(cov3),max(eig(cov3)),det(cov3)]

cov = blkdiag(cov2,cov3);
[trace(cov),max(eig(cov)),det(cov)]

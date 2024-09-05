clear all
clc

% script generating data for interconnection of three ARX models
M = 3;          % number of interconnected systems
N = 500;        % number of samples (lenght of time interval)
sigma = 0.1;   % noise standard deviation

load('exp_3/x_3.mat');
load('exp_3/r_3.mat');
load('exp_3/e_3.mat');

load('exp_3/x_validation3.mat');
load('exp_3/r_validation3.mat');
load('exp_3/e_validation3.mat');

y1 = x(0*N+1:1*N);
y2 = x(1*N+1:2*N);
r1 = double(r(:,1));
r2 = double(r(:,2));
r3 = double(r(:,3));


%%

Z1 = iddata(y1,(y2+r1));
nf = [2]; 
nb = [2];
nk = [1];
est1 = oe(Z1, [nf nb nk])


Z2 = iddata(y2,[r2+y1 y2+r3]);
nf = [2 2];  %better result than [2 3]
nb = [2 2];  %better result than [2 3]
nc = [2];
nd = [2];
nk = [1 1];
est23= bj(Z2, [nb nc nd nf nk])


cov1 = getcov(est1);
cov23 = getcov(est23);


%% Computing Fit value

y1_val = x_val(0*N+1:1*N);
y2_val = x_val(1*N+1:2*N);
r1_val = double(r_val(:,1));
r2_val = double(r_val(:,2));
r3_val = double(r_val(:,3));


G1 = tf(est1.B,est1.F,1);
G2 =tf(est23.B{1},est23.F{1},1);
G3 =tf(est23.B{2},est23.F{2},1);

G1G2 = series(G1,G2);
G2G3 = series(G2,G3);


Tbase = feedback(1,-G1G2-G2G3);
Tr1 = series(Tbase,1-G2G3);
Tr2 = series(G2,Tbase);
Tr3 = series(G3,Tr2);

u1_hat = lsim(Tr1,r1_val)+ lsim(Tr2,r2_val)+ lsim(Tr3,r3_val);
y2_hat = u1_hat - r1_val;
y1_hat = lsim(G1,u1_hat);
u3_hat = y2_hat + r3_val;
y3_hat = lsim(G3,u3_hat);
u2_hat = y3_hat + y1_hat + r2_val;


xo_hat = [y1_hat;y2_hat];
xo_val = [y1_val;y2_val];
xm_hat = [y3_hat;u1_hat;u2_hat;u3_hat];
xm_val = x_val(2*N+1:6*N);

%fit_1 = 1 - norm(y1_hat-y1_val)/norm(y1_hat-mean(y1_val))
%fit_2 = 1 - norm(y2_hat-y2_val)/norm(y2_hat-mean(y2_val))

fit_xo = 1 - norm(xo_hat-xo_val)/norm(xo_hat-mean(xo_val))
fit_xm = 1 - norm(xm_hat-xm_val)/norm(xm_hat-mean(xm_val))


%[trace(cov1),max(eig(cov1)),det(cov1)]
%[trace(cov23),max(eig(cov23)),det(cov23)]

cov = blkdiag(cov1,cov23);
[trace(cov),max(eig(cov)),det(cov)]
clear all
clc

M = 10;
N = 50;
sigma = 0.1;

L = 100

trial = 0
while true
theta = 2*rand(1,2*M)-1;
out = sim('net_test.slx',5000);
y1 = out.y1;
y2 = out.y2;
y3 = out.y3;
y4 = out.y4;
y5 = out.y5;
y6 = out.y6;
y7 = out.y7;
y8 = out.y8;
y9 = out.y9;
y10 = out.y10;
n = size(y3,1);
if y1(n)<L & y2(n)<L & y3(n)<L & y4(n)<L & y5(n)<L & y6(n)<L & y7(n)<L & y8(n)<L & y9(n)<L &   y10(n)<L 
    break
else
    trial = trial +1
end
end

r1 = out.r1;
r2 = out.r2;
r3 = out.r3;

u1 = y2+y3+r1;
u2 = y1+r2;
u3 = y8+r3;
u4 = y10+y9;
u5 = y10+y9;
u6 = y7+y5;
u7 = y4+y6;
u8 = y7+y5;
u9 = y8+r3;
u10 = y1+r2;


x = [y1(1:N);y2(1:N);y3(1:N);y4(1:N);y5(1:N);y6(1:N);y7(1:N);y8(1:N);y9(1:N);y10(1:N);u1(1:N);u2(1:N);u3(1:N);u4(1:N);u5(1:N);u6(1:N);u7(1:N);u8(1:N);u9(1:N);u10(1:N)];
r = [r1(1:N) r2(1:N) r3(1:N)];

save("big_net1.mat","x","r","theta")
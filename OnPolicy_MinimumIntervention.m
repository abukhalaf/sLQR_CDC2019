%SLQR Shared Linear Quadratic Control
%   On-policy learning of a Minimum Inervention policy.
%
%   Used to generate Figure 3 in:
%
%   Murad Abu-Khalaf, Sertac Karaman, Daniela Rus, "Shared Linear Quadratic
%   Regulation Control: A Reinforcement Learning Approach", to appear in
%   IEEE CDC 2019.
%
%   The example considered is for a car-following problem where the following
%   car needs to stay at a particular spacing from the preceding car while
%   also converging to a desired speed.

% Author: Murad Abu-Khalaf
% Date: March-13-2019

clc; clear; close all;

%% Uses multiple trajectories equal at least the number of unknowns

% Car following - two cars involved

m1 = 1; % Mass of leading car
m2 = 1; % Mass of following car
alpha1 = 1; % Taylor series 1st coefficient for the preceding car
alpha2 = 1; % Taylor series 1st coefficient for the following car

Ch = [ 0 0 0; 
       0 1 0;
       0 0 1];
 
Kh = [0 1 -1];

B = [ 0;
      0;
      1/m2];
  
A =[-alpha1/m1   0            0;
     1           0           -1;
     0           0   -alpha2/m2];

[n,m]=size(B);

Q = 5*eye(n);
M = eye(m);
R = 10*eye(m);

Tr = 0.01; % Reward window
Tn = 0.01; % Nudge time

P = zeros(n);

nUnknowns = n*(n+1)/2; % Accounts for symmetry
nTrajectorySamplesPerPolicy = 1 * nUnknowns; % Number of Unknowns Accounting for Symmetry
nTrajectorySamples = 5*nTrajectorySamplesPerPolicy;

PHI = zeros(nTrajectorySamplesPerPolicy,nUnknowns);
Y = zeros(nTrajectorySamplesPerPolicy,1);

x = [40 100 10]; % Keep as row to be compataible with ode x
t = 0;

% These are used to store data for plotting
t_all = []; x_all = []; uh_all = []; u_all = [];

for i = 0 : (nTrajectorySamples-1)
    j = mod(i,nTrajectorySamplesPerPolicy) + 1;
    
    X_t0 = sigma(x(end,:)');
    
    Ki = -inv(R)*B'*P;
    [t,x_v]= ode45(@(t,x_v) AxBu_V(t,x_v,A,B,Ch,Kh,Q,M,R,Ki),t(end) + [0 Tr],[x(end,:)';0;0;0]);
    x = x_v(:,1:n);
    vx = x_v(end,end-2);
    vh = x_v(end,end-1);
    va = x_v(end,end);
    
    x_all  = [x_all ;x];  %#ok<AGROW> % The end points are unecessarily duplicated
    t_all  = [t_all ; t]; %#ok<AGROW> % The end points are unecessarily duplicated
    uh_all = [uh_all; x*(Kh*Ch)']; %#ok<AGROW>
    u_all = [u_all; x*(Ki)']; %#ok<AGROW>
    
    X_t1 = sigma(x(end,:)');
    
    PHI(j,:)= X_t1' - X_t0';
    Y(j,:) = -vx - vh - va;
    
    PRBS = randn;
    [t,x] = ode45(@(t,x) nudge(t,x,A,B,Ch,Kh,Ki,PRBS), t(end) + [0 Tn], x(end,:)' );
    %x = 10*randn(n,1); x = x + 0.001*randn(n,1);
    
    x_all  = [x_all ;x];  %#ok<AGROW> % The end points are unecessarily duplicated
    t_all  = [t_all ; t]; %#ok<AGROW> % The end points are unecessarily duplicated
    uh_all = [uh_all; x*(Kh*Ch)']; %#ok<AGROW>
    u_all = [u_all; x*(Ki)' + PRBS]; %#ok<AGROW>
    
    % the batch least squares is made on 6 values
    if j == nTrajectorySamplesPerPolicy
        W=PHI\Y;
        %Calculating the P matrix
        P = getP(W,n); display(P);
        eig(A+B*Kh*Ch+B*Ki);
        Qc = Q + Ch'*Kh'*M*Kh*Ch;
        P_LYAP = lyap((A+B*Kh*Ch+B*Ki)',Qc + Ki'*R*Ki); display(P_LYAP);
    end
end

% Verify the solution of the ARE
P_CARE = care(A+B*Kh,B,Qc,R); display(P_CARE);

Ki = -inv(R)*B'*P;
[t,x_v]= ode45(@(t,x_v) AxBu_V(t,x_v,A,B,Ch,Kh,Q,M,R,Ki),t(end) + [0 3],[x(end,:)';0;0;0]);
x = x_v(:,1:n);

x_all  = [x_all ;x];  % The end points are unecessarily duplicated
t_all  = [t_all ; t]; % The end points are unecessarily duplicated
uh_all = [uh_all; x*(Kh*Ch)'];
u_all = [u_all; x*(Ki)'];


figure(1); hold on;

subplot(2,1,1); hold on;
plot(t_all,x_all,'LineWidth',4);set(gca,'FontSize',30);
title('On-Policy Learning of a Minimum Intervention Policy','FontSize',30);xlabel('Time (sec)','FontSize',30);ylabel('State Variables','FontSize',30);
legend({'$\tilde{v}_1(t)$','$\tilde{s}(t)$','$\tilde{v}_2(t)$'},'FontSize',30,'Interpreter','latex')

subplot(2,1,2); hold on;
plot(t_all,uh_all,'LineWidth',4); plot(t_all,u_all,'LineWidth',4); set(gca,'FontSize',30);
xlabel('Time (sec)','FontSize',30);ylabel('Control Inputs','FontSize',30);
legend({'$u_h(t)$','$u_a(t)$'},'FontSize',30,'Interpreter','latex')

function xdot = nudge(~,x,A,B,Ch,Kh,Ki,pseudorandom)

%calculating the control signal
ui = Ki*x;
ua = ui + pseudorandom;
uh = Kh*Ch*x;
u = uh + ua;

%updating the derivative of the state=[x(1:3) V]
xdot = A*x + B*u;
end

function x_vdot=AxBu_V(~,x_v,A,B,Ch,Kh,Q,M,R,Ki)
[n,~]=size(B);
x = x_v(1:n);
%calculating the control signal
ui = Ki*x;
uh = Kh*Ch*x;
ua = ui;
u = uh + ua;

%updating the derivative of the state=[x(1:3) V]
x_vdot=[A*x + B*u
        x'*Q*x
        uh'*M*uh
        ui'*R*ui];
end

function vec_L = sigma(x)
% For n=6, this returns
% [x(1)*x(1:6); x(2)*x(2:6); x(3)*x(3:6); x(4)*x(4:6); x(5)*x(5:6); x(6)*x(6)];

n = numel(x);
vec_L = kron(x,x);

% sum(1:(n-1)) is the number of off-diagonal upper triangular elements of x*x'
idx = zeros(sum(1:(n-1)),1); 

for i=2:n  % for all columns from 2 to n
    i1 = sum(1:(i-2)) + 1; % i1 is equal to previous i2 + 1. On first iteration, i1 = i2.
    i2 = sum(1:(i-1));     % Cumulative number of UpperTriangular elements by this column
    UpperTriangularIndexKron = (i-1)*n + (1:(i-1));  % Indices of the off-diagonal upper triangular elements in Kron vector
    idx(i1:i2,1) = UpperTriangularIndexKron;
end

vec_L(idx) = []; % Eliminate collinear regressors
end

function P = getP(W,n)
P = zeros(n);
idx = 1;
for j=1:n
    for i = 1:j-1
        P(i,j) = P(j,i);
    end
    for i = j:n
        if i==j
            P(i,j) = W(idx);
        else
            P(i,j) = W(idx)/2;
        end
        idx = idx + 1;
    end
end
end
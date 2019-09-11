%SLQR Shared Linear Quadratic Control
%   This is a baseline for comparision when only human input is involved
%   and when no takeover takes place.
%
%   Used to generate the baselines for Figures 3, 4 and 5 in:
%
%   Murad Abu-Khalaf, Sertac Karaman, Daniela Rus, "Shared Linear Quadratic
%   Regulation Control: A Reinforcement Learning Approach", to appear in
%   IEEE CDC 2019.
%
%   The example considered is for a car-following problem where the following
%   car needs to stay at a particular spacing from the preceding car while
%   also converging to a desired speed.

% Author: Murad Abu-Khalaf
% Last Updated: March-14-2019

clc; clear;

%% Uses multiple trajectories equal at least the number of unknowns

% Car following - two cars involved

m1 = 1; % Mass of preceding car
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

R = 10*eye(m);

% These are used to store data for plotting
t_all = []; x_all = []; uh_all = [];

intervention = false;

if intervention
    % Intevention Baseline
    [t,x]= ode45(@(t,x) AxBu(t,x,A,B,Kh,Ch), 0 + [0 3.6], [40 100 10]);
    x_all  = [x_all ;x];  % The end points are unecessarily duplicated
    t_all  = [t_all ; t]; % The end points are unecessarily duplicated
    uh_all = [uh_all; x*(Kh*Ch)'];
    
    figure(1); hold on;
    subplot(2,1,1); hold on;
    plot(t_all,x_all,'--k');
    legend({'$\tilde{v}_1(t)$','$\tilde{s}(t)$','$\tilde{v}_2(t)$','Without Intervention'},'FontSize',30,'Interpreter','latex')
else
    % Takeover Baseline
    Kh = [0 1 -1];
    [t,x]= ode45(@(t,x) AxBu(t,x,A,B,Kh,Ch), 0 + [0 0.42], [40 100 10]);
    x_all  = [x_all ;x];  % The end points are unecessarily duplicated
    t_all  = [t_all ; t]; % The end points are unecessarily duplicated
    uh_all = [uh_all; x*(Kh*Ch)'];
    
    Kh = [0 0 0];
    [t,x]= ode45(@(t,x) AxBu(t,x,A,B,Kh,Ch), t(end) + [0 3], x(end,:)');
    x_all  = [x_all ;x];  % The end points are unecessarily duplicated
    t_all  = [t_all ; t]; % The end points are unecessarily duplicated
    uh_all = [uh_all; x*(Kh*Ch)'];
    
    figure(1); hold on;
    subplot(2,1,1); hold on;
    plot(t_all,x_all,'--k');
    legend({'$\tilde{v}_1(t)$','$\tilde{s}(t)$','$\tilde{v}_2(t)$','Without Takeover'},'FontSize',30,'Interpreter','latex')
end

function x_vdot=AxBu(~,x,A,B,Kh,Ch)
uh = Kh*Ch*x;
x_vdot = A*x + B*uh;
end
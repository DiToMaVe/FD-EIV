% Example 3 in Soverini & Catania (2014), using the Koopman-Levin method
% for frequency domain identification.
clear all, close all, clc

%% Experiment parameters
%%%%%%%%%%%%%%%%%%%%%%%
fs = 400;   % sampling rate in Hz
Ts = 1/fs;  % sample time in s
N = 4000;   % lenght of data sequency
N_mc = 100; % number Monte Carlo simulations

%% Define mdof system
%%%%%%%%%%%%%%%%%%%%%
n_dof = 3;

% Mass matrix
M = diag([11,5,10]);
% Stifness matrix
k1 = 10^5; k2 = 4*10^5; k3 = 6*10^5; k4 = 2*10^5;
K = [k1+k2 -k2 0; -k2 k2+k3 -k3; 0 -k3 k3+k4];

% Damping matrix
c1 = 110.15; c2 = 0.6; c3 = 0.9; c4 = 100.3; c5 = 50; 
C = [c1+c2 -c2 0; -c2 c2+c3+c5 -c3; 0 -c3 c3+c4];

% Construct system model
%%%%%%%%%%%%%%%%%%%%%%%%

% Inverse mass matrix 
diag_M = diag(M);
M_inv = diag(1./diag_M);

% Matrices of the augmented state-space model
A_ss = zeros(2*n_dof);
B_ss = zeros(2*n_dof,n_dof);
C_ss = zeros(n_dof,2*n_dof);
D_ss = zeros(n_dof);

A_ss(1:n_dof,n_dof+1:end) = eye(n_dof);
A_ss(n_dof+1:end,1:n_dof) = -M_inv*K;
A_ss(n_dof+1:end,n_dof+1:end) = -M_inv*C;
B_ss(n_dof+1:end,:) = M_inv;
C_ss(:,1:n_dof) = eye(n_dof);

% Construct Matlab ss-model
ss_c = ss(A_ss,B_ss,C_ss,D_ss);
ss_d = c2d(ss_c,Ts);

% Convert to zpk model
zpk_c = zpk(ss_c);
% Get zpk data
[z,p,k] = zpkdata(zpk_c);

% Find transmissibility G13 = Y13/Y33
G13_z = cell2mat(z(3,1));
G13_p = cell2mat(z(3,3));
G13_k = k(3,1)/k(3,3);
G13_c = zpk(G13_z,G13_p,G13_k);

% Bode diagram of the transmissibility (continuous)
opts = bodeoptions;
opts.FreqUnits = 'Hz';
opts.FreqScale = 'linear';
opts.Xlim = [0,160];
figure(1)
bode(G13_c,opts)

% Convert to discrete system with default zoh method
G13 = c2d(G13_c,Ts);

% Bode diagram of the transmissibility (discrete)
figure(2)
bode(G13,opts)

%% Simulation of system response
% The system is excited on the third mass by a discrete-time white noise
% process with unit variance and length N = 4000. The measurements are
% disturbed by white noise realizations with variances lambda_y3 = 0.02 and
% lambda_y1 = 0.02 (rho=1) corresponding to a SNR of about 30 dB.

sigma_y1 = sqrt(0.02);
sigma_y3 = sqrt(0.02);
rho = sigma_y1/sigma_y3;

% time vector
t = (0:1:N-1)*Ts;           

% Allocate matrices
f12 = zeros(N,n_dof-1);     % excitation force     
f3 = zeros(N,N_mc);
y = zeros(N,n_dof,N_mc);    % response signals, nodal displacements

for ii = 1:1%N_mc
    f3(:,ii) = 10^12*randn(N,1);
    f = [zeros(N,n_dof-1),f3(:,ii)];
    n_y1 = sigma_y1*randn(N,1);
    n_y3 = sigma_y1*randn(N,1);
    y(:,:,ii) = lsim(ss_d,f,t)+[n_y1,zeros(N,1),n_y3]; 
end

%% The FD-KL method

% System order
n = 4;
N = size(y(:,1,ii),1);

% Construct DFT matrices Pi and Psi (32)
omega = 2*pi*(0:1:N-1)/N;
power = repmat(n:-1:0,N,1);
Omega = repmat(exp(-1i*omega'),1,n+1); 
Pi = Omega.^power;
Omega(:,1) = [];
power(:,1) = [];
Psi = Omega.^power;

% Compute DFT
ii = 1;
uu = y(:,3,ii);
yy = y(:,1,ii);
Vu = 1/sqrt(N)*fft(uu);  % (21)
Vy = 1/sqrt(N)*fft(yy);  % (22)

% Alternative computation of Vu, Vy
jj = 0:N-1;
kk = 0:N-1;
power = jj'*kk;

Fn = 1/sqrt(N)*exp(-1i*2*pi/N).^power;
Vu = Fn*y(:,3,ii);
Vy = Fn*y(:,1,ii);

% Construct data matrices Vu and Vy (33-34)
Vu = diag(Vu);
Vy = diag(Vy);

% Compute the matrices Phi_A, Phi_B and Phi_T (35-36)
Phi_A = Vy*Pi;
Phi_B = Vu*Pi;
Phi_T = Psi;
% Construct the matrix Phi (37)
Phi = [Phi_A,Phi_B,Phi_T];

% Compute the positive definite matrix Sigma (46)
Sigma = 1/N*(Phi'*Phi);
Lambda = zeros(3*n+2);
Lambda(1:n+1,1:n+1) = rho*eye(n+1);
Lambda(n+2:2*n+2,n+2:2*n+2) = eye(n+1);

% Solve for lambda_u (51)
[~,D] = eig(Sigma\Lambda);
lambda_u = 1/max(abs(diag(D)));     % ensure that lambda is a real number

% Obtain the parameter vector as the kernel of (52) 
Theta = null(Sigma-lambda_u*Lambda);
Theta = real(Theta);                 % ensure parameters are real-valued

% Normalize the(n+1)-th entry of the parameter vector to 1.
Theta = Theta/Theta(n+1);
A = fliplr(Theta(1:n+1)');
B = -fliplr(Theta(n+2:2*n+2)');

%% Inspect identified model
sys_id = tf(B,A,Ts);

figure(3)
bode(sys_id)

q = exp(-sqrt(-1)*omega(1:N/2-1));        % z^(-1) as a function of the frequency 
G_id = polyval(fliplr(B), q) ./ polyval(fliplr(A), q);
G_id = G_id.';
figure(4)
plot(omega(1:N/2-1),db(G_id), 'r')

%% LPM estimate
% data 
data.u = y(:,3,ii)';                 
data.y = y(:,1,ii)';                             
data.Ts = Ts;                         

method.order = 2;             
method.startfreq = fs/N;        
method.stopfreq = (N/2-1)*fs/N;

% local polynomial estimate FRF and its variance
[CZ, Z, freq, G, CvecG, dof, CL] = ArbLocalPolyAnal(data, method);

G = squeeze(G);                         % FRF estimate
varG = squeeze(CvecG);                  % variance FRF estimate
varV = squeeze(CZ.n(1,1,:));            % estimate output noise variance


%% Inspect results

% Comparison identified model and estimated FRF
figure(4)
plot(freq, db(G), 'r');
hold on
plot(freq,db(G_id), 'b');
hold on
plot(freq,db(G-G_id), 'k')
xlabel('Frequency (Hz)')
ylabel('Amplitude (dB)')
zoom on;
shg

% Comparison identified model and true model 
figure(5)
bode(G13,opts)
hold on
bode(sys_id,opts)
hold on
bode(G13-sys_id,opts)



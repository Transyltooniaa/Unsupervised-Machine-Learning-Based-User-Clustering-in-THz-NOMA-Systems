clear all;
N = 10;
K = 4;
% M = 5;
Rk = 1;
rhoP = 1; % primary users' transmit power
sigma2_dbm = -90; % Thermal noise in dBm
sigma = 10^((sigma2_dbm-30)/10);
Pmax = rhoP;  
xi = 0; %100000000; %1.0000e+15;
taux = 0.0000001;
tauy = 50;
epsilon_cbv = 0.01;
al = 2; % path loss exponent
c0 = 3*10^8;
fc = 300*10^9;
rl = 5*exp(-3); % molecular absorption
ct = 5000;
eps_BB = 0.1;
max_itr = 200;

Mvec = [1 2 4 6 8];
%Kvec = [1 3 5 7 10];
for iM = 1:length(Mvec)
    M = Mvec(iM);
    weight = ones(1, M);
    sum4 = 0;
    
    % Initialize storage for rates over Monte Carlo iterations
    primary_rate_sim = zeros(1, ct);
    secondary_rate_method1 = zeros(1, ct);
    secondary_rate_method2 = zeros(1, ct);
    
    for ict = 1:ct
        %% Primary users' channels
        akP = complex(sqrt(0.5)*randn(K, 1), sqrt(0.5)*randn(K, 1));
        rP = 10; % size of the square
        coodP = [rP*rand(K,1).*sign(randn(K,1)) , rP*rand(K,1).*sign(randn(K,1))]; % locations
        rkP = sqrt(coodP(:,1).^2 + coodP(:,2).^2);
        PLkP = (c0/4/pi/fc)^2 * exp(-rl*rkP) ./ (1+rkP.^al);
        scale = max(1./PLkP); % scaling factor for optimization
        
        thetak = -pi/2 + pi/K * (1:K)'; % angle of departure for primary users
        NN = (0:N-1)';
        H = exp(-1i*pi*sin(thetak)'.*NN) * (diag(akP) .* sqrt(PLkP)); % channel matrix, N-by-K
        
        %% Build the codebook and analog beamforming
        NQ = 10; % number of codewords/beams in the codebook
        theta_vector = pi*(0:NQ-1)/NQ - pi/2;
        ABF = []; beam = [];
        for i = 1:K
            [~, indexk] = min(abs(thetak(i) - theta_vector));
            ABF(:, i) = exp(-1i*pi*sin(theta_vector(indexk)) * NN) / sqrt(N);
            theta_vector(indexk) = []; % avoid reusing the same beam
        end
        
        %% Digital beamforming
        Habf = H' * ABF; % channel matrix after analog beamforming
        D_power = diag(1./diag(inv(Habf'*Habf)));
        DBF = inv(Habf) * sqrt(D_power); % normalized digital beamforming
        for k = 1:K
            beam(:, k) = ABF * DBF(:, k);
        end
        % ABF = beam; % Uncomment if composite precoding is desired
        
        %% Secondary users' channels
        akS = complex(sqrt(0.5)*randn(M,1), sqrt(0.5)*randn(M,1));
        rS = 10; % size of square for secondary users
        coodS = [rS*rand(M,1).*sign(randn(M,1)) , rS*rand(M,1).*sign(randn(M,1))];
        rkS = sqrt(coodS(:,1).^2 + coodS(:,2).^2);
        PLkS = (c0/4/pi/fc)^2 * exp(-rl*rkS) ./ (1+rkS.^al);
        thetak_s = pi*rand(M,1) - pi/2;
        G = exp(-1i*pi*sin(thetak_s)'.*NN) * (diag(akS) .* sqrt(PLkS)); % channel matrix, N-by-M
        
        %% Generate h^P_ki (for primary users)
        hPki = zeros(K, K);
        for k = 1:K
            hPki(k,:) = abs(H(:,k)' * ABF).^2; % each row is for primary user k
        end
        
        % Compute primary users' sum rate (approximated SINR)
        primary_sum_rate = sum(log2(1 + diag(hPki)*rhoP/sigma));
        primary_rate_sim(ict) = primary_sum_rate;
        
        %% Generate h^S_jk (for secondary users)
        hSjk = zeros(M, K);
        for j = 1:M
            hSjk(j,:) = abs(G(:,j)' * ABF).^2; % each row corresponds to a secondary user
        end
        
        %% Generate H^P_kk and related variables for primary constraints
        for k = 1:K
            HPkk(k) = sum(hPki(k,:)) - hPki(k,k);
            ck(k) = (sum(hPki(k,:)) - hPki(k,k)) / hPki(k,k) - rhoP/(2^Rk - 1) + sigma/hPki(k,k);
        end
        
        %% Generate H^S_jk, bjk, and tjk for secondary users
        for j = 1:M
            for k = 1:K
                HSjk(j,k) = sum(hSjk(j,:)) - hSjk(j,k);
                bjk(j,k) = (sum(hSjk(j,:)) - hSjk(j,k)) / hSjk(j,k) * rhoP - rhoP/(2^Rk - 1) + sigma/hSjk(j,k);
                tjk(j,k) = (sum(hSjk(j,:)) - hSjk(j,k)) * rhoP + sigma;
            end
        end
        
        %% Algorithm initialization and secondary user scheduling
        structure = zeros(M, K);
        for j = 1:M
            for k = 1:K
                if bjk(j,k) <= 0 && ck(k) <= 0
                    structure(j,k) = 1;
                end
            end
        end
        
        % Build the scheduled indices matrix S and force S to be two-column
        [row, col] = find(structure);
        if isempty(row) || isempty(col)
            continue;
        end
        S = [row(:), col(:)];  % Each row: [secondary_user_index, primary_beam_index]
        number_sij = size(S, 1);
        if number_sij == 0  % no beam available
            continue;
        end
        
        %% Generate the matrices to recover each row
        R = [];
        Rev_mat = zeros(M, number_sij, K);
        for k = 1:K
            for n = 1:number_sij
                if S(n,2) == k
                    Rev_mat(S(n,1), n, k) = 1;
                end
            end    
            R = [R; Rev_mat(:,:,k)];
        end
        
        %% Generate cp, dp, ep, fp and dpx
        cp = [];
        dp = [];
        ep = [];
        fp = [];
        dpx = [];
        for p = 1:number_sij
            cp = [cp, [zeros(1, p-1), hSjk(S(p,1), S(p,2)), zeros(1, number_sij-p)]'];
            
            dpp = [];
            dppx = [];
            for k = 1:K        
                if k == S(p,2)
                    onesp2 = ones(M,1); onesp2(S(p,1)) = 0;
                    dpp = [dpp; xi * hSjk(S(p,1), k) * onesp2];
                    onesp2x = zeros(M,1);
                    dppx = [dppx; onesp2x];
                else
                    dpp = [dpp; hSjk(S(p,1), k) * ones(M,1)];
                    dppx = [dppx; hSjk(S(p,1), k) * ones(M,1)];
                end
            end            
            dp = [dp, dpp];
            dpx = [dpx, dppx];
            
            epp = [];
            for k = 1:K        
                if k == S(p,2)
                    zerosp2 = zeros(M,1); zerosp2(S(p,1)) = 1;
                    epp = [epp; zerosp2];
                else
                    epp = [epp; hPki(S(p,2), k) / hPki(S(p,2), S(p,2)) * ones(M,1)];
                end
            end            
            ep = [ep, epp];
            
            fpp = [];
            for k = 1:K        
                if k == S(p,2)
                    zerosp2 = zeros(M,1); zerosp2(S(p,1)) = 1;
                    fpp = [fpp; zerosp2];
                else
                    fpp = [fpp; hSjk(S(p,1), k) / hSjk(S(p,1), S(p,2)) * ones(M,1)];
                end
            end            
            fp = [fp, fpp];
        end
        
        %% Initialization for both algorithms (secondary scheduling)
        tempy = zeros(number_sij,1);
        for p = 1:number_sij
            tempy(p) = hSjk(S(p,1), S(p,2));
        end
        [~, indy] = max(tempy);
        ym = zeros(2*number_sij, 1);
        ym(indy) = Pmax;
                
        %% Secondary Scheduling (Method 1)
        ybas = ym(1:number_sij);
        ybas(indy) = min([Pmax; -ck(S(indy,2)); -bjk(S(indy,1),S(indy,2))]);
        secondary_rate_method1(ict) = frate(ybas, number_sij, cp, dpx, R, tjk, S);
                
        %% Secondary Scheduling (Method 2)
        tempz = zeros(number_sij,1);
        for p = 1:number_sij
            rhotemp = min([Pmax; -ck(S(p,2)); -bjk(S(p,1),S(p,2))]);
            tempz(p) = log2(1 + hSjk(S(p,1), S(p,2)) * rhotemp / tjk(S(p,1), S(p,2)));
        end
        [~, indz] = max(tempz);
        ymz = zeros(number_sij,1);
        ymz(indz) = min([Pmax; -ck(S(indz,2)); -bjk(S(indz,1),S(indz,2))]);
        secondary_rate_method2(ict) = frate(ymz, number_sij, cp, dpx, R, tjk, S);
        
        [iM, ict]  % display progress
    end 
    Ratebas(iM) = mean(secondary_rate_method1);
    Ratebas_new(iM) = mean(secondary_rate_method2);
    PrimaryRate(iM) = mean(primary_rate_sim);
end

%% Plotting Primary Users' Sum Rate vs. Number of Secondary Users
figure;
plot(Mvec, PrimaryRate, 'k-*', 'LineWidth', 2);
xlabel('Number of Secondary Users (M)');
ylabel('Average Primary Sum Rate (bps/Hz)');
title('Primary Users: Sum Data Rate vs. Number of Secondary Users');
grid on;

%% Plotting Secondary Users' Sum Rate vs. Number of Secondary Users
figure;
plot(Mvec, Ratebas, 'b-o', 'LineWidth', 2);
hold on;
plot(Mvec, Ratebas_new, 'r--', 'LineWidth', 2);
xlabel('Number of Secondary Users (M)');
ylabel('Average Secondary Sum Rate (bps/Hz)');
title('Secondary Users: Sum Data Rate vs. Number of Secondary Users');
legend('Scheduling Method 1', 'Scheduling Method 2');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the rate function frate(x)
function [ratex] = frate(y, xsize, cp, dpx, R, tjk, S)
    ratex = 0;
    for i = 1:xsize
        if y(i) > 0.01
            ratex = ratex + log2(1 + cp(:,i)' * y / (dpx(:,i)' * R * y + tjk(S(i,1), S(i,2))));
        end
    end
end

% Define the objective function f(x)
function [f] = fobj(xtilde, weight, S)
    f = 0;
    for i = 1:length(xtilde)
        f = f + log2(1 + xtilde(i));
    end 
end

% Define the SCA objective function
function [f] = sca_obj(yz, cp, dp, R, tjk, weight, S, scale)
    f = 0;    
    xsize = length(yz)/2;
    y = yz(1:xsize); 
    z = yz(xsize+1:end);
    for p = 1:xsize
        f = f + log2(exp(1)) * log(scale * (cp(:,p)' + dp(:,p)' * R * y) + scale * tjk(S(p,1), S(p,2))) - log(scale) - z(p);
    end  
end

% Bisection method to find the projection on G 
function [temp] = proj(Box, Pmax, S, cp, dp, ep, fp, tjk, bjk, ck, R)
     zk = Box(:,2); % upper corner
     xmin = Box(:,1); % lower corner
     amin = 0; 
     amax = 1;
     delta = 0.001;
     while amax - amin >= delta
         alpha = (amin + amax)/2;
         if alpha * zk >= xmin  % projection is inside of Box
             [temp1, y] = feasibility(alpha * zk, Pmax, S, cp, dp, ep, fp, tjk, bjk, ck, R);
             if ~temp1 % not feasible  
                amax = alpha;
             else % feasible 
                amin = alpha;
             end     
         else
             amin = alpha;
         end
    end
    temp = amin * zk;
end
 
% Bisection method to check feasibility 
function [temp, y] = feasibility(x, Pmax, S, cp, dp, ep, fp, tjk, bjk, ck, R)
    xsize = length(x);
    A = zeros(3*xsize, xsize);
    b = zeros(3*xsize, 1);
    for i = 1:xsize       
        A(3*(i-1)+1, :) = - (cp(:,i)' - x(i)*dp(:,i)' * R) / tjk(S(i,1), S(i,2));
        b(3*(i-1)+1) = -x(i);
        A(3*(i-1)+2, :) = ep(:,i)' * R;
        b(3*(i-1)+2) = -ck(S(i,2));
        A(3*(i-1)+3, :) = sign(x(i)) * fp(:,i)' * R;
        b(3*(i-1)+3) = -sign(x(i)) * bjk(S(i,1), S(i,2));
    end
    A = [A; ones(1, xsize)];
    b = [b; Pmax];    
    Aeq = [];
    beq = [];
    lb = zeros(xsize, 1);
    ub = [];
    options = optimoptions('linprog','Display','off');
    y = linprog([], A, b, Aeq, beq, lb, ub, options);
    if isempty(y)
        temp = 0; % feasible
    else
        temp = 1; % infeasible
    end
end

% Finding a tighter lower bound  
function [xnew] = flow_bound(xbox, number_sij, cp, ep, R, ck, fp, bjk, Pmax, tjk, S, dp)
    xmin = xbox(1:number_sij, 1);
    xmax = xbox(1:number_sij, 2);
    for p = 1:number_sij
        if xmax(p) == 0
            xnew(p) = 0;
            continue;
        end
        A = zeros(2*number_sij, number_sij);
        b = zeros(2*number_sij, 1);
        for i = 1:number_sij      
            A(2*(i-1)+1, :) = ep(:,i)' * R;
            b(2*(i-1)+1) = -ck(S(i,2));
            if i == p
                A(2*(i-1)+2, :) = fp(:,i)' * R;
                b(2*(i-1)+2) = -bjk(S(i,1), S(i,2)); 
            else
                A(2*(i-1)+2, :) = sign(xmin(i)) * fp(:,i)' * R;
                b(2*(i-1)+2) = -sign(xmin(i)) * bjk(S(i,1), S(i,2));
            end
        end
        A = [A; ones(1, number_sij)];
        b = [b; Pmax];    
        A = [A; (cp(:,p)' - xmax(p)*dp(:,p)' * R) / tjk(S(p,1), S(p,2))];
        b = [b; xmax(p)];        
       
        Aeq = [] ;
        beq = []; 
        for i = 1:number_sij      
            if i == p
                continue;
            end
            Aeq = [Aeq; (cp(:,i)' - xmin(i)*dp(:,i)' * R) / tjk(S(i,1), S(i,2))];
            beq = [beq; xmin(i)];
        end
                
        if number_sij > 1
            warning('off')
            for ia = 1:size(Aeq,1)
                anorm = max(abs(Aeq(ia, :)));
                if anorm == 0
                    continue;
                end
                Aeq(ia, :) = Aeq(ia, :) / anorm;
                beq(ia) = beq(ia) / anorm;
            end
            Aeq_tilde = Aeq;  
            Aeq_tilde(:, p) = [];
            Aeqinv = inv(Aeq_tilde);
            svd_A = svd(Aeq_tilde);
            if min(svd_A) < 0.0001
                lb = zeros(number_sij, 1);
                ub = [];
                y0 = xmin;
                options = optimoptions('fmincon','Display','off');
                y = fmincon(@(y) -cp(:,p)'/tjk(S(p,1), S(p,2))*y/(dp(:,p)'*R/tjk(S(p,1), S(p,2))*y+1), y0, A, b, Aeq, beq, lb, ub, [], options);
                xnew2(p) = cp(:,p)'/tjk(S(p,1), S(p,2))*y/(dp(:,p)'*R/tjk(S(p,1), S(p,2))*y+1);
            else
                A_tilde = A;  
                A_tilde(:, p) = [];  
                app = A(:, p);
                aeqpp = Aeq(:, p);
                vec_temp1 = b - A_tilde * Aeqinv * beq;
                vec_temp2 = app - A_tilde * Aeqinv * aeqpp;
                index_min = find(vec_temp2 > 0);
                index_max = find(vec_temp2 < 0);
                upper_bound = vec_temp1(index_min) ./ vec_temp2(index_min);
                lower_bound = vec_temp1(index_max) ./ vec_temp2(index_max);
                yp = min(upper_bound);
                ynew_tilde = Aeqinv * (beq - aeqpp * yp);
                ynew = [ynew_tilde(1:p-1); yp; ynew_tilde(p:end)];
                xnew2(p) = cp(:,p)'/tjk(S(p,1), S(p,2))*ynew/(dp(:,p)'*R/tjk(S(p,1), S(p,2))*ynew+1);
            end
            xnew(p) = xnew2(p);
        else
            ynew = min(b./A);
            xnew(p) = cp(:,p)'/tjk(S(p,1), S(p,2))*ynew/(dp(:,p)'*R/tjk(S(p,1), S(p,2))*ynew+1);
        end
        xnew(p) = max(0, xnew(p));
    end
end

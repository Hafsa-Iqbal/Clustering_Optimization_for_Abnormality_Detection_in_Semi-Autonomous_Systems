% % (OPTIMIZATION) AT DIFFERENT VALUES OF K.  FINDING
% % THE MINIMUM POINT WHICH WILL PROVIDE US OPTIMAL NUMBER OF CLUSTERS
clc
clear
close all
numData = 1;
curDir = pwd;
plotMap = true;

%% load data
load('PMDatafile.mat')
InputData =[structSyncData.Filtered.xPos, structSyncData.Filtered.yPos,...
    structSyncData.Filtered.divxPos, structSyncData.Filtered.divyPos];
title = 'PM Pose';
strSave = 'PM Pose';

%% GNG parameters
params.N = 100;                                                              %    Number of nodes
params.MaxIt = 10;                                                           %    Iteration (repetition of input data)
params.L_growing = 1000;                                                     %    Growing rate
params.epsilon_b = 0.05;                                                     %    Movement of winner node
params.epsilon_n = 0.0006;                                                   %    Movement of all other nodes except winner
params.alpha = 0.5;                                                          %    Decaying global error and utility
params.delta = 0.9995;                                                       %    Decaying local error and utility
params.T = 100;                                                              %    It could be a function of params.L_growing, e.g., params.LDecay = 2*params.L_growing
params.L_decay = 1000;                                                       %    Decay rate sould be faster than the growing then it will remove extra nodes
params.alpha_utility = 0.0005;                                               %    It could be a function of params.delta, e.g., params.alpha_utility = 0.9*params.delta


%% optimization

% initialization of variables
var_k_input = [];
var_k_comp = [];
seedstore = [];
matrixStore = [];
K_Value = [];
u = 0;
store_min = [];
storeNodes=[];
store_kposition =[];
% global loop for the randomization of data
for seedvector = 1:1:50
    u = u + 1;
    F_k = [];
    cycle = 0;
    params.seedvector  = seedvector;
    nodesStore = [];
    % locally optimal step size
    for  k =0:0.01:0.28
        params.k = k;
        net = GrowingNeuralGasNetwork(InputData, params);
        %% variance of  data
        varianceInputData = [];
        N = size(net.datanodesNorm,2);                                      % number of nodes at each k
        K_Value = [K_Value; k];
        cycle = cycle + 1;
        for i = 1:N           % normalize and ordered data
            tempInputData = cell2mat(net.datanodesNorm(1,i));               % temprary vector having data samples of a node
            var_InputData = var(tempInputData);                             % variance of first derivative
            varianceInputData =[varianceInputData ; var_InputData];
        end
        var_k_input{cycle} = varianceInputData;
        % LOSS FUNCTION
        loss_function = (sigmoid(varianceInputData(:,1)) + (1 - sigmoid(varianceInputData(:,3))) +...
            sigmoid(varianceInputData(:,2)) + (1 - sigmoid(varianceInputData(:,4))) ); % Eq.(5)
        
        avg_loss_function = sum(loss_function)./N;                          %  Eq.(6) averaging of loss function
        F_k = [F_k; avg_loss_function];                                     %  store F_k values in last seed value
        nodesStore = [nodesStore N];
        
        % PLOT RESULTS
        figure(3)
        plot(F_k)
        hold on
        
    end    % end of local loop
    
    info{u} = nodesStore;
    matrixStore = [matrixStore ,F_k] ;                                      % store k values at last seed position
    
    [minF_k, associated_k_position] =  min(F_k);                            % Eq.(7), minimum of loss vector
    store_min = [store_min; minF_k];
    store_kposition = [store_kposition; associated_k_position];
    plotresult{u} = matrixStore;
    
    
end    % end of global loop

[Fmin, k_position]  = min(store_min);
kOptimized = store_kposition(k_position)  ;
%% error plot
k =[0:0.01:0.28]; % set the range of K values same as the local loop
x = (1:size(matrixStore,1))';
y = mean(matrixStore,2);
e = std(matrixStore,1,2);
figure(12)
errorbar(k,y,e,'rx');



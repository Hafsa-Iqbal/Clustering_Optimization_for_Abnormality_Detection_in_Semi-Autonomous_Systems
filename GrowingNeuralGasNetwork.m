%% GNG function
function net = GrowingNeuralGasNetwork(inputData, params, plotFlag)
% function net = GrowingNeuralGasNetwork(inputData, compData, params, plotFlag)
%% GNG parameters
N = params.N;                                                               % number of nodes in GNG
MaxIt = params.MaxIt;
L_growing = params.L_growing;
epsilon_b = params.epsilon_b;                                               % stepsize to update weight of winner node
epsilon_n = params.epsilon_n;                                               % stepsize to update weight of all direct neibours
alpha = params.alpha;
delta = params.delta;
T = params.T;
k = params.k;
L_decay = params.L_decay;
seedvector  = params.seedvector;
%
if ~exist('PlotFlag', 'var')
    plotFlag = true;
end

% Normalize input  data
minDataNorm = min(inputData);
dataNorm = inputData - repmat(minDataNorm,size(inputData,1),1);
maxDataNorm = max(dataNorm);
inputNorm = dataNorm./repmat(maxDataNorm,size(inputData,1),1);
inputNormOrd = inputNorm;

% Load Data
nData = size(inputNorm,1);                                                  % Size of input data (number of training samples)
nDim = size(inputNorm,2);                                                   % Dimension of input data
rng(seedvector)
inputNorm = inputNorm(randperm(nData), :);                                  % Random permutation of input data

CVmin = min(inputNorm);
CVmax = max(inputNorm);

% Initialization

Ni = 2;                                                                     % Initial 2 nodes for training the algorithm

wNorm = zeros(Ni, nDim);
for i = 1:Ni
    wNorm(i,:) = unifrnd(CVmin, CVmax);                                     % It returns an array of random numbers generated from the continuous uniform distributions with lower and upper endpoints specified by 'CVmin' and 'CVmax'.
end

E = zeros(Ni,1);
utility = ones(Ni,1);
C = zeros(Ni, Ni);
t = ones(Ni, Ni);

% Loop
p1store =[];
nx = 0;
f=0;
for it = 1:MaxIt
    f =1;
    
    for c = 1:nData                                                         %   Number of data
        % Select Input
        nx = nx + 1;                                                        %   Counter of cycles inside the algorithm
        x = inputNorm(c,:);                                                 %   pick first input vector from permuted inputs
        
        % Competion and Ranking
        
        
        d = pdist2(x, wNorm,'euclidean');                                   %   pdist- Pairwise distance between two sets of observations(Eucledian distance between input and 2 nodes initialised before)
        %         d = [];
        %         for wi =1:1:size(wNorm,1)
        %             DistN = dist_alpha(x,wNorm(wi,:),alphaN,betaN);
        %             d = [d, DistN];
        %         end
        [~, SortOrder] = sort(d);                                           %   Organize distances between nodes and the first data point in an ascending order
        
        s1 = SortOrder(1);                                                  %   Closest node index to the first data point
        s2 = SortOrder(2);                                                  %   Second closest node index to the first data point
        
        % Aging
        t(s1, :) = t(s1, :) + 1;                                            %   Increment the age of all edges emanating from s1
        t(:, s1) = t(:, s1) + 1;
        
        % Add Error
        dist0  = d(s1)^2;
        dist1  = d(s2)^2;
        E = E + dist0;    % E = E+d';    but the right thing is E(s1) = E(s1) + dist0;
        
        % utility
        deltaUtility =  dist1 - dist0;                                      % Initial utility is zero in first case and dist is the error of first node
        utility(s1) =  utility(s1) + deltaUtility ;                         % Difference between error of two nodes
        
        
        % Adaptation
        wNorm(s1,:) = wNorm(s1,:) + epsilon_b*(x-wNorm(s1,:));              %   Move the nearest distance node and it's neibors to wards input signal by fractions Eb and En resp.
        Ns1 = find(C(s1,:) == 1);                                           %   Take all the connections of the closest node to the data in question
        for j = Ns1
            wNorm(j,:) = wNorm(j,:) + epsilon_n*(x-wNorm(j,:));             %   Move the direct topological neibors of nearest distance node (S1) and it's neibors to wards input signal by fractions Eb and En resp.
        end
        
        % Create Link
        C(s1,s2) = 1;                                                       %  If s1 and s2 are connected by an edge , set the age of this edge to zero , it such edge doesn't exist create it
        C(s2,s1) = 1;
        t(s1,s2) = 0;                                                       %   Age of the edge
        t(s2,s1) = 0;
        % Remove Old Links
        C(t > T) = 0;                                                       %   remove edges with an age larger than Amax(a threshold value)
        nNeighbor = sum(C);                                                 %   Number of conecctions of each node
        AloneNodes = (nNeighbor==0);
        C(AloneNodes, :) = [];
        C(:, AloneNodes) = [];
        t(AloneNodes, :) = [];
        t(:, AloneNodes) = [];
        wNorm(AloneNodes, :) = [];
        E(AloneNodes) = [];
        utility(AloneNodes) = [];
        
        
        if   size(wNorm,1) < N  && mod(nx, L_growing) == 0  %&&  ( xx < th)
            
            [~, q] = max(E);                                            %   Determine the unit q with the maximum accumulated error
            [~, f] = max(C(:,q).*E);                                        %   Maximum index related to the error related to a connected node
            
            r = size(wNorm,1) + 1;                                          %   Total number of nodes
            wNorm(r,:) = (wNorm(q,:) + wNorm(f,:))/2;                       %   Insert a new unit r halfway between q and it's neibor f with the largest error variable
            
            %   Remove old connections and introduce the presence of the
            %   new created node
            C(q,f) = 0;
            C(f,q) = 0;
            C(q,r) = 1;
            C(r,q) = 1;
            C(r,f) = 1;
            C(f,r) = 1;
            t(r,:) = 0;
            t(:, r) = 0;
            
            E(q) = alpha*E(q);                                              %   Decrease the error variable of q and f by multiplying them with a constand 'alpha'
            E(f) = alpha*E(f);
            E(r) = 0.5 *( E(q) + E(f) );                                    %   Initialize the error of the new node equal to error of the winner node
            
            utility(r) = 0.5 *( utility(q) + utility(f) );
        end
        
        %
        if mod(nx, L_decay) == 0
            
            [max_E, ~] = max(E);                                            % Maximum accumelated error
            
            [min_utility,node_useless] = min(utility);                      % Node node_useless having minimum utility
            
            CONST = min_utility * k;                                        % Utility factor
            
            if (CONST < max_E)
                C(node_useless,:) = [];                                     % Remove the connection having smaller utility factor                                     % Remove the node having smaller utility factor
                C(:,node_useless) = [];
                wNorm(node_useless,:) = [];                                 % Remove the node having smaller utility factor
                utility(node_useless) = [];                                 % Remove the min utility value from the utility vector
                E(node_useless) = [];                                       % Remove error vector correspond to the node having min utility
                t(node_useless,:) = [];                                     % Remove agging vector correspond to the node having min utility
                t(:, node_useless) = [];
                
            end
            
        end  % end of decaying loop
        
        % Decrease Errors
        E = delta * E;                                                      % Decrease error variables by multiplying them with a constant delta
        utility = delta*utility;                                            % Decrease the utility by alpha_utility constant
        storeE{f}= E;
        f = f+1;
        
    end     % end of data samples loop
    for i = 1:MaxIt
        
        p1 = mean(storeE{it});
        p1store = [p1store, p1];
        
    end
    %% plot clustering results
    if plotFlag
        GlobalIteration = it;
        figure(7)
        PlotResults(inputNorm, wNorm, C);
        pause(0.01);
    end
    
end  % end of iteration loop



figure(9)

plot(p1store)
title('mean of global error')
xlabel('iteration')
ylabel('error')
hold on
%% Export Results data samples in nodes
datanodesNorm = cell(1,size(wNorm,1));
datanodes = cell(1,size(wNorm,1));
dataColorNode = [];

for c = 1:nData                                                             %   Number of data
    x = inputNormOrd(c,:);
    d = pdist2(x, wNorm,'euclidean');                                       %   pdist- Pairwise distance between two sets of observations(Eucledian distance between input and 2 nodes initialised before)
    [~, minNode] = min(d);                                                  %   Organize distances between nodes and the first data point in an ascending order
    dataColorNode = [dataColorNode; minNode];
    datanodesNorm{1,minNode} = [datanodesNorm{1,minNode}; x];               % normalize and ordered data
    
    x = inputData(c,:);      %not normalize and orderd data
    datanodes{1,minNode} = [datanodes{1,minNode}; x];
    
end


%% Denormalization of GNGN generated centroids
w = wNorm.*repmat(maxDataNorm,size(wNorm,1),1);
w = w + repmat(minDataNorm,size(wNorm,1),1);

%%  Calculation of mean and covariances of generated nodes
nodesMean = [];
nodesMeanNorm = [];
for i = 1:size(datanodes,2)
    %   Calculation of mean values
    nodesMean = [nodesMean; mean(datanodes{1,i})];
    nodesMeanNorm = [nodesMeanNorm; mean(datanodesNorm{1,i})];
    
    %   Calculation of covariance values
    nodesCov{1,i} = cov(datanodes{1,i});
    nodesCovNorm{1,i} = cov(datanodesNorm{1,i});
    
    %   Calculation of radius of acceptances
    nodesRadAccept(1,i) = sqrt(sum((3*std(datanodes{1,i})).^2));
    nodesRadAcceptNorm(1,i) =sqrt(sum((3*std(datanodesNorm{1,i})).^2));
    % sqrt(sum(nodesMeanNorm(i,:))+ sum((3*std(datanodesNorm{1,i})).^2))
end

%% outputs of GNG

%   Data inside each node
net.datanodes  = datanodes;
net.datanodesNorm  = datanodesNorm ;

net.N = size(datanodes,2);  %   number of nodes
net.wNorm = wNorm;   % protocol values
net.w = w;   % protocol values (calculated centroids)
net.E = E;   % error value
net.C = C;   % connections
net.t = t;   % connection values
net.Parameters = params;
net.dataColorNode = dataColorNode;

%   Nodes' mean values and covariances
net.nodesMean = nodesMean;
net.nodesMeanNorm = nodesMeanNorm;
net.nodesCov= nodesCov;
net.nodesCovNorm = nodesCovNorm;

%   Nodes' radius of acceptances
net.nodesRadAccept = nodesRadAccept;
net.nodesRadAcceptNorm = nodesRadAcceptNorm;
net.dataNorm = inputNormOrd;
net.data = inputData;

end
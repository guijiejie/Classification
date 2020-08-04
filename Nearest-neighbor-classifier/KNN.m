function rate = KNN(Train_data,Train_label,Test_data,Test_label,k,Distance_mark);
% K-Nearest-Neighbor classifier(K-NN classifier)
%Input:
%     Train_data,Test_data are training data set and test data
%     set,respectively.(Each row is a data point)
%     Train_label,Test_label are column vectors.They are labels of training
%     data set and test data set,respectively.
%     k is the number of nearest neighbors
%     Distance_mark           :   ['Euclidean', 'L2'| 'L1' | 'Cos'] 
%     'Cos' represents Cosine distance.
%Output:
%     rate:Accuracy of K-NN classifier
%
%    Examples:
%      
% %Classification problem with three classes
% A = rand(50,300);
% B = rand(50,300)+2;
% C = rand(50,300)+3;
% % label vector for the three classes
% gnd = [ones(300,1);2*ones(300,1);3*ones(300,1)];
% fea = [A B C]';
% trainIdx = [1:150,301:450,601:750]';
% testIdx = [151:300,451:600,751:900]';
% fea_Train = fea(trainIdx,:);
% gnd_Train = gnd(trainIdx);
% fea_Test = fea(testIdx,:);
% gnd_Test = gnd(testIdx);
% rate = KNN(fea_Train,gnd_Train,fea_Test,gnd_Test,1)
%
%
%
%Reference:
%
% If you used my matlab code, we appreciate it very much if you can cite our following papers:
% Jie Gui, Tongliang Liu, Dacheng Tao, Zhenan Sun, Tieniu Tan, "Representative Vector Machines: A unified framework for classical classifiers", IEEE  
% Transactions on Cybernetics (Accepted).
% Jie Gui et al., "Group sparse multiview patch alignment framework with view consistency for image classification", IEEE Transactions on Image Processing, vol. 23, no. 7, pp. 3126-3137, 2014
% Jie Gui et al., "How to estimate the regularization parameter for spectral regression
% discriminant analysis and its kernel version?", IEEE Transactions on Circuits and 
% Systems for Video Technology, vol. 24, no. 2, pp. 211-223, 2014
% Jie Gui, Zhenan Sun, Wei Jia, Rongxiang Hu, Yingke Lei and Shuiwang Ji, "Discriminant
% Sparse Neighborhood Preserving Embedding for Face Recognition", Pattern Recognition, 
% vol. 45, no.8, pp. 2884¨C2893, 2012
% Jie Gui, Wei Jia, Ling Zhu, Shuling Wang and Deshuang Huang, 
% "Locality Preserving Discriminant Projections for Face and Palmprint Recognition," 
% Neurocomputing, vol. 73, no.13-15, pp. 2696-2707, 2010
% Jie Gui et al., "Semi-supervised learning with local and global consistency", 
% International Journal of Computer Mathematics (Accepted)
% Jie Gui, Shu-Lin Wang, and Ying-ke Lei, "Multi-step Dimensionality Reduction and 
% Semi-Supervised Graph-Based Tumor Classification Using Gene Expression Data," 
% Artificial Intelligence in Medicine, vol. 50, no.3, pp. 181-191, 2010
    
%This code is written by Gui Jie in the evening 2009/03/11.
%If you have find some bugs in the codes, feel free to contract me
if nargin < 5
    error('Not enought arguments!');
elseif nargin < 6
    Distance_mark='L2';
end
 
[n dim]    = size(Test_data);% number of test data set
train_num  = size(Train_data, 1); % number of training data set
% Normalize each feature to have zero mean and unit variance.
% If you need the following four rows,you can uncomment them.
% M        = mean(Train_data); % mean & std of the training data set
% S        = std(Train_data);
% Train_data = (Train_data - ones(train_num, 1) * M)./(ones(train_num, 1) * S); % normalize training data set
% Test_data            = (Test_data-ones(n,1)*M)./(ones(n,1)*S); % normalize data
U        = unique(Train_label); % class labels
nclasses = length(U);%number of classes
Result  = zeros(n, 1);
Count   = zeros(nclasses, 1);
dist=zeros(train_num,1);
for i = 1:n
    % compute distances between test data and all training data and
    % sort them
    test=Test_data(i,:);
    for j=1:train_num
        train=Train_data(j,:);V=test-train;
        switch Distance_mark
            case {'Euclidean', 'L2'}
                dist(j,1)=norm(V,2); % Euclead (L2) distance
            case 'L1'
                dist(j,1)=norm(V,1); % L1 distance
            case 'Cos'
                dist(j,1)=acos(test*train'/(norm(test,2)*norm(train,2)));     % cos distance
            otherwise
                dist(j,1)=norm(V,2); % Default distance
        end
    end
    [Dummy Inds] = sort(dist);
    % compute the class labels of the k nearest samples
    Count(:) = 0;
    for j = 1:k
        ind        = find(Train_label(Inds(j)) == U); %find the label of the j'th nearest neighbors 
        Count(ind) = Count(ind) + 1;
    end% Count:the number of each class of k nearest neighbors
    
    % determine the class of the data sample
    [dummy ind] = max(Count);
    Result(i)   = U(ind);
end
correctnumbers=length(find(Result==Test_label));
rate=correctnumbers/n;

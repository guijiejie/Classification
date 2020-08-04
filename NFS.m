function rate = NFS(Train_data,Train_label,Test_data,Test_label);
% Nearest feature subspace classifier(NFS classifier)
%
%Input:
%     Train_data,Test_data are training data set and test data
%     set,respectively.(Each row is a data point)
%     Train_label,Test_label are column vectors.They are labels of training
%     data set and test data set,respectively.
%Output:
%     rate:Accuracy of Nearest feature subspace classifier

%	Reference:
%   
%     I. Naseem, et al., "Linear Regression for Face Recognition," IEEE  Transactions on Pattern Analysis and Machine Intelligence, vol. 32, pp. 2106-2112, Nov 2010.
% 
%     J. T. Chien and C. C. Wu, "Discriminant waveletfaces and nearest feature classifiers for face recognition," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, pp. 1644-1649, Dec 2002.              

%This code is written by Gui Jie in the morning 2011/01/07.
%If you have find some bugs in the codes, feel free to contract me

[test_num dim]    = size(Test_data);% test_num denotes the number of test data set
train_num  = size(Train_data, 1); % number of training data set

% Normalize each feature to have zero mean and unit variance.
% If you need the following four rows,you can uncomment them.
% M        = mean(Train_data); % mean & std of the training data set
% S        = std(Train_data);
% Train_data = (Train_data - ones(train_num, 1) * M)./(ones(train_num, 1) * S); % normalize training data set
% Test_data            = (Test_data-ones(n,1)*M)./(ones(n,1)*S); % normalize data

classLabel = unique(Train_label); % class labels
nclasses = length(classLabel);%number of classes
Result  = zeros(test_num, 1);

dist=zeros(test_num,nclasses);
for j = 1:nclasses
    index = find(Train_label==classLabel(j));
    %temp = (Train_data(index,:))'*inv(Train_data(index,:)*(Train_data(index,:))')*Train_data(index,:)-eye(dim);
    [U,V] = eig(Train_data(index,:)*(Train_data(index,:))');
    V = diag(V);
    V = 1./V;
    V = diag(V);
    temp = (Train_data(index,:))'*U*V*U'*Train_data(index,:)-eye(dim);
    for i = 1:test_num
        % compute distances between test data and each class
        test = (Test_data(i,:))';
        dis(i,j) = norm(temp*test);
    end
end

% determine the class of the data sample
for i = 1:test_num
    [dummy ind] = min(dis(i,:));
    Result(i)   = classLabel(ind);
end

correctnumbers=length(find(Result==Test_label));
rate=correctnumbers/test_num;

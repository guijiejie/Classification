function rate = NC(Train_data,Train_label,Test_data,Test_label);
% Nearest Center classifier(NC classifier)
%Input:
%     Train_data,Test_data are training data set and test data
%     set,respectively.(Each row is a data point)
%     Train_label,Test_label are column vectors.They are labels of training
%     data set and test data set,respectively.
%Output:
%     rate:Accuracy of NFP classifier
%This code is written by Gui Jie on the afternoon of 12/05/2012.
%If you have find some bugs in the codes, feel free to contract me

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

for k = 1:nclasses
    index = find(Train_label==U(k));
    MeanVector(k,:) = mean(Train_data(index,:));
end

for i = 1:n
    test = Test_data(i,:);
    temp = repmat(test,[nclasses 1]);
    dist =  sum((temp-MeanVector).^2,2);
    % determine the class of the data sample
    [dummy ind] = min(dist);
    Result(i) = U(ind);
end
correctnumbers=length(find(Result==Test_label));
rate=correctnumbers/n;

function rate = Fast_NFL(Train_data,Train_label,Test_data,Test_label,ratio);
% Fast Nearest Feature Line classifier(NFL classifier)
% For the datasets such as Action recognition database ASLAN, the number of
% each class is very large. The code 'ttt=combntns(1:length_index,3)' or
% 'ttt=combntns(1:length_index,2)' is out of memory. Our code randomly
% selects some combinations.
% The smaller ratio is, the faster our code is.
%Input:
%     Train_data,Test_data are training data set and test data
%     set,respectively.(Each row is a data point)
%     Train_label,Test_label are column vectors.They are labels of training
%     data set and test data set,respectively.
%Output:
%     rate:Accuracy of NFL classifier
%This code is written by Gui Jie in the afternoon 2012/12/12.
%The difference between NFL_V1 and NFL is that NFL_V1 uses less cycling.
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

for i = 1:n
    test=Test_data(i,:);
    dist = repmat(realmax,[nclasses 1]);
    for k = 1:nclasses
        index = find(Train_label==U(k));
        length_index = length(index);
        temp1 = randperm(length_index);
        temp2 = temp1(1:floor(length_index*ratio));
        ttt = combntns(temp2,2);
        %ttt=combntns(1:length_index,2);
        combination_num = size(ttt,1);
        for j = 1:combination_num
            temp = PointToLineDistance(test,Train_data(index(ttt(j,1)),:),Train_data(index(ttt(j,2)),:));
            if temp < dist(k,1)
                dist(k,1) = temp;
            end
        end
    end
    % determine the class of the data sample
    [dummy ind] = min(dist);
    Result(i) = U(ind);
end
correctnumbers=length(find(Result==Test_label));
rate=correctnumbers/n;

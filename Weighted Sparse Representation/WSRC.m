function [rate predictlabel] = WSRC( trnX ,trnY ,tstX , tstY )

% weighted SRC，　min_{alpha} ||diag(w)alpha||_1 
%           s.t. ||x-Dalpha||_2^2 <= lambda

% Input:
% trnX [dim * num ] - each column is a training sample
% trnY [ 1  * num ] - training label 
% tstX
% tstY

% Output:
% rate             - Recognition rate of test sample
% predictlabel     - predict label of test sample

ntrn = size( trnX , 2 ) ;
ntst = size( tstX , 2 ) ;
nClass = length( unique(trnY) ) ;

% normalize 
for i = 1 : ntrn
    trnX(:,i) = trnX(:,i) / norm( trnX(:,i) ) ;
end
for i = 1 : ntst
    tstX(:,i) = tstX(:,i) / norm( tstX(:,i) ) ;
end

sigma = 1.5;
W = SimilarityMatrix( trnX , tstX , sigma ) ;
% W = SimilarityMatrix( trnX , tstX ) ;

% classify
param.lambda = 0.0005 ; % not more than 20 non-zeros coefficients
%     param.numThreads=2; % number of processors/cores to use; the default choice is -1
% and uses all the cores of the machine
param.mode = 1 ;       % penalized formulation
param.verbose = false ;
A = mexLassoWeighted( tstX , trnX , W , param ) ;
[rate predictlabel] = Decision_Residual( trnX ,trnY ,tstX , tstY , A ) ;

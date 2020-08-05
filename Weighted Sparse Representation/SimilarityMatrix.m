function W = SimilarityMatrix( trnX , tstX , sigma )

% trnX  - each column is a sample
% tstX
% sigma
% for each column x_i of trnX, y_j of tstX, calculate
% W(i,j) = exp( ||x_i - y_j||^2 / sigma )

if nargin < 3
    sigma = 100 ;
end

ntrnX = size( trnX , 2 ) ;
ntstX = size( tstX , 2 ) ;

W = zeros( ntrnX , ntstX ) ; 
for j = 1 : ntstX
   for i = 1 : ntrnX
      W(i,j) = norm( trnX(:,i) - tstX(:,j) ) ;
%       W(i,j) = max( abs( trnX(:,i) - tstX(:,j) ) ) ;
%       W(i,j) = ( norm( trnX(:,i) - tstX(:,j) , 1 ) )^sigma ;
   end
end

W = exp( W / sigma ) ;      % WSRC_SPAMS
% W = sqrt( W.^2 + sigma^2 ) ;      % WSRC_SPAMS2
% W = W * sigma ;
% normalize
% for j = 1 : ntstX
%    W(:,j) = W(:,j) / max(W(:,j)) ;    
% %    W(:,j) = W(:,j) / sum(W(:,j)) ;
% %    W(:,j) = W(:,j) / norm(W(:,j)) ;
% end

% W = ones( ntrnX , ntstX ) ; 


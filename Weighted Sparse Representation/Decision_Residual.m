function [accuracy predictlabel] = Decision_Residual( fea_Train ,gnd_Train ,fea_Test , gnd_Test , Coeff_Test )


num_Test = size(fea_Test,2) ;
nClass = length( unique(gnd_Train) ) ;

R = zeros( nClass , num_Test ) ;
for k = 1 : nClass
    index = find( gnd_Train ~= k ) ;
    delta_Coeff = Coeff_Test ;
    delta_Coeff( index , : ) = 0 ;
    temR = fea_Test - fea_Train * delta_Coeff ;   
    for i = 1 : num_Test
        index = find( gnd_Train ~= k ) ;
        delta_Coeff = Coeff_Test ;
        delta_Coeff( index , : ) = 0 ;
        R(k,i) = norm( temR(:,i) )^2/ norm(delta_Coeff(:,i));
    end
end
[Rmin predictlabel] = min( R ) ;    % find the minimum residual
accuracy = length( find( predictlabel == gnd_Test ) ) / num_Test ;  % Recognition rate






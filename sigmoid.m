function out = sigmoid(x)
% x = [1 ; 0 ;3 ;9; 4; 5; 6]
out = exp(x)./(exp(x)+1);
end

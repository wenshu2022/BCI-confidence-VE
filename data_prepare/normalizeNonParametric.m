function x_norm = normalizeNonParametric(x)
%Normalize values in vector x to their cumulative probabilities, x_norm. 
%The middle CDF probability is chosed for multiple identical values in x.

%Input and output are assumed column vectors
if isrow(x)
    x = x';
    row_bool = true;
else
    row_bool = false;
end
num_x = length(x);

%For each x, get the index that refers to its location in a list of unique values in x (N.B.  x = unique_x(idx_unique_x))
[~,~,idx_unique_x] = unique(x);                     

%Find how many values of each unique x there are
num_unique_x = accumarray(idx_unique_x,ones(num_x,1));

%Find the middle cumulative probability of each set of unique values
cdf_upper_step_idx = cumsum(num_unique_x);
cdf_lower_step_idx = [0; cdf_upper_step_idx(1:(end-1))];      
cdf_middle_probs = .5*(cdf_lower_step_idx+cdf_upper_step_idx)/num_x;

%Return to original order of the values
x_norm = cdf_middle_probs(idx_unique_x);

%Return as row vector instead of column vector?
if row_bool
    x_norm = x_norm';
end

end %[EoF]

function [label, index, energy] = kmedoids_sam(X, init)
% Perform k-medoids clustering.
% Input:
%   X: d x n data matrix
%   init: k number of clusters or label (1 x n vector)
% Output:
%   label: 1 x n cluster label
%   index: index of medoids
%   energy: optimization target value
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
end
X = X-mean(X,2);             % reduce chance of numerical problems
v = dot(X,X,1);
D = v+v'-2*(X'*X);            % Euclidean distance matrix
D(sub2ind([n,n],1:n,1:n)) = 0;              % reduce chance of numerical problems
last = zeros(1,n);
while any(label ~= last)
    [~,~,last(:)] = unique(label);   % remove empty clusters
    [~, index] = min(D*sparse(1:n,last,1),[],1);  % find k medoids
    [val, label] = min(D(index,:),[],1);                % assign labels
end
energy = sum(val);

function [b,m,n] = myunique(varargin)
A = varargin{1};
[b,m,n] = unique(A, 'first');
if length(varargin) > 1 && varargin{2} == 'stable'
	[~,ii] = sort(m);
    b = b(ii);
    m = m(ii);
    [~, i] = sort(ii);
    n = i(n);
end
assert(isequal(b,A(m)));
assert(isequal(b(n),A));
end

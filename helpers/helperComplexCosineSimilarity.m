function similarity = helperComplexCosineSimilarity(A,B,method)
%helperComplexCosineSimilarity Complex cosine similarity
%   S = helperComplexCosineSimilarity(A,B) calculates the mean value of the
%   cosine similarity between corresponding rows of input matrices A and B.
%   This is the same as the "meanrows". 
%
%   S = helperComplexCosineSimilarity(A,B,"perm) calculates the cosine
%   similarity between all permutations of rows of input matrices A and B.
%
%   S = helperComplexCosineSimilarity(A,B,"rows") calculates the cosine
%   similarity between corresponding rows of input matrices A and B. Inputs
%   A and B must have the same size.
%
%   S = helperComplexCosineSimilarity(A,B,"meanrows") calculates the cosine
%   similarity between corresponding rows of input matrices A and B and
%   returns the mean similarity off all corresponding rows.  Inputs
%   A and B must have the same size.

%   Copyright 2024 The MathWorks, Inc.

arguments
  A double {mustBeFinite,mustBeNonempty,mustBeNonNan}
  B double {mustBeFinite,mustBeNonempty,mustBeNonNan}
  method string {mustBeMember(method,["rows","meanrows","perm"])} = "meanrows"
end
if (ndims(A) > 3) || (ndims(B) > 3)
  error('Inputs A and B must have at most three dimensions.');
end

switch method
  case "rows"
    % diag(A*B') ./ (vecnorm(A').*vecnorm(B'));
    sizesMustBeSame(A,B,method)
    similarity = sum(conj(A).*B,2) ./ (vecnorm(A,2,2).*vecnorm(B,2,2));
  case "meanrows"
    % mean(diag(A*B') ./ (vecnorm(A').*vecnorm(B')));
    sizesMustBeSame(A,B,method)
    similarity = mean(sum(conj(A).*B,2) ...
      ./ (vecnorm(A,2,2).*vecnorm(B,2,2)));
  case "perm"
    % A*B' ./ (vecnorm(A')'*vecnorm(B'))
    A_HtimesB = pagemtimes(B,"none",A,"ctranspose");
    vnormAvnormB = pagemtimes(vecnorm(B,2,2),"none", ...
      vecnorm(A,2,2),"ctranspose");
    similarity = A_HtimesB ./ vnormAvnormB;
end
end

function sizesMustBeSame(A,B,method)
if any(size(A) ~= size(B))
  error('When using "%s" method, inputs A and B must be the same size.', ...
    method);
end
end
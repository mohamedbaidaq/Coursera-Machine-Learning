function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% step 1 -> Caculating the cost function J without regularization
% (num_movies x num_features) X (num_features x num_users)  
J = (1/2) * sum(sum((((X * Theta').*R) - Y).^2));

% Adding the regularization term to the cost function J
J = J + ((lambda/2)*sum(sum(Theta.^2))) + ((lambda/2)*sum(sum(X.^2)));

% step 2 -> Caculating the gradients of feature matrix X 
% the resulting matrix should be num_movies x num_features per movie
% so every row in X_grad represents feature vector for every movie
for i=1:num_movies
    % each row of each iteration represents feature vector of each movie
    % so the resulting vector of every iteration is (1 x num_features)
    for j = 1:num_users
        X_grad(i,:) = X_grad(i,:) + ((R(i,j)*(X(i,:)*Theta(j,:)')) - Y(i,j))*Theta(j,:);
    end
    % adding the regularization term
    X_grad(i,:) = X_grad(i,:) + (lambda*X(i, :));
end

% step 3 -> Caculating the gradients of parameter matrix Theta 
% the resulting matrix should be num_users x num_features per user
% so every row in Theta_grad represents parameter vector for every user
for j=1:num_users
    % each row of each iteration represents parameter vector of each user
    % so the resulting vector of every iteration is (1 x num_features)
    for i = 1:num_movies
        Theta_grad(j,:) = Theta_grad(j,:) + ((R(i,j)*(X(i,:)*Theta(j,:)')) - Y(i,j))*X(i,:);
    end
    % adding the regularization term
    Theta_grad(j,:) = Theta_grad(j,:) + (lambda*Theta(j,:));
end
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
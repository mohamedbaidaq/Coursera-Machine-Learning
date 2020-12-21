function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% fprintf('size of input grad = ');
% size([Theta1_grad(:) ; Theta2_grad(:)])

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% ====================== Start of Part 1 CODE HERE ======================
% Calculating the output for all examples
y_all = zeros(num_labels, m); 
for i=1:m
    y_i = y_all(:, i);
    y_i(y(i)) = 1;
    y_all(:, i) = y_i; % resulting y_all matrix size will be 10 x 5000
end

% Calculating the predictions for all examples
p_all = zeros(num_labels, m);
% Add ones to the X data matrix
X = [ones(m, 1) X];
z2_all = Theta1 * X'; % resulting matrix will be 25 x 5000 (column per example)
a2_all = sigmoid(z2_all); % layer 2 inputs, also 25 x 5000 (column per example)
% Add ones to the a2 data matrix
a2_all = [ones(1, m); a2_all]; % add one more row (bias row) for all 5000 examples, resulting matrix will be 26 x 5000
z3_all = Theta2 * a2_all ; % resulting matrix will be 10 x 5000 (column per output for each example)
a3_all = sigmoid(z3_all) ; % resulting matrix will be 10 x 5000 (column per output for each example)
p_all = a3_all; % p_all size equal to 10 x 5000

% Calculating the cost per example (i)
cost_per_example = zeros(m, 1); % resulting matrix will be 5000 x 1
for i=1:m
    cost_per_example(i) = sum((-y_all(:, i).*log(p_all(:, i))) - ((1 - y_all(:, i)).*log(1 - p_all(:, i))));
end

J_no_reg = (1/m).*sum(cost_per_example);
J = J_no_reg + ((lambda/(2*m)).*sum(sum(Theta1(:,2:end).^2))) + ((lambda/(2*m)).*sum(sum(Theta2(:,2:end).^2)));
% ====================== End of Part 1 CODE HERE ======================

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% ====================== Start of Part 2 CODE HERE ======================
% we already have p_all and y_all for all training examples 
% we need to caculate delta_three_all for all training examples
% as p_all size is 10 X 5000 and also y_all 
% so the resulting matrix will have the same size 
% every coulmn will be the resulting delta for one training example
delta_three_all = p_all - y_all; 
Theta2_t = Theta2'; % resulting matrix size is 26 x 10
% we need to caculate delta_two_all for all training examples
% as Theta2' size is 26 X 10 and delta_three_all is 10 X 5000
% so the resulting matrix will have 26 X 5000
% every column will be the resulting delta for one training example
% a2_all size is 26 x 5000
% skip the delta_two_zero row so the resulting matrix will be 25 x 5000
% so we get one delta column per every training example
% delta_two_all = (Theta2_t(2:end, :) * delta_three_all) .* (a2_all(2:end, :) .* (1 - a2_all(2:end, :))); 
delta_two_all = (Theta2_t * delta_three_all) .* (a2_all .* (1 - a2_all)); % 26 x 5000
Delta_Theta1_grad = zeros(size(Theta1)); % 25 x 401
Delta_Theta2_grad = zeros(size(Theta2)); % 10 x 26
a2_all_t = a2_all'; % 5000 x 26 [row per example]
for t = 1:m 
    % step 1 Perform a feedforward pass using the feeded training example
    % step 2 caculate the delta for every output unit
    % step 3 caculate delta for the hidden layer
    % step 4 accumulate the value of Big Delta 
    % This is very important, this was the last bug of the program, as we
    % needed to include the bias term in the resulting matrix 
    Delta_Theta1_grad = Delta_Theta1_grad + (delta_two_all(2:end, t) * X(t, :)); % 25 x 401 
    % This is very important, this was the last bug of the program as we
    % needed to include the bias term in the resulting matrix 
    Delta_Theta2_grad = Delta_Theta2_grad + (delta_three_all(:, t) * a2_all_t(t, :)); % 10 x 26 
end
% step 5 get the gradient by scaling the big Deltas by 1/m
Theta1_grad_un_reg = (1/m) * Delta_Theta1_grad; % 25 x 401
Theta2_grad_un_reg = (1/m) * Delta_Theta2_grad; % 10 x 26
% Theta1_grad = Theta1_grad_un_reg;
% Theta2_grad = Theta2_grad_un_reg;
% the follwing two lines were wrong implementations (don't know why ?!! )
% Theta1_grad(:, 2:end) = Theta1_grad_un_reg(:, 2:end) + (lambda/m) .* (Theta1(:, 2:end));
% Theta2_grad(:, 2:end) = Theta2_grad_un_reg(:, 2:end) + (lambda/m) .* (Theta2(:, 2:end));
% ====================== END of Part 2 CODE HERE ======================

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
Theta1_grad = [Theta1_grad_un_reg(:, 1) (Theta1_grad_un_reg(:, 2:end) + ((lambda/m) .* (Theta1(:, 2:end))))];
Theta2_grad = [Theta2_grad_un_reg(:, 1) (Theta2_grad_un_reg(:, 2:end) + ((lambda/m) .* (Theta2(:, 2:end))))];
% =========================================================================
% Unroll gradients
% fprintf('size of resulting grad = ');
% size([Theta1_grad(:) ; Theta2_grad(:)])
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

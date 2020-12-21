function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
z2_all = Theta1 * X'; % resulting matrix will be 25 x 5000 (column per example)
a2_all = sigmoid(z2_all); % layer 2 inputs, also 25 x 5000 (column per example)
% Add ones to the a2 data matrix
a2_all = [ones(1, m); a2_all]; % add one more rwo (bias row) for all 5000 examples, resulting matrix will be 26 x 5000
z3_all = Theta2 * a2_all ; % resulting matrix will be 10 x 5000 (column per output for each example)
a3_all = sigmoid(z3_all) ; % resulting matrix will be 10 x 5000 (column per output for each example)
% p = max(a3_all, [], 1) ; % get the max value of each column (each example)
for i = 1:m
    example_output = a3_all(:, i);
    p(i) = find(max(example_output) == example_output);
end
% =========================================================================
end

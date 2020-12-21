function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% step 1 is to train the SVM classifier using the cross validation set
% (Xval, yval) using selected combination of parameters. 
% so the result will be 64 trained modules (8 X 8)
% C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% sigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% Train the SVM using every combination from C and sigma
C_vector = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vector = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
for C_idx = 1:length(C_vector)
    for sigma_idx = 1:length(sigma_vector)
        model = svmTrain(X, y, C_vector(C_idx), @(x1, x2) gaussianKernel(x1, x2, sigma_vector(sigma_idx)));
        predictions = svmPredict(model, Xval);
        pred_error(C_idx, sigma_idx) = mean(double(predictions ~= yval));
    end
end

disp(length(pred_error(:)));
best_model_idx = find(pred_error == min(pred_error(:)));
sigma = sigma_vector(floor(best_model_idx/8) + 1); 
C = C_vector(mod(best_model_idx, 8)); 

% 
% % convert the training models into a vectpr
% trained_models = trained_models(:); 
% 
% disp(trained_models);
% 
% % step 2 is to get the predictions for every trained model on the cross
% % validation set, Thus every column in the resulting matrix represents the
% % predictions per every trained model.
% for model_idx = 1:length(trained_models)
%     predictions(:, model_idx) = svmPredict(trained_models(model_idx), Xval);
%     % step 3 caculate the error on every trained model in order to select the
%     % best model to predict the data.
%     pred_error(model_idx) = mean(double(predictions(:, model_idx) ~= yval));
% end
% 
% % step 4 is to select the best model based on the calculated error per every model.
% best_model_idx = find(pred_error == min(pred_error));
% 
% sigma = sigma_vector(floor(best_model_idx/8) + 1); 
% C = C_vector(mod(best_model_idx, 8)); 

% =========================================================================

end

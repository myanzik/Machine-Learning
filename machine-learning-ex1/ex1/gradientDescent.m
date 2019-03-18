function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %saving old values of theta for simultaneous updates
     theta_old = theta;
     
    %no. of features 
     i = size(X, 2);
     
    %finding the values of theta
     for j=1:i
       %derivative part of GRADIENTDESCENT
       der = (((X * theta_old) - y)' * X(:,j))/m; 
       %Simultaneous update of theta
       theta(j) = theta_old(j) - (alpha * der); 
       
     end
     


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
     

end

end

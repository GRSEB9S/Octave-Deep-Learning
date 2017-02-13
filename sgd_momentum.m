function [params, param_winc] = sgd_momentum(rate, mu, weight_decay, params, param_winc, param_grad)
% update the parameter with sgd with momentum

%% function input
% rate (scalar): learning rate at current step
% mu (scalar): momentum
% weight_decay (scalar): weigth decay of w
% params (cell array): original weight parameters
% param_winc (cell array): buffer to store history gradient accumulation
% param_grad (cell array): gradient of parameter

%% function output
% params (cell array): updated parameters
% param_winc (cell array): updated buffer

for i=1:size(params)(2)
  param_winc{i}.w=mu*param_winc{i}.w+rate*(param_grad{i}.w + weight_decay*params{i}.w);
  param_winc{i}.b=mu*param_winc{i}.b+rate*param_grad{i}.b;
  params{i}.w=params{i}.w-param_winc{i}.w;
  params{i}.b=params{i}.b-param_winc{i}.b;
end

end
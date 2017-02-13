
add image
 
clear
rand('seed', 100000)
randn('seed', 100000)
 
%define lenet
layers{1}.type = 'DATA';
layers{1}.height = 28;
layers{1}.width = 28;
layers{1}.channel = 1;
layers{1}.batch_size = 64;
 
layers{2}.type = 'CONV';
layers{2}.num = 20;
layers{2}.k = 5;
layers{2}.stride = 1;
layers{2}.pad = 0;
layers{2}.group = 1;
 
layers{3}.type = 'POOLING';
layers{3}.k = 2;
layers{3}.stride = 2;
layers{3}.pad = 0;
 
 
layers{4}.type = 'CONV';
layers{4}.k = 5;
layers{4}.stride = 1;
layers{4}.pad = 0;
layers{4}.group = 1;
layers{4}.num = 50;
 
 
layers{5}.type = 'POOLING';
layers{5}.k = 2;
layers{5}.stride = 2;
layers{5}.pad = 0;
 
layers{6}.type = 'IP';
layers{6}.num = 500;
layers{6}.init_type = 'uniform';
 
layers{7}.type = 'RELU';
 
layers{8}.type = 'LOSS';
layers{8}.num = 10;
 
 
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
% learning rate parameters
mu = 0.9; % momentum
epsilon = 0.01; % initial learning rate
gamma = 0.0001; 
power = 0.75;
weight_decay = 0.0005; % weight decay on w
 
 
% display information
test_interval = 500;
display_interval = 10;
snapshot = 5000;
max_iter = 10000;
 
% initialize all parameters in each layers
% params = init_convnet(layers);
load lenet.mat
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img')
 
[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);
% Fill in your code here to plot the features.

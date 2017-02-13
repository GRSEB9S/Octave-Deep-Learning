function [output] = relu_forward(input, layer)
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

output.data=max(0,input.data);
end

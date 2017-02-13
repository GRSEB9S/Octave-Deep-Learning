function [output] = elu_forward(input, layer)
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

alpha = 1.0;
output.data = zeros(size(input.data));

case1=input.data;
case2=exp(input.data)-1;
bools=(input.data>0);

output.data = bools.*case1 + (~bools).*case2;

end

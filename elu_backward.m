function [input_od] = elu_backward(output, input, layer)

alpha = 1.0;
input_od = zeros(size(input.data));

case1=output.diff;
case2=output.diff.*e.^(input.data);
bools=(input.data>0);

input_od = bools.*case1 + (~bools).*case2;

end

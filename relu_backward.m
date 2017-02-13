function [input_od] = relu_backward(output, input, layer)

input_od = zeros(size(input.data));
bools=(input.data>0);
input_od = bools.*output.diff;

end

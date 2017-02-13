function [input_od_approx] = elu_finite_difference(output, input, h)

input_od_approx = zeros(size(input.data));
layer=0;

x=elu_forward(input, layer);
input_h.data=input.data+h;
input_h.height=input.height;
input_h.width=input.width;
input_h.channel=input.channel;
input_h.batch_size=input.batch_size;
x_h=elu_forward(input_h, layer);
values=(x_h.data-x.data)/h;
input_od_approx=output.diff.*values;

end

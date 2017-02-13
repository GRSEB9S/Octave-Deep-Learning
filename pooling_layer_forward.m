function [output] = pooling_layer_forward(input, layer)

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;

output.data = zeros([h_out * w_out * c, batch_size]);

%perform per batch
for batch=1:batch_size
  input_batch.data=input.data(:,batch);
  input_batch.height=h_in;
  input_batch.width=w_in;
  input_batch.channel=c;
  imcol=im2col_conv(input_batch, layer, h_out, w_out);
  %selections of size k*k to examine for a max
  ksquares=reshape(imcol, k*k, c, h_out*w_out);
  max_k=max(ksquares);
  %each k*k now a single max
  maxpool=reshape(max_k, c, h_out*w_out);
  output_batch=reshape(maxpool', h_out, w_out, c);
  output.data(:,batch)=output_batch(:);
end

end


function [input_od] = pooling_layer_backward(output, input, layer)

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

input_od = zeros(size(input.data));

%per batch
for batch=1:batch_size
  input_batch.data=input.data(:,batch);
  input_batch.height=h_in;
  input_batch.width=w_in;
  input_batch.channel=c;
  imcol=im2col_conv(input_batch, layer, h_out, w_out);
  %selections of size k*k to examine for a max
  %also might be error here. need to figure out dimensions.
  ksquares=reshape(imcol, k*k, c, h_out, w_out);
  max_k=max(ksquares);
  %expand max and compare to original
  expanded_max = repmat(max_k, k*k, 1, 1, 1);
  bools= (ksquares==expanded_max);
  %get diffs to same dimensions as max
  diff_batch = output.diff(:,batch);
  diffs= reshape(diff_batch, h_out*w_out, c)';
  %might be error here
  k_diffs= reshape(diffs, 1, c, h_out, w_out);
  %expand diffs and multiply by 0/1 values
  expanded_diffs=repmat(k_diffs, k*k, 1, 1, 1);
  full_diffs = expanded_diffs.*(bools);
  %convert to image format
  input_od_batch=col2im_conv(full_diffs(:), input, layer, h_out, w_out);
  input_od(:,batch)=input_od_batch(:);
end

end

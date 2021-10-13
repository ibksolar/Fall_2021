



%% Outline
% Load echogram and layer files
% Find the index of the surface
% Break into row block
% Break into col_block
% Pass each col_block to the NN and identify location of layer


% Initializations
clear; clc
row_block_length = 21; % The number of rows in each row block
rb_col_sz = 15; % The number of columns in each row block. PS: This value must be odd
half_col_sz = (rb_col_sz-1)/2; % This number must be an integer
debug_plot = 0;

%% Load simulated echogram and layer files

base_dir_echo = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/JSTARRS2021_Sep21/snow/2012_Greenland_P3/frames_001_243_20120330_04/image';
echo_fns = get_filenames(base_dir_echo, 'image','00','.mat',struct('recursive',true));

% Pre-allocate
echo_cnn_input = [];
echo_cnn_target =[];
total_layer_bin_mod =[];

% total_pred =[];
train_start = 1; 
train_stop = 500;


for fn_idx = train_start : train_stop % length(echo_fns)  
    
    echo_fn = echo_fns{fn_idx};
    
    [fn_dir,fn_name] = fileparts(echo_fn);
    fprintf('%d of %d (%s)\n', fn_idx, length(echo_fns), datestr(now));
    fprintf('  %s\n', fn_name);

    %  Load data (echogram and layer)
    tmp = load(echo_fn);
    
    echo_tmp1 = tmp.data; % Load echo.       PS: variable field may change    
    layer_tmp1 = tmp.layer;  % Load layer
    
    
    if debug_plot
      figure(100)
      imagesc(lp(echo_tmp1)); colormap(1-gray); title('Before decimation')
      hold on
      for idx = 1:size(layer_tmp1,1)  
        hold on;
        plot(tmp.layer(idx,:))
      end
    end
    
    % Decimation    
    echo_tmp = fir_dec( (fir_dec(echo_tmp1.',4)).',4) ; % Decimate both fast and slow time
    vec_layer = fir_dec(layer_tmp1,4);
    vec_layer = floor((vec_layer-1)/4+1);
    
    % Filter in slow time
    filter_len = 51;
    if debug_plot
      figure(200)
      subplot(211)
      imagesc(lp(echo_tmp)); colormap(1-gray); title('Echogram before filtering but after decimation')
      hold on
      for idx = 1:size(vec_layer,1)  
        hold on;
        plot(vec_layer(idx,:))
      end
      hold off         
    end
    
    echo_tmp = fir_dec(echo_tmp,ones(1,filter_len)/filter_len,1);
   
    if debug_plot
       figure(200)
       subplot(212)
       imagesc(lp(echo_tmp)); colormap(1-gray); title('Echogram after filtering and decimation')
       hold on
          for idx = 1:size(vec_layer,1)  
            hold on;
            plot(vec_layer(idx,:))
          end
       hold off 
    end
   
   
    % Create new raster    
   raster = zeros(size(echo_tmp)); 
   
   for layer_idx = 1:size(vec_layer,1)
    for layer_col = 1:size(vec_layer,2)

      tmp = vec_layer(layer_idx,layer_col); % This idx represents the column of the layer
      
      if ~isnan(tmp)
        if raster(tmp,layer_col) == 0
          raster(tmp,layer_col) = layer_idx;
        end        
      else
        continue        
      end

    end

   end
    
% 
%     if debug_plot
%       % Debug plot
%       figure(101)
%       imagesc(lp(echo_tmp1)); colormap(1-gray); title('After decimation')
%       figure(3);
%       hold on
%       for idx = 1:size(layer_tmp1,1)  
%         hold on;
%         plot(pp2(idx,:))
%       end
%       
%       
%     end

    
    % Plot echogram and layers
    if debug_plot
      
        figure(101); 
        imagesc(lp(echo_tmp)); colormap(1-gray)
        hold on
        a = title(sprintf(' Echogram %s ',fn_name));
        set(a,'Interpreter','none' )
          for plot_idx = 1: size(vec_layer,1)
            plot(vec_layer(plot_idx,:),'b--','Linewidth',2)
            hold on
          end
    end
 


    %% Modify current layer bin: pick rows starting from surface location
    [num_layers,num_cols] = size(vec_layer); 
    [num_rows,~] = size(echo_tmp);
    surf = vec_layer (1,:); % Surface location
%     mod_rw_length = m - min(surf); % Number of rows from surface to end of echo
% 
%     layer_bin_mod = zeros(mod_rw_length,n); % Initialize 
% 
%     for iter = 1:n % Iterate over columns of echo_tmp    
%       layer_bin_mod(:,iter) = curr_layer_bin(surf(iter)+1:m,iter);
%     end
% 
%     num_of_row_block = floor(size(layer_bin_mod,1)/row_block_length) +1;  
% 
%     % Zero pad modified echogram to a factor of row_block_length  
%       temp = (num_of_row_block*row_block_length) - size(layer_bin_mod,1) ;
%       layer_bin_mod = [layer_bin_mod; zeros(tmp,n) ];    
      
      
      

%% Break into row blocks   

    
    for lay_idx = 1: num_layers     


      if any(~isfinite(vec_layer(lay_idx,:))) 
        % Handles layers with NaN but first layer is never NaN
        layer_loc = min( layer_loc + ...
          round(row_block_length*exp(-0.025*lay_idx)), num_rows - row_block_length);      
%        layer_loc(~isfinite(layer_loc)) = min(surf(1) + scale * lay_idx, num_rows - row_block_length);
      else
        layer_loc = vec_layer(lay_idx,:);      
      end
     
      [blk_start,blk_end, row_block] = get_row_block(layer_loc +1, echo_tmp,row_block_length);
      [~,~,layer_block] = get_row_block(layer_loc +1, raster,row_block_length);
      [cnn_target,conseq_zeros] = getLayer_loc_v2(logical(layer_block)); % layer_loc = target
      
      cnn_target(cnn_target==0) = row_block_length+1; % zero maps to highest class
      
      % Creating a zero condition after all the layers have been taken
      if lay_idx == num_layers
        cnn_target(:) = row_block_length+1;
      end
      
      %% Create input matrix for CNN ignoring columns at the edges
      % cnn_input dim = (no of rows in block *no of neighboring cols) X ( No
      % of cols away from edge)    
      cnn_input = return_cnn_input_allcol(row_block,rb_col_sz); % 
      
      if debug_plot
        figure(100);
        subplot(211); 
        imagesc(echo_tmp); colormap(1-gray);title('Data')
        ylim([blk_start,blk_end])
        clims = caxis;
        subplot(212);  
        imagesc(row_block); colormap(1-gray); title('Row block')
        caxis(clims);
        
        figure(101);
        subplot(211);
         imagesc(layer_block); colormap(1-gray); title('Layer block')
        
        subplot(212); 
         plot(cnn_target); title('Cnn_target')
         
         fprintf('echo start %d : echo end %d \n',blk_start,blk_end)
%          keyboard
          
      end
     
      echo_cnn_input = [echo_cnn_input;cnn_input];
      echo_cnn_target = [echo_cnn_target; cnn_target.']; % This should return a vector

    end
    
%     total_layer_bin_mod = [ total_layer_bin_mod; layer_bin_mod ];
     
end




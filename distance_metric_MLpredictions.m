


% Distance metric for ML prediction
clearvars -except AdditionalPaths gRadar; 
clc;

% pred_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try'; 
% echo_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try/image';% layer_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try/layer';

% Prediction files created from Winproc
pred_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/predictions';

% Original image and layer files
echo_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/image';
layer_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/layer';

pred_files = get_filenames(pred_path, 'predictions','echo','.mat',struct('recursive',true));

all_cost = [];
debug_plot = 1; %false;

for img_idx = 1:length(pred_files)
    
    % Select prediction file
    predict_str = pred_files{img_idx};
    prediction = struct2cell( load(predict_str) ); 
    pred_vals = prediction{1};    
    
    % Get prediction file id
    [token,~] = regexp(predict_str,'echo(\d*)','tokens');
    img_id = token{:}; 
    img_id = str2double(img_id); % +1   

    img_path = get_filenames(echo_path,'image',sprintf('%06d',img_id),'.mat');
    lay_path = get_filenames(layer_path,'image',sprintf('%06d',img_id),'.mat');
    
    % Load needed files
    echo = load(img_path{1});
    layer = load(lay_path{1});
    layer = layer.vec_layer;
    
    % Select only finite layers
    fin_idx = find( ~all(isnan(layer')) ); % Layers with finite values
    
    % Remove all non_valid layers in ground truth?
    mod_layer = layer(fin_idx,:);
    
    
    %% Options of what to do to get accurately compare with NaN ground truth: 
      % (a) Interpolate nans from ground truth to ensure correct layer comparison
      % (b) Skip all ground truth with nan (This might be better)
      
    % Question: What happens if the prediction is NaN whereas Ground truth
    % is finite??
    
    % Answer: Updated to use the previous layer as the prediction     

    [~,Nl] = max( max(layer,[],2) ); % Number of layers(Nl) x Slow time(Nx)
    [Nx,Nt] = size(mod_layer);
    
    if debug_plot
      % Plot image with overlaid ground truth
      figure(100);
      imagesc(echo.echo_tmp); colormap(1-gray(256)); hold on; plot(mod_layer','b-','linewidth',1.5);
    end
    
    pos = 0;
    cost_per_frame = [];
    gt_layer_idx = []; % Ground Truth layer idx
    
    for lay_idx = 1:size(pred_vals,1)

        curr_pred_vals = pred_vals(lay_idx,:); 
        
        if ~all(isnan(curr_pred_vals)) && any(isnan(curr_pred_vals)) && lay_idx ~=1
          pred_nan_idx = isnan(curr_pred_vals);
          curr_pred_vals(pred_nan_idx) = pred_vals(lay_idx-1,pred_nan_idx);
        end
          
        
        if ~all(isnan(curr_pred_vals))                    
            
            % Compare current predicted layer with all the layers in the ground truth
            res1 = abs(repmat(curr_pred_vals,Nx,1) - mod_layer);
            
%             if any( any(isnan(res1),2) )
%                 nan_idx = isnan(res1);                
%                 % Set NaN values to very large number
%                 res1(nan_idx) = 1e12;               
%             end
            
            res2 = nanmean(res1,2);            
            [error_val ,pos] = nanmin(res2);
            
            if ~ismember(pos,gt_layer_idx)
              gt_layer_idx(end+1) = pos;              
              cost_per_frame(end+1) = error_val/size(pred_vals,1); % Should this be divided by Nx to get the mean?     
            
            else
              % The ground truth layer has been assigned to a previous prediction
              % layer: Resolve by choosing the prediction with the lowest error              
              if error_val < cost_per_frame(pos)
                cost_per_frame(pos) = error_val;              
              end
            end         
                   
            if debug_plot
              figure(100); hold on; plot(curr_pred_vals,'r*');
              title(sprintf('Image %d: Predict layer %d, Error: %2.2f(MAE pixels) compared to layer %d',img_idx,lay_idx,cost_per_frame(end), pos))
            end

        end        
    end    
    all_cost(end+1) = sum(cost_per_frame);
    
end
        
        
        
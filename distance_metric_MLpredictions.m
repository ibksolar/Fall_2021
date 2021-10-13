


% Distance metric for ML prediction
clearvars -except AdditionalPaths gRadar; 
clc;

% pred_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try'; 
% echo_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try/image';
% layer_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try/layer';

% Prediction files created from Winproc
pred_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/predictions';

% Original image and layer files
echo_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/image';
layer_path = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/layer';

pred_files = get_filenames(pred_path, 'predictions','echo','.mat',struct('recursive',true));

all_cost = [];

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
    
    % Answer: Whenever either prediction or ground truth is NaN, both cases
    % are exempted from loss calculation ( Should probably have another
    % metric to indicate when a prediction is NaN for a valid GT )
       
    if  0      
      mod_layer = layer;
      for iter_idx = fin_idx
        if any(isnan(layer(iter_idx,:)))
          mod_layer(iter_idx,:) = interp_finite(layer(iter_idx,:));
          %             temp1(isnan(temp1)) = round(mean(temp1));
        end
      end      
    end

    [~,Nl] = max( max(layer,[],2) ); % Number of layers(Nl) x Slow time(Nx)
    [Nx,Nt] = size(mod_layer);
    
    % Plot image with overlaid ground truth
    figure(100);
    imagesc(echo.echo_tmp); colormap(1-gray(256)); hold on; plot(mod_layer','b-','linewidth',1.5);
    
    pos = 0;
    cost_per_frame = [];
    
    for lay_idx = 1:size(pred_vals,1)

        curr_pred_vals = pred_vals(lay_idx,:); 
        
        if ~all(isnan(curr_pred_vals)) && any(isnan(curr_pred_vals)) && lay_idx ~=1
          pred_nan_idx = isnan(curr_pred_vals);
          curr_pred_vals(pred_nan_idx) = pred_vals(lay_idx-1,pred_nan_idx);
        end
          
        
        % Decide when to stop ( When prediction is greater than max in ground truth ) 
        if nanmean( curr_pred_vals) >= nanmean(mod_layer(end,:))
          keyboard
          break;
        end
        
        if ~all(isnan(curr_pred_vals))                    
            
            % Compare current predicted layer with all the layers in the ground truth
            res1 = abs(repmat(curr_pred_vals,Nx,1) - mod_layer);
            
            if any( any(isnan(res1),2) )
                nan_idx = isnan(res1); 
                
                % Set NaN values to very large number
                res1(nan_idx) = 1e12;               
            end
            
            res2 = nanmean(res1,2);            
            [cost_per_frame(end+1) ,pos] = min(res2);
            cost_per_frame(end) = cost_per_frame(end)/size(pred_vals,2); % Should this be divided by Nx to get the mean?
            
            if lay_idx ~= pos
              % Something is wrong
              keyboard
            end
            
            figure(100); hold on; plot(curr_pred_vals,'r*');
            title(sprintf('Predict layer %d, Error: %2.2f(MAE pixels) compared to layer %d',lay_idx,cost_per_frame(end), pos))
            

        end        
    end
    
    all_cost(end+1) = sum(cost_per_frame);
    
end
        
        
        
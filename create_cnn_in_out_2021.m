
% create_cnn_in_out_2021 ( Base from train_real_data2021.m )


%% Outline
% Load echogram and layer files
% Find the index of the surface
% Break into row block
% Break into col_block
% Pass each col_block to the NN and identify location of layer


% Initializations
clearvars -except AdditionalPaths gRadar;
clc

% Row Block settings
row_block_length = 21 ;  % 21  The number of rows in each row block
rb_col_sz = 15; % The number of columns in each row block. PS: This value must be odd
half_col_sz = (rb_col_sz-1)/2; % This number must be an integer
debug_plot = 0;


% Flags
create_segment = true;
decimate_flag = true;
reduce_zero_class = false;
Randomize = false;
create_row_block = true;
save_results = true;


% Pre-allocate
echo_cnn_input = [];
echo_cnn_target =[];
orig_echo_idx =[];
coords =[];
train_start = 1;

filter_len = 21;

% Iterative search or manual setting to produce same sized image (multiple of 416)
Nt_fix = 0;


%% Automated section

if decimate_flag
  id = 'Dec';
else
  id = 'Full';
end

train_or_test = 'Train';

% Load echograms and layer files
% base_dir_echo = '/cresis/snfs1/scratch/ibikunle/Python_Env/final_layers_rowblock15_21/filtered_image';
base_dir_echo = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/JSTARRS2021_final290621/snow/2012_Greenland_P3/frames_001_243_20120330_04/Train_set/image';

exclude_train = importdata('/cresis/snfs1/scratch/ibikunle/ct_user_tmp/JSTARRS2021_final290621/snow/2012_Greenland_P3/frames_001_243_20120330_04/exclude_train.txt');
echo_fns = get_filenames(base_dir_echo, 'image','00','.mat',struct('recursive',true));

train_stop =  length(echo_fns); %1250;

time_stamp = sprintf('%s_%s_block_len_%d_%s',id,train_or_test,row_block_length,datestr(now,'ddmmyy_hhMM'));
save_echo_mat_fn = fullfile('/cresis/snfs1/scratch/ibikunle/Python_Project/Fall_2021/all_block_data/Old_data', time_stamp);


if Nt_fix
  % Search for the deepest Nt in all of the data ( Might take time )
  sz_vector = []; for iter= 1: length(echo_fns); tt = load(echo_fns{iter}); sz_vector(end+1,:) = size(tt.data); end
  Nt_needed = max(sz_vector(:,1)); % Needed dim for segmentation deep learning models
else
  % Keep the Nt of the output fixed based on decided architecture of Neural Net (i.e produced same sized image - multiple of 416)
  Nt_needed = 1664; % 416; %1664;
end


% Randomize echograms to be used for training
if Randomize
  echo_list = randperm(length(echo_fns));
  echo_list = echo_list(1:train_stop);
else
  echo_list = 1:train_stop;
end

my_counter = 0;
for fn_idx = echo_list  %   length(echo_fns)
  
  my_counter = my_counter + 1;
  echo_fn = echo_fns{fn_idx};
  
  [fn_dir,fn_name] = fileparts(echo_fn);
  
  if ~ismember(fn_name,exclude_train) % Skip very bad frames
    
    fprintf('%d (Echo %d) of %d (%s)\n', my_counter, fn_idx, length(echo_fns), datestr(now));
    %     fprintf('  %s\n', fn_name);
    
    %  Load data (echogram and layer)
    tmp = load(echo_fn);
    echo_tmp1 = tmp.data; % Load echo.       PS: variable field may change
    layer_tmp1 = tmp.layer;  % Load layer
    
    % Interpolate layers with NaN gaps in the data
    check_nans = find ( any(isnan(layer_tmp1),2)  &  ~all(isnan(layer_tmp1),2) );
    if check_nans
      for iter_idx1 = 1:length(check_nans)
        iter_idx2 = check_nans(iter_idx1);
        layer_tmp1(iter_idx2,:) = round(interp_finite(layer_tmp1(iter_idx2,:)));
      end
    end
    
    % Augment the size of the echograms
    if Nt_needed %
      if size(echo_tmp1,1)~= Nt_needed
        extra = Nt_needed - size(echo_tmp1,1);
        echo_tmp1 = [echo_tmp1 ; zeros(extra, size(echo_tmp1,2) ) ];
      end
    end
    
    if debug_plot
      figure(100); clf;
      imagesc(echo_tmp1); colormap(1-gray); title('Before decimation')
      hold on
      plot(tmp.layer')
    end
    
    % Decimation
    if decimate_flag
      echo_tmp = fir_dec( (fir_dec(echo_tmp1.',4)).',4) ; % Decimate both fast and slow time
      vec_layer = fir_dec(layer_tmp1,4);
      vec_layer = floor((vec_layer-1)/4+1);
      
      if debug_plot
        figure(200)
        subplot(211)
        imagesc(lp(echo_tmp)); colormap(1-gray); title('Echogram before filtering but after decimation')
        hold on
        plot(vec_layer')
      end
      
    else
      vec_layer = layer_tmp1;
    end
    
    % Filter in slow time
    
    if exist('echo_tmp','var')
      % Decimate flag is on
      echo_tmp = fir_dec(echo_tmp,ones(1,filter_len)/filter_len,1);
    else
      echo_tmp = fir_dec(echo_tmp1,ones(1,filter_len)/filter_len,1);
    end
    
    if debug_plot
      figure(200)
      subplot(212)
      imagesc(lp(echo_tmp)); colormap(1-gray); title('Echogram after filtering and decimation')
      hold on
      plot(vec_layer')
      hold off
    end
    
    
    % Create new raster
    
    
    raster = zeros(size(echo_tmp));
    for layer_idx = 1:size(vec_layer,1)
      for layer_col = 1:size(vec_layer,2)
        temp = vec_layer(layer_idx,layer_col); % This idx represents the column of the layer
        
        if ~isnan(temp)
          if raster(temp,layer_col) == 0
            raster(temp,layer_col) = layer_idx;
          end
        else
          continue
        end
      end
    end
    
    
    % Plot echogram and layers
    if debug_plot
      figure(101);
      imagesc(lp(echo_tmp)); colormap(1-gray)
      hold on
      a = title(sprintf(' Echogram %s ',fn_name));
      set(a,'Interpreter','none' )
      hold on
      plot(vec_layer','b--','Linewidth',2)
    end
    
    % =======================================================================%
    %% Create semantic segment ground truth
    if create_segment % temporary - delete later
      semantic_seg = zeros(size(raster));
      [Nt2,Nx2] = size(vec_layer);
      for lay_idx = 1:Nt2
        for col_idx = 1:Nx2
          
          if ~isnan(vec_layer(lay_idx)) && ~isnan(vec_layer(lay_idx+1))
            
            if lay_idx ~= Nt2
              semantic_seg( vec_layer(lay_idx):vec_layer(lay_idx+1),col_idx) = lay_idx;
            else
              semantic_seg( vec_layer(lay_idx):vec_layer(lay_idx)+10,col_idx) = lay_idx;
            end
          end
        end
      end
    end
    %% =======================================================================%
    if save_results
      % Save created data
      % Save image and layer
      echo_dir = fullfile(save_echo_mat_fn,'image');
      layer_dir = fullfile(save_echo_mat_fn,'layer');
      raster_dir = fullfile(save_echo_mat_fn,'raster_dir');
      segment_dir = fullfile(save_echo_mat_fn,'segment_dir');
      figure_dir = fullfile(save_echo_mat_fn,'figure');
      
      if ~exist(echo_dir,'dir')
        mkdir(echo_dir)
      end
      
      if ~exist(layer_dir,'dir')
        mkdir(layer_dir)
      end
      
      if ~exist(raster_dir,'dir')
        mkdir(raster_dir)
      end
      
      if ~exist(segment_dir,'dir')
        mkdir(segment_dir)
      end
      
      if ~exist(figure_dir,'dir')
        mkdir(figure_dir)
      end
      
      %%  Saving echo, raster, segmentation_gt and figure with plotted layers
      fprintf( 'Saving image %d of %d \n',fn_idx,length(echo_fns) );
      
      
      if decimate_flag
        echo_title = [fn_name '_dec'];
      else
        echo_title = fn_name;
      end
      
      % Save echo
      save(fullfile(echo_dir,[echo_title '.mat']), 'echo_tmp');
      
      % Save layer
      save(fullfile(layer_dir,[echo_title '.mat']), 'vec_layer');
      
      % Save Raster
      save(fullfile(raster_dir,[echo_title,'_raster','.mat']), 'raster');
      
      % Save Semantic Segmentation
      save(fullfile(segment_dir,[echo_title,'_segment','.mat']), 'semantic_seg');
      
      
      
      % Save figure with plotted layers
      fn = fullfile(figure_dir,[echo_title,'.png']);
      figure(120);
      imagesc(echo_tmp1); colormap(1-gray(256));
      hold on;
      plot(layer_tmp1','linewidth',1.2);
      title(sprintf('%s',echo_title),'Interpreter','None');
      saveas(120,fn);
      
      close(120);
    end
    %% =======================================================================%
    if create_row_block
      
      %% Pick rows starting from surface location
      [num_layers,num_cols] = size(vec_layer);
      [num_rows,Nx2] = size(echo_tmp);
      surf = vec_layer (1,:); % Surface location
      
      
      %% Break into row blocks
      for lay_idx = 1: num_layers
        
        if any(~isfinite(vec_layer(lay_idx,:)))
          % Handles layers with NaN but first layer is never NaN
          if all(~isfinite(vec_layer(lay_idx,:))) % if all is nan
            continue;
            % layer_loc = min( layer_loc + round(row_block_length*exp(-0.025*lay_idx)), num_rows - row_block_length);
          else % if some are finite
            layer_loc = interp_finite(vec_layer(lay_idx,:)) ;
            fprintf('Interpolating for finite points in layer %d of echo %d \n',lay_idx, fn_idx);
          end
        else
          layer_loc = vec_layer(lay_idx,:);
        end
        
        [row_block] = get_row_block(layer_loc +1, echo_tmp,row_block_length);
        [layer_block] = get_row_block(layer_loc +1, raster,row_block_length);
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
        echo_cnn_input = cat(1,echo_cnn_input,cnn_input);
        echo_cnn_target = cat(1,echo_cnn_target, cnn_target.'); % This should return a vector
        orig_echo_idx = cat(1,orig_echo_idx, fn_idx*ones(length(cnn_target),1) );
        coords = cat(1, coords,[layer_loc' (layer_loc +row_block_length)' (1:Nx2)' min(64,7+(1:Nx2)')]);
        
      end
    end
  end
end
if reduce_zero_class
  zero_idx = find(echo_cnn_target == row_block_length+1); % 17
  delete_idx = zero_idx( randperm(length(zero_idx)) );
  
  class_cnt = arrayfun(@(x)length(find(echo_cnn_target == x)), unique(echo_cnn_target), 'Uniform', false); class_cnt = cell2mat(class_cnt);
  keep_cnt = round(mean(class_cnt));
  
  delete_idx = delete_idx(1:length(zero_idx)-keep_cnt);
  
  % Create new copies
  echo_cnn_input_reduced = echo_cnn_input;
  echo_cnn_target_reduced = echo_cnn_target;
  orig_echo_idx_reduced = orig_echo_idx;
  coords_reduced = coords;
  
  echo_cnn_input_reduced(delete_idx,:) = [];
  coords_reduced(delete_idx,:) = [];
  
  echo_cnn_target_reduced(delete_idx) = [];
  orig_echo_idx_reduced(delete_idx) = [];
  
  if 0
    echo_cnn1_reduced = echo_cnn_input_reduced(1:1000000,:);
    echo_cnn2_reduced  = echo_cnn_input_reduced(1000001:end,:);
    
    echo_target1_reduced = echo_cnn_target_reduced(1:1000000);
    echo_target2_reduced = echo_cnn_target_reduced(1000001:end);
    
    orig_echo1_reduced = orig_echo_idx_reduced(1:1000000);
    orig_echo2_reduced = orig_echo_idx_reduced(1000001:end);
    
    coords_reduced1 = coords_reduced(1:1000000,:);
    coords_reduced2 = coords_reduced(1000001:end,:);
    
    out_fn1 = '/cresis/snfs1/scratch/ibikunle/Python_Env/echo_cnn_in_out_GOOD_layers/new_echo_cnn_in_out_jstars1.mat';
    out_fn2 = '/cresis/snfs1/scratch/ibikunle/Python_Env/echo_cnn_in_out_GOOD_layers/new_echo_cnn_in_out_jstars2.mat';
    
    save(out_fn1,'echo_cnn1_reduced','echo_target1_reduced','coords_reduced1', 'orig_echo1_reduced');
    save(out_fn2,'echo_cnn2_reduced','echo_target2_reduced','coords_reduced2','orig_echo2_reduced');
    
  end
  
end


if create_row_block
  
  out_fn = fullfile(save_echo_mat_fn,'echo_cnn_in_out_jstars.mat');
  save(out_fn,'-v7.3','echo_cnn_input','echo_cnn_target','orig_echo_idx', 'coords');
  
  
  %     out_fn2 = '/cresis/snfs1/scratch/ibikunle/Python_Env/new_echo_cnn_in_out_jstarrs2021_first_try/echo_cnn_in_out_jstars2.mat';
  %     out_fn1 = '/cresis/snfs1/scratch/ibikunle/Python_Env/echo_cnn_in_out_GOOD_layers2/new_echo_cnn_in_out_jstars1.mat';
  %     out_fn2 = '/cresis/snfs1/scratch/ibikunle/Python_Env/echo_cnn_in_out_GOOD_layers2/new_echo_cnn_in_out_jstars2.mat';
  %     out_fn3 = '/cresis/snfs1/scratch/ibikunle/Python_Env/echo_cnn_in_out_GOOD_layers2/new_echo_cnn_in_out_jstars3.mat';
  %
  %     echo_cnn1 = echo_cnn_input(1:1000000,:);
  %     echo_target1 = echo_cnn_target(1:1000000,:);
  %     echo_idx1 = orig_echo_idx(1:1000000,:);
  %
  %     echo_cnn2 = echo_cnn_input(1000001:2000001,:);
  %     echo_target2 = echo_cnn_target(1000001:2000001,:);
  %     echo_idx2 = orig_echo_idx(1000001:2000001,:);
  %
  %     echo_cnn3 = echo_cnn_input(2000001:end,:);
  %     echo_target3 = echo_cnn_target(2000001:end,:);
  %     echo_idx3 = orig_echo_idx(2000001:end,:);
  %
  %     save(out_fn1,'echo_cnn1','echo_target1','echo_idx1');
  %     save(out_fn2,'echo_cnn2','echo_target2','echo_idx2');
  %     save(out_fn3,'echo_cnn3','echo_target3','echo_idx3');
  
  
end


%% Check: Confirm the CNN_inputs are correct
%   check_input = reshape(echo_cnn_input.',[row_block_length,rb_col_sz,size(echo_cnn_input,1)]);
%   [~,~,all_cols] = size(check_input);
%
%   if debug_plot
%
%       for col_iter_idx = 1:5:all_cols
%
%         figure;
%         title(sprintf('Column %d', col_iter_idx));
%         imagesc(check_input(:,:,col_iter_idx));
%         hold on
%         plot(1,echo_cnn_target(col_iter_idx),'rx', 'linewidth',5);
%         hold off
%
%       end
%   end
% ===========================   End Check ===========================================
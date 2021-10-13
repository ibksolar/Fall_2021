
clearvars -except AdditionalPaths gRadar
clc;

fn = 'new_simulations_aug2021/new';
num_echograms = 150;

%% Output path

out_dir = ct_filename_tmp('',fn);
% out_dir = ct_filename_ct_tmp('',fn);
% out_dir = fullfile('/cresis/snfs1/scratch/ibikunle/ct_user_tmp',fn);
base_layer_bin = fullfile(out_dir,'layer_binary');
base_layer = fullfile(out_dir,'layer');
base_image = fullfile(out_dir,'data');
base_figures = fullfile(out_dir,'figures');

%% Load in measured data
data_dir = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/snow/2012_Greenland_P3/final_20120330_04/stored_params.mat';
data = load(data_dir);
fs = 25;

% Group layer thickness into 4 ranges from very shallow to very deep
thickness_means = fir_dec(data.layer_thickness_means,3);
thickness_stds = fir_dec(data.layer_thickness_stds,3);
% cut_off_freq = fir_dec(data.cut_offs,5);

% Need to confirm this
% Nearest neighbor interpolation to fill zero spots
thickness_means(thickness_means==0) = nan;
thickness_stds(thickness_stds==0) = nan;

for idx = 1:size(thickness_means,1)
    
    while sum(isnan(thickness_means(idx,:)))>0
        thickness_means(idx,:) = interp_finite(thickness_means(idx,:));
        thickness_stds(idx,:) = interp_finite(thickness_stds(idx,:));
    end
end

num_layers = round( 30*rand(1) + 5 ); % Layer can be btw 5-45: Max number of layers from data is 25


for block = 1 : num_echograms
    
    
    %% Parameters of each instatiation
    
    bottom_gap = 20; % Pad after the deepest layer
    each_Nx = 256; % Number of rangelines (slow time) for each iteration
    no_frames = 1;
    num_spread = 100;
    
    Nx = each_Nx * no_frames;
    
    thickness_mult_mean = [1.3992, 1.1151, 1.0064, 0.9184]; % 1.238; % layer thickness multiplier mean
    thickness_mult_std = [0.4095 0.2156 .1908 .1767]; % 0.3627 % layer thickness multiplier standard deviation
    fast_time_dec_factor = 1;   
    
    
    %% A. Generate layers
    
    trans_mtx = [0.9 0.08 0.02 0;...       % 1. Shallow
        0.04 0.9 0.04 0.02;... % 2. Medium-shallow
        0.01 0.04 0.9 0.05;...  % 3. Medium-deep
        0.05 0 0.05 0.9];       % 4. Deep
    
    layers_final = [];
    states = [];
    
    %% A(i). Generate flat surface
    surf_bin = 100;  % Mean surface bin
    
    
    % Filter configuration
    padding = 200; 
    t = linspace(0,1,32*each_Nx/2); % Create a longer sequence

    % High pass frequency filter cut-off ( Controls high freq content )
    f_cutoff = ( 20*rand(1) + 35 ); %Min=35 [Max high freq], Max = 75 [Reduces high freq];   Old: %0.002*rand(1) + 0.015; 

    % Low pass filter cut-off (Agg cut off high freq)
    f_cutoff2 = round ( 200*rand(1) + 500, -2 );   %Min:200 Max:700 Rounded to nearest 100

    f1 = 10^(16/20);
    f2 = 10^(-13/20);
    
    f3 = 10^(30/20);
    f4 = 10^(-20/20);
    
    H = f2 + (f1-f2) * exp( -t*f_cutoff);
    H = [H, fliplr(H)];
    
    H2 = f4 + (f3-f4) * exp( -t * f_cutoff2);
    H2 = [ H2  fliplr(H2) ] ;
    
    if 0
        figure(100); clf;
        plot(lp(H,2)); hold on; plot(lp(H2,2))
    end
    
    %% A(ii). Generate internal layers
    
    % Choose a state from the state-machine model
%     curr_state = 1; % Reset state    
%     curr_state = find (mnrnd(1,trans_mtx(curr_state,:)));
%     %     curr_state2 = find (mnrnd(1,trans_mtx(curr_state1,:)));
%     states(end+1) = curr_state;    
    
    avg_1st_layer_thickness = randsample([22:99],1,true); % thickness_means(1,curr_state); % Choose layer1 thickness
        
    if (avg_1st_layer_thickness <= 40)
        lay1_idx = 1;
    elseif (avg_1st_layer_thickness > 40 && avg_1st_layer_thickness <= 60)
        lay1_idx = 2;
    elseif (avg_1st_layer_thickness > 60 && avg_1st_layer_thickness <= 80)
        lay1_idx = 3;
    elseif (avg_1st_layer_thickness > 80 && avg_1st_layer_thickness <= 100)
        lay1_idx = 4;
    else
        fprintf('BUG ALERT - Generated mean is outside allowed range \n')
    end
            
    
    for frame_idx = 1:no_frames
        
        layers(1,:) = surf_bin * ones(1,each_Nx);        
        means_vec = [];    

        for lay_idx = 2:num_layers
            
%             curr_mean1 = thickness_means( lay_idx,curr_state);
%             curr_std1 = thickness_stds( lay_idx,curr_state );
            
            curr_mean1 = round( avg_1st_layer_thickness * (thickness_mult_mean(lay1_idx) + thickness_mult_std(lay1_idx)* randn(1) ) ) ;
            
            % Generate a random process and filter in the freq domain
            
            temp_layer = randn(1,length(H)) ;
            temp_fft = fft(temp_layer);
            layer_high_freq = round( ifft( temp_fft.*H,'symmetric') ); % High frequency comp
            
            layer_high_freq_used = layer_high_freq(padding:padding+each_Nx-1);           

            layer_low_freq = round ( (ifft ( temp_fft .*H2,'symmetric' )) ) ; % Low freq comp
            layer_low_freq_used = layer_low_freq(padding:padding+each_Nx-1);
            
            
            % Generating the next layer by adding new mean and high freq            
            % Each_layer =  curr_mean + low_freq_comp + high_fr
            
            layers(lay_idx,:) =  round( curr_mean1  + layer_high_freq_used); %+ layer_low_freq           

            
            if 0 % Confirm this fix with Dr Paden.
                layers(lay_idx,:) = round( fir_dec( layers(lay_idx,:),ones(1,11)/11,1 ) );
            end
            
        end
        
        layers_final = [layers_final cumsum(layers)];
        
    end
    
    Nt = max(layers_final(:)) + bottom_gap; % Fast time
    
    layers_mod = layers_final + repmat( Nt*(0:Nx-1),num_layers,1) ;
    
    layer_thickness = diff(layers_final);
    
    
    
    %% Generate powers for layers
    
    power_stats = load('/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/snow/2012_Greenland_P3/final_20120330_04/stored_params2.mat');
    power_stats2 = load('/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/Aug2021_new_images_and_data/nsf_report_data.mat');
    
    lay_means = power_stats.lay_power_means;
    lay_var = power_stats.lay_power_var;
    
    decay_mean = power_stats2.mean_decay_rate;
    decay_std = power_stats2.std_decay_rate;
    
    N_incoh_avg = 55;
    
    
    if 0
        for lay_idx = 1:num_layers
            
            % trails are the indexes before and after the layer itself that we want
            % to populate using the thickness of each layer
            if lay_idx ~= num_layers
                trail = round(0.9*mean(layer_thickness(lay_idx,:))); % 0.75
            end
            
            % lay_means has the means of the layer power from data.
            % 50 is the index of the layer in lay_means and 49 idxs before and
            % 80 idxs after the layer is collected equaling 130 points
            
            if lay_idx == 1
                % Generate index starting from 1 to half_layer thickness after the
                % layer
                indexes = 1: 50+trail;
                
            elseif lay_idx == num_layers
                indexes = 50 + (-trail:50);
                
            else
                trail1 = layer_thickness(lay_idx-1,:) - round(0.5 * mean(layer_thickness(lay_idx-1,:)));
                indexes = 50 + (-trail1:trail);
            end
            
            len_indexes = length(indexes);
            
            wf_check(end+1: end+len_indexes) = lay_means(indexes,lay_idx);
            
            %     power_val = repmat(lay_means(indexes,lay_idx),1,Nx,N_incoh_avg) .* abs(randn(len_indexes,Nx,N_incoh_avg) +1j*randn(len_indexes,Nx,N_incoh_avg)).^2;
            power_val = repmat(lay_means(indexes,lay_idx),1,Nx,N_incoh_avg) .* (randn(len_indexes,Nx,N_incoh_avg)).^2;
            
            power_val = reshape(mean(power_val,3),[len_indexes,Nx]);
            
            indexes = indexes - 50;
            
            for each = 1:len_indexes
                echo (layers_mod(lay_idx,:) + indexes(each) ) =  power_val(each,:);
            end
            
        end
        
        figure; imagesc(lp(echo)); colormap(1-gray(256))
    end
    
    
    % Attempt 2
    
    if 0
        echo2 = zeros(Nt, Nx);
        
        for rline = 1: Nx/10
            for lay_idx = 1 : num_layers
                
                
                % lay_means has the means of the layer power from data.
                % 50 is the index of the layer in lay_means and 49 idxs before and
                % 80 idxs after the layer is collected equaling 130 points
                
                if lay_idx == 1
                    % Generate index starting from 1 to layer thickness bins after the
                    % layer
                    trail = layer_thickness(lay_idx,rline);
                    indexes = 1: 50+trail;
                    
                elseif lay_idx == num_layers
                    indexes = 50:130;
                else
                    trail = layer_thickness(lay_idx,rline);
                    indexes = 50: 50+trail-1;
                end
                
                len_indexes = length(indexes);
                
                p_loc = find(indexes==50,1,'first');
                
                c_power = lay_means(indexes,lay_idx); % Current layer(waveform) power
                v_power = lay_var(indexes,lay_idx); % Current layer(waveform) variance
                
                % Randomize power to within one std deviation based on variance from data.
                %         temp0 = (-1 +2*rand(len_indexes,1)).* sqrt(v_power);
                temp0 = randn(len_indexes,1);
                temp0(p_loc) = abs(temp0(p_loc)); % Keep peak power positive
                
                used_power = c_power + temp0;
                m_power = [m_power; used_power];
                
                power_val = mean( repmat(c_power,1,N_incoh_avg) .* (randn(len_indexes,N_incoh_avg)).^2  ,2 );
                
                indexes = indexes - 50;
                
                
                echo2 (layers_final(lay_idx,rline) + indexes, rline ) =  power_val;
                
                
                
            end
        end
        
        figure; imagesc(lp(echo2)); colormap(1-gray)
        
    end
    
    
    if 1
        
        % Attempt 3
        % Convolution of Gaussian and Exponential function

        Nt2 = 1500;
        
        if Nt > Nt2
            Nt2 = round(Nt,-2) + 200;
        end
        
        x = 1:Nt2;
        echo3 = zeros(Nt2,Nx);
        
        pwr_ratio = [];
        
        % Roll/ Elevation effect as a state machine
        roll_trans_mtx = [ 0.9995 0.0005; 0.005 0.995];
        roll_state = [1 1]; % initial state
                
        decay_rt = [ 0.01 , 0.005, 0.005, 0.015,0.025, 0.04*ones(1,num_layers-5)];
        
        rline_wf = zeros(Nt2,num_layers);
        
        lay_power = (lay_means(50,:) ); % 50 is the index for the layer power
        lay_pow_var = (lay_var(50,:) );
        
        glob_max_depth = max(layers_final(:)) + 5;
        
        curr_lay_pwr = nan(num_layers,Nx);
        decay_ratio = nan(num_layers,Nx);
        
        for lay_iter = 1:num_layers
            rho = 0.9;
            if lay_iter > 3
                decay_ratio(lay_iter,:) = 0.55 + 0.018*( rho*randn(1,Nx)+ sqrt(1-rho^2)*randn(1,Nx) ) ;
            else
                decay_ratio(lay_iter,:) = 0.15 + 0.002*( rho*randn(1,Nx)+ sqrt(1-rho^2)*randn(1,Nx) ) ;
            end
            
            if lay_iter <= length(lay_power)-2               
                curr_lay_pwr(lay_iter,:) = lay_power(lay_iter)+ lay_pow_var(lay_iter)*( rho*randn(1,Nx)+ sqrt(1-rho^2)*randn(1,Nx) ) ; % Correlated random Guassian                 
            else
                curr_lay_pwr(lay_iter,:) = lay_power(end)+ lay_pow_var(end)*( rho*randn(1,Nx)+ sqrt(1-rho^2)*randn(1,Nx) ) ;
            end                
        end

        
        for iter = 1: size(layers_final,2)
            
            % Compute roll state
            roll_state(end+1) =  find (mnrnd(1,roll_trans_mtx(roll_state(end),:)));
            
            if roll_state(end) == 1
                roll_val = 1;
            else
                % Check previous roll_state
                if roll_state(end-1) ~= roll_state(end)
                    roll_val = min ( 0.7 + .05*randn(1), 0.9); % randomize roll state if roll_state changed
                end
            end
            
            layer_pos_vector = layers_final(:,iter);
            rate_time = diff(layer_pos_vector);            

            
            %  Generate layers by convolving Gaussian and exponential
            for layer_idx = 1:num_layers
                
                if layer_idx <= 30
                    pow_idx = layer_idx;
                else
                    pow_idx = 30;% Use power of the deepest tracked layer for deeper layers
                end               

%                 hgt =  lay_power (pow_idx) + 0.01*lay_pow_var(pow_idx)*randn(1); % / (2.5*lay_idx) ; %*fading_factor (pow_idx) ;  hgt2 = lay_power (pow_idx+1)  / (2.5*lay_idx); % * fading_factor(pow_idx+1) ;
                hgt = curr_lay_pwr(layer_idx, iter);
                
                wdt = round( 31+ 10*randn(1) );   %randsample(20:25,1); % Integer between 1-3 , width(pow_idx) ; %pow_idx,layer_idx
                if wdt < 5
                    wdt = randsample([18,28,39,51],1); % Values from histogram of width
                end                
            
          
                if layer_idx ~= length(layer_pos_vector)
%                     pwr_ratio(end+1) = 0.3 + 0.018*randn(1);                     
%                     if layer_idx < 4
%                         pwr_ratio(end) = min( 0.1+ 0.018*randn(1) , pwr_ratio(end) );
%                     end
                    
                    decay_rt2 = log ( decay_ratio(layer_idx, iter) ) / (-rate_time(layer_idx)); % decay rate to btw 30%-35% of height

                    if decay_rt2 < 0
                        keyboard;
                    end
                end
                
                % pos = layer_pos_vector(layer_idx);
                pos = layers_final(layer_idx,iter);
                
                G =  exp( - ((x-pos).^2)/ ( 2*wdt) ); % Gaussian
                
                try
                    E =  exp( -1* decay_rt2.*x ); % Exponential
                catch
                    E =  exp( -1* decay_rt(layer_idx).*x ); % Exponential
                end
                
                temp_conv = conv(G,E);
                [pk,pk_loc] = max(temp_conv);
                temp_conv = hgt * (temp_conv/pk); % Re-normalize after conv and scale appropriately
                
                % Shifted convolution result + noise for each layer
                rline_wf(:,layer_idx) = temp_conv( [1:Nt2]+ (pk_loc-pos) ) ;
                
            end
            
            
            max_depth = max(layers_final(:,iter));
            
            
            rl_scope = sum(rline_wf,2);
            rl_scope( rl_scope < 0 ) = 0;            

            therm_noise = repmat( 0.0041,Nt2,N_incoh_avg,1)+ sqrt(0.005)*( randn(Nt2,N_incoh_avg,1) + 1j* randn(Nt2,N_incoh_avg,1) ) ; %[ 0.0005*ones(surf_bin-1,1); 0.0006*ones(Nt2-surf_bin+1,1)];
            
            % 0.0005            
            % Generate Chi-squared: using mean from data as variance
            rl_scope = repmat( sqrt(rl_scope),1,N_incoh_avg) .* ( randn(Nt2,N_incoh_avg,1) +1j* randn(Nt2,N_incoh_avg,1) );
            echo3(:,iter) = mean( abs((rl_scope +therm_noise).^2) ,2);
            
        end
        
%         internal_layers_nfloor = repmat( 0.02* exp(-0.001*[0:Nt2-surf_bin]'),1, Nx) ;
%         echo3(surf_bin:end,:) = echo3(surf_bin:end,:) + internal_layers_nfloor;
        
        echo = lp( echo3(1:Nt,:) );
        figure(61); clf; imagesc(echo); colormap(1-gray); title('Simulated echogram');
        %         hold on; plot(layers_final');
        
    end
    
    used_max = max ( max( lp(echo3(surf_bin:Nt, :)) ) );
    
    echo = ( echo - (used_max- 7) )/ 7 ; % 8dB range
    echo = uint8(255*echo);
    
    nf = mean ( echo_noise( lp(echo3(1:Nt,:)) )  );
    used_min = min ( min( lp(echo3(surf_bin:Nt, :)) ) ); % Min btw surf_bin to Nt
    
    
    %% Create layer raster from layer_rangebin (layers_final)
    raster = zeros(size(echo));
    for layer_idx = 1:size(layers_final,1)
        for layer_col = 1:size(layers_final,2)
            temp = round(layers_final(layer_idx,layer_col)); % This idx represents the column of the layer
            %         temp = temp + top_gap; % Add top_gap offset
            if ~isnan(temp)
                if raster(temp,layer_col) == 0
                    raster(temp,layer_col) = layer_idx;
                end
            else
                continue;
            end
        end
    end
    
    
    
    
    % Explanation of scale
    % scale -->  max_valid_dB
    % .6    -->  15dB           .3 --> 30dB   .15 --> 60dB
    %     scale_min = 0.6* (used_min - nf) /  (nf -used_max);
    
    %     nparam = []; % Normalization param
    %     nparam.window_units = '%';
    %     nparam.scale = [scale_min 1-scale_min]; % Scale min rep approx. noise floor scale(in %) relative to max(data(:))
    %     nparam.scale = [0.4 1];
    %     nparam.window = [surf_bin/Nt2; glob_max_depth/Nt2 ]; % Represents what portion of the data should be used in estimating the noise
    %     echo_norm(lp(echo3(1:Nt,:)), struct('scale',[s 1],'valid_max_range_dB',[15 inf]));
    %
    %     nparam.valid_max_range_dB = [30 inf];
    %     echo = echo_norm( lp(echo3(1:Nt,:)), nparam );
    %
    
    if 0
        % Visualize noise floor and scale min
        figure(10); clf;
        plot(lp(echo3)); hold on;
        % Plot noise floor
        line([0,size(echo3,1)], [nf nf],'linewidth',1.2); text(round(0.3*size(echo3,1)),nf+0.2,'nf');
        hold on;
        % Plot scale min
        line([0,size(echo3,1)], [used_min used_min],'linewidth',1.2); text(round(0.3*size(echo3,1)),used_min+0.2,'Used min');
        hold off
        
        figure(13);
        imagesc(echo); colormap(gray); caxis([0 1])
        
    end
    
    
    
    %% Save data and images
    
    tmp_block_data = [];
    tmp_block_data.block                = block;
    tmp_block_data.data                 = echo;
    tmp_block_data.layer                = layers_final  ;
    tmp_block_data.f_cutoff1            = f_cutoff;
    tmp_block_data.f_cutoff2            = f_cutoff2;
    tmp_block_data.surf_bin             = surf_bin;
    tmp_block_data.num_layers           = num_layers;
    tmp_block_data.layer_mean_state     = curr_state;
    
    tmp_block_data.layer_mean_all_state = states;
    tmp_block_data.high_frq_lay_idx     = high_frq_lay_idx;
    
    
    if ~exist(base_layer_bin, 'dir')
        mkdir(base_layer_bin);
    end
    
    if ~exist(base_layer, 'dir')
        mkdir(base_layer);
    end
    
    if ~exist(base_image, 'dir')
        mkdir(base_image);
    end
    
    if ~exist(base_figures, 'dir')
        mkdir(base_figures);
    end
    
    fprintf('Saving %d of  %d\n', block,num_echograms);
    
    out_fn = fullfile(out_dir,'image',sprintf('data_%06d.mat',block));
    fprintf('    Save %s\n', out_fn);
    save(out_fn,'-struct','tmp_block_data');
    out_fn = fullfile(out_dir,'image',sprintf('data_%06d_test.png',block));
    imwrite(echo,out_fn);
    
    out_fn = fullfile(out_dir,'figures',sprintf('data_fig_%06d.png',block));
    fprintf('    Saving image with layers plotted as %s\n', out_fn);
    
    saveas(60,out_fn);
    close(60)
    
    
    out_fn = fullfile(out_dir,'layer',sprintf('layer_%06d.png',block));
    fprintf('    Save %s\n', out_fn);
    imwrite(raster,out_fn);
    
    out_fn = fullfile(out_dir,'layer_bin',sprintf('layer_binary_%06d.png',block));
    fprintf('    Save %s\n', out_fn);
    imwrite(logical(raster),out_fn);
    
    
    
    %%  Checks: (a) Correlation along track
    if 0
        % Layer 1 correlation
        lay1 = echo3(100,1:1000)';
        corr_vals = (lay1'*lay1)/( norm(lay1) * norm(lay1) ); % first element equals 1
        
        for iter_idx = 1:2000
            next_rl = echo3(100,1+iter_idx: 1000+iter_idx)';
            corr_vals(end+1) = (lay1'*next_rl)/( norm(lay1) * norm(next_rl) );
        end
        
    end
    
end



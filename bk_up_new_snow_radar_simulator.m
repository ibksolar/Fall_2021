


% new_snow_radar_simulator
% rng(10);
clearvars -except AdditionalPaths gRadar
clc;

%% Load measured data
fn = 'simulated_images/output2';
out_dir = ct_filename_tmp('',fn);
base_layer_bin = fullfile(out_dir,'layer_binary');
base_layer = fullfile(out_dir,'layer');
base_image = fullfile(out_dir,'data');
base_figures = fullfile(out_dir,'figures');

layers_fn = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/Lora_2012/test/gt/manually_completed';
layers_dir = get_filenames(layers_fn,'layer','20','.png');

debug_plot = 1;
gen_layers = 1; % Flag indicating whether layers should be generated or Lora manually picked layers should be used
save_output = 1;

if gen_layers
    
    data_dir = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/snow/2012_Greenland_P3/final_20120330_04/stored_params.mat';
    data = load(data_dir);   

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
    
    %% Parameters of each instatiation
    bottom_gap = 20; % Pad after the deepest layer
    each_Nx = 256; % Number of rangelines (slow time) for each iteration
    no_frames = 1;
    Nx = each_Nx * no_frames;
    padding = 50; % Trend filter padding to avoid edge effects
    min_layer_diff = 0; % The minimum distance between consecutive layers
    
    num_echograms = 50;
    fast_time_dec_factor = 1;  %0.5; % fast_time_decimation_factor
    
    thickness_mult_mean = 1.238; % - fast_time_dec_factor; % layer thickness multiplier mean
    thickness_mult_std = 0.3627; % - 0.5*fast_time_dec_factor; % layer thickness multiplier standard deviation
    
    avg_layer_thickness = 47.5 * fast_time_dec_factor;
    
    %% A. Generate layers
    trans_mtx = [0.9 0.08 0.02 0;...       % 1. Shallow
        0.04 0.9 0.04 0.02;... % 2. Medium-shallow
        0.01 0.04 0.9 0.05;...  % 3. Medium-deep
        0.05 0 0.05 0.9];       % 4. Deep
    states = [];
    
    
    trend_trans_mtx = [0.3 0.7; 0.6 0.4];    
    
    %% A(i). Generate flat surface
    surf_bin = 55;  % Mean surface bin
    
    % Filter configuration
    t = linspace(0,1,32*each_Nx/2);   

else
    num_echograms = length(layers_dir);
    out_dir = fullfile(out_dir,'sim_from_UMBC_lora_layers');
    
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
end



for echo_idx = 45: num_echograms
    
    if gen_layers
        %% A(ii). Generate internal layers
        % Choose a state
        curr_state = 1; trend_state = 2; % Reset state
        curr_state = find (mnrnd(1,trans_mtx(curr_state,:)));
        
        trend_state = find (mnrnd(1,trend_trans_mtx(trend_state,:)));
        states(end+1) = curr_state;
        
        % High frequency filter cut-off
        f_cutoff = 4 * ( 20*rand(1) + 40 ); %Min=20, Max = 40;   Old: %0.002*rand(1) + 0.015; %Max:0.035 Min:0.015
        
        % Low frequency filter cut-off
        f_cutoff2 = round ( 200*rand(1) + 500, -2 );   %Min:200 Max:450 Rounded to nearest 100
        
        f1 = 10^(16/20);
        f2 = 10^(-13/20);
        %     f3 = 10^(30/20); f4 = 10^(-20/20);
        %     H = f2 + (f1-f2) * exp( -(0:(each_Nx/2)-1)/(each_Nx/2)/f_cutoff);
        
        H = f2 + (f1-f2) * exp( -t*f_cutoff);
        H = [H, fliplr(H)];
        
        H2 = f2 + (f1-f2) * exp( -t * f_cutoff2);
        H2 = [ H2  fliplr(H2) ] ;
        
        layers_final = [];
        layers = [];
        
        for frame_idx = 1:no_frames
            layers(1,:) = surf_bin * ones(1,each_Nx);
            means_vec = [];
            
            
            % Trend cut off : vary for each echogram
            if trend_state == 1
                trend_cut_off =  0.01 + 0.002*rand(1) ;
                num_layers = round( 15*rand(1) + 15 ); % Layer can be btw 5-15: Max number of layers from data is 25
            else
                trend_cut_off = 0.05 + 0.05*rand(1); % 0.015 Good:0.03 + 0.005*rand(1) - .015
                num_layers = round( 5*rand(1) + 5 ); % Wet/Ablation Zone - fewer layers
            end
            
            [B1,A1] = butter(2, trend_cut_off);
            
            % Trend is a pattern to rep geographical/climatic features of a
            % location e.g slope of the ice sheet
            % It should not be added all the time since most surfaces are
            % typically smooth
            
            
            layer_trend = filtfilt(B1,A1, randn(1,2*padding+each_Nx));
            K = randperm(length(layer_trend)); K = K(1:each_Nx);
            layer_trend_used = layer_trend(padding+1:padding+each_Nx); % Remove edge effect
            
            
            for lay_idx = 2:num_layers
                %  curr_mean1 = thickness_means( lay_idx,curr_state);
                curr_mean1 = round( avg_layer_thickness * (thickness_mult_mean + thickness_mult_std* randn(1) ) ) ;
                
                % Generate a random process and filter in the freq domain
                
                temp_layer = randn(1,length(H)) ;
                temp_fft = fft(temp_layer);
                layer_high_freq =  ifft( temp_fft.*H,'symmetric') ; % High frequency comp
                layer_high_freq_used = layer_high_freq(padding:padding+each_Nx-1);
                
                layer_low_freq = ( ifft ( fft (randn(1,length(H2))) .*H2,'symmetric') ) ;% Low freq comp
                layer_low_freq = round( fir_dec( 4*layer_low_freq, ones(1,25)/25,1) ); % Might remove this ( Scale and filter)
                layer_low_freq_used = layer_low_freq(padding:padding+each_Nx-1);
                
                % Generating the next layer by adding new mean and high freq to the previous
                % Each_layer = curr_mean + low_freq_comp + high_fr
                
                if trend_state == 1
                    layers(lay_idx,:) = round( curr_mean1 + layer_low_freq_used  ); %layer_high_freq_used + layer_low_freq_used
                else
                    layers(lay_idx,:) = round( curr_mean1 + layer_low_freq_used  + lay_idx*layer_trend_used); %2*  + layer_low_freq_used
                end
                % Check gap
                if any( layers(lay_idx,:) - layers(lay_idx-1,:)) <= min_layer_diff;
                    layers(lay_idx,:) = layers(lay_idx,:) + min_layer_diff;
                end
                
            end
            
            layers_final = [layers_final cumsum(layers)];
        end
        
        if debug_plot
            figure(7); plot(layers_final'); axis ij;
        end
        
        Nt = max(layers_final(:)) + bottom_gap; % Fast time
        layer_thickness = diff(layers_final);
        
    else
        
        [~,layer_title] = fileparts(layers_dir {echo_idx});
        layer_rd =  imread(layers_dir {echo_idx}) ;  % This sometimes have repeating indexes
        [Nt,Nx] = size(layer_rd);
        
        % A condition to check repeating indexes will be nice ( not done)
        layer_gt = zeros(Nt,Nx);
        for idx = 1: Nx
            [~,ia] = unique(layer_rd(:,idx),'stable');
            layer_gt(ia,idx) = 1;
        end
        
        layer_gt(1,:) = 0;
        num_layers = max( sum(layer_gt));
        
        layers_final = layer_gt .* repmat( (1:Nt)',1,Nx) ;
        first_find = find( any(layers_final(1:100,:)),1,'first');
        surf_bin = find( layers_final(:,first_find),1,'first');
        
    end
    
    
    
    %% Generate powers for layers
    % (a.) Load saved power profiles
    power_stats = load('/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/snow/2012_Greenland_P3/final_20120330_04/stored_params2.mat');
    
    lay_means = power_stats.lay_power_means;
    lay_var = power_stats.lay_power_var;
    N_incoh_avg = 50; % 50;
    
    Nt2 = 1500; % 1000
    
    if Nt > Nt2
        Nt2 = round(Nt,-2) + 200;
    end
    
    x = 1:Nt2;
    
    
    % Roll/ Elevation effect as a state machine
    roll_trans_mtx = [ 0.9995 0.0005; 0.005 0.995];
    roll_state = [1 1]; % initial state
    
    fading_factor = [1; 0.5+.1* randn(num_layers+1,1)]; % Fading factor to be applied to each layer
    lay_power = (lay_means(50,:) ); % 50 is the index for the layer power
    
    rline_wf = zeros(Nt2,num_layers);
    
    echo3 = zeros(Nt2,Nx);
    power_decay_ratio = abs( 0.25 + exp(-(20+10*rand(1))* (0:num_layers-1)/num_layers) + 0.15*rand(1,num_layers));
    
    for iter = 1: size(echo3,2)
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
        
        if gen_layers
            layer_pos_vector = layers_final(:,iter);
        else
            layer_pos_vector =  find( layers_final(:,iter)>0 ); % layers_final(:,iter) ;
        end
        
        rate_time = diff(layer_pos_vector);
        max_depth = max(layer_pos_vector) + 10;
        
        
        %  Generate layers by convolving Gaussian and exponential
        for layer_idx = 1:length(layer_pos_vector) %num_layers
            
            hgt =  lay_power (1) * power_decay_ratio(layer_idx) ;% (pow_idx) ; %*fading_factor (pow_idx) ; gt2 = lay_power (pow_idx+1); % * fading_factor(pow_idx+1) ;
            
            wdt =  randsample([10:15],1); %  width(pow_idx) ;Integer between 1-3
            
            
            % Automatically compute decay rate
            % 0.2 --> Reduce to 20% before the next layer
            if layer_idx < 4
                decay_rt2 = log ( (0.3 + 0.05*rand(1) ) )/ (- (rate_time(layer_idx)) );
            elseif layer_idx > 4 && layer_idx ~= length(layer_pos_vector)
                decay_rt2 = log ( (0.55 + 0.05*rand(1) ) )/ (- (rate_time(layer_idx)) );
                
            end
            
            
            
            pos = layer_pos_vector(layer_idx);
            G = exp( -((x-pos)).^2 / (2*wdt) );  % wdt /sqrt(wdt)  Gaussian -> exp( - ((x-pos).^2)/ (2* sqrt(wdt)) )
            
            
            E =  exp( -1* decay_rt2.*x ); % Exponential
            
            if layer_idx == length(layer_pos_vector)
                E = 0.8 + exp( -1* decay_rt2.*x );
            end
            
            temp_conv = conv(G,E);
            [pk,pk_loc] = max(temp_conv);
            temp_conv = hgt * (temp_conv/pk); % *fading_factor (pow_idx) Re-normalize after conv and scale appropriately
            rline_wf(:,layer_idx) = temp_conv([1:Nt2]+ (pk_loc-pos)) ;
        end
        
        
        rl_scope = sum(rline_wf,2);
        rl_scope( rl_scope < 0 ) = 0; % This is not needed.
        
        
        % Add roll/elevation effect to power from surface to deepest layer
        rl_scope =  roll_val *  rl_scope ;
        
        % Complex thermal noise
        %         therm_wf = sqrt(0.005) .* exp(1:Nt2)';
        %         therm_wf(surf_bin:end) = therm_wf(surf_bin); % Assuming that thermal noise stays constant after surface
        %         therm_noise = repmat( therm_wf,1,N_incoh_avg) .* ( randn(Nt2,N_incoh_avg,1) + 1j* randn(Nt2,N_incoh_avg,1) ) ; %[ 0.0005*ones(surf_bin-1,1); 0.0006*ones(Nt2-surf_bin+1,1)];
        
        therm_noise = repmat( 0.0025,Nt2,N_incoh_avg,1) .* ( randn(Nt2,N_incoh_avg,1) + 1j* randn(Nt2,N_incoh_avg,1) ) ; %[ 0.0005*ones(surf_bin-1,1); 0.0006*ones(Nt2-surf_bin+1,1)];
        % 0.0005
        
        % Generate Chi-squared: using mean from data as variance
        rl_scope = repmat( sqrt(rl_scope),1,N_incoh_avg) .* ( randn(Nt2,N_incoh_avg,1) +1j* randn(Nt2,N_incoh_avg,1) );
        
        echo3(:,iter) = mean( abs((rl_scope +therm_noise).^2) ,2);
        %         echo3(:,iter) =  mean( repmat(rl_scope,1,N_incoh_avg) .* randn(Nt2,N_incoh_avg).^2 ,2)  ;
        
        if 0
            figure(15); hold on; plot(rl_scope);title( sprintf('RL %d rt %i',iter,decay_rt(1:length(layer_pos_vector))) ); grid minor
        end
    end
    
    echo = lp( echo3(1:Nt,:) );
    
    if debug_plot
        figure(60);clf;  imagesc(echo); colormap(1-gray)
    end
    
    %% Normalization
    
    used_max = max(echo(:)) ;
    used_min = min( min(echo(1:surf_bin,:)) ) - 3; % Use just before surface as thermal noise floor as min
    
    norm_echo = uint8( 255*(echo - used_min)/(used_max -used_min) ); % Reduce saturation by using 220 instead of 255??
    
    if debug_plot
        figure(10); clf;
        imagesc(norm_echo); colormap(gray); caxis([0 255]);
        figure(11); plot(norm_echo);
    end
    
    if ~gen_layers
        sim_out_fn = fullfile(out_dir,'data');
        
        if ~exist(sim_out_fn,'dir');
            mkdir(sim_out_fn)
        end
        
        out_fn = fullfile(sim_out_fn,sprintf('data_%06d.mat',echo_idx));
        fprintf('    Saving data to %s\n', out_fn);
        save(out_fn,'norm_echo');
        out_fn = fullfile(sim_out_fn,sprintf('data_%06d.png',echo_idx));
        imwrite(norm_echo,out_fn);
        
    else
        
        if save_output
            %% Create layer raster from layer_rangebin (layers_final)
            raster = zeros(size(norm_echo));
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
            
            
            
            %% Save data and images
            
            tmp_block_data = [];
            tmp_block_data.echo_idx             = echo_idx;
            tmp_block_data.data                 = norm_echo;
            tmp_block_data.layer                = layers_final  ;
            tmp_block_data.f_cutoff1            = f_cutoff;
            tmp_block_data.f_cutoff2            = f_cutoff2;
            tmp_block_data.surf_bin             = surf_bin;
            tmp_block_data.num_layers           = num_layers;
            tmp_block_data.layer_mean_state     = curr_state;
            tmp_block_data.layer_mean_all_state = states;
            tmp_block_data.trend_state = trend_state;
            
            
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
            
            fprintf('Saving %d of  %d\n', echo_idx,num_echograms);
            
            out_fn = fullfile(out_dir,'data',sprintf('data_%06d.mat',echo_idx));
            fprintf('    Save %s\n', out_fn);
            save(out_fn,'-struct','tmp_block_data');
            out_fn = fullfile(out_dir,'data',sprintf('data_%06d.png',echo_idx));
            imwrite(norm_echo,out_fn);
            
            if debug_plot
                out_fn = fullfile(out_dir,'figures',sprintf('data_fig_%06d.png',echo_idx));
                fprintf('    Saving image with layers plotted as %s\n', out_fn);
                saveas(60,out_fn);
                close(60)
            end
            
            out_fn = fullfile(out_dir,'layer',sprintf('layer_%06d.png',echo_idx));
            fprintf('    Save %s\n', out_fn);
            imwrite(raster,out_fn);
            
            out_fn = fullfile(out_dir,'layer_binary',sprintf('layer_binary_%06d.png',echo_idx));
            fprintf('    Save %s\n', out_fn);
            imwrite(logical(raster),out_fn);
        end
    end
    
    trend_state
    
end







































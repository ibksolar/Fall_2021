
% filtered_data = fir_dec(all_data2,ones(1,11)/11,1);


pk_pwr_data = [];
min_pwr_data = [];
dec_data = [];
width_data = [];
decay_data = [];

layer_new = [];

used_Nt = 2000;
used_Nx = 170000;
gap = 15;

% Group layer thickness and compute along-track mean and variance
lay_means = squeeze( nanmean( reshape(layer_thickness(:,1:used_Nx),28,100,[]),2) );
lay_var = squeeze( nanvar( reshape(layer_thickness(:,1:used_Nx),28,100,[]),[],2) );

norm_thck2 = bsxfun(@times, lay_means(2:end,:), (1./lay_means(1:end,:)) );

%% Compute correlation between layer 1 thickness and other layers

full_length_corr = [];
corr_other = [];

lay1 = layer_thickness(1,:);
lay1(isnan(lay1)) = 0;

L2 = 15000; % <==== THIS CAN BE CHANGED 

for corr_idx = 2:size(layer_thickness,1)
    
    other = layer_thickness(corr_idx,:);
    other(isnan(other)) = 0;    
     
    temp_res = corrcoef( lay1,other);
    full_length_corr(end+1) = temp_res(2,1);     
    
    other_short = other(1: floor(length(other)/L2)*L2);
    lay1_short = lay1(1: floor(length(other)/L2)*L2);
    
    lay1_reshaped = reshape( lay1_short,L2,[] );
    other_reshaped = reshape( other_short,L2,[] );    

    for iter2 = 1:size(lay1_reshaped,2)
        temp2_res = corrcoef( lay1_reshaped(:,iter2),other_reshaped(:,iter2) );
        corr_other(iter2,corr_idx-1) = temp2_res(2,1);
    end
    
end
    




count = 0;

for iter3 = 1:10:used_Nx
    count = count+1;
    
%     res = nanmean( all_data2(1:used_Nt, iter3:iter3+20),2);
    res = filtered_data(1:used_Nt, iter3);
    dec_data(:,end+1) = res;
    
    [~,lay_loc] = findpeaks(res,'MinPeakDistance',40);
    lay_loc = lay_loc(lay_loc>=100);
    
    layer_new(1:length(lay_loc),end+1) = lay_loc; 
    
    for iter4 = 1:length(lay_loc)
        
        % Create search window
        if iter4 == 1 || iter4 == length(lay_loc)
            if iter4 == 1
                start = max(1, lay_loc(iter4)-gap);
                stop = min([used_Nt, lay_loc(iter4)+gap, lay_loc(iter4+1)-gap]);
            end
             if iter4 == length(lay_loc)
                stop = min([used_Nt, lay_loc(iter4)+gap]); 
                start = max([1, lay_loc(iter4-1)+10, lay_loc(iter4)-gap]);
            end
            
        else
            start = max([1, lay_loc(iter4-1)+10, lay_loc(iter4)-gap]);
            stop = min([used_Nt, lay_loc(iter4)+gap, lay_loc(iter4+1)-gap]);
        end
        
        %% Perform search
        if length(start:stop) < 5
            keyboard;
        end
        
        % Peak power
        [pk,pk_loc] = max(res(start:stop));
        pk_pwr_data(iter4,count) = pk;
        
        % Width data
        w1 = find(res(start + pk_loc-1: -1: start-gap) < .45*pk,1,'first'); %0.37
        w2 = find(res(start + pk_loc-1: min(stop+gap,used_Nt)) < .45*pk,1,'first');
        
        if ~isempty(w1) && ~isempty(w2)
            width_data(iter4,count) = w2 + w1;
        else
            width_data(iter4,count) = nan;
        end
        
        % Decay data
        [min_pwr_data(iter4,count),decay_data(iter4,count)] = min(res(start+pk_loc-1: stop));
        
        
    end
    
end

norm_thickness = bsxfun( @times, layer_thickness(2:end, :),(1./layer_thickness(1,:)) );
[ nanmean(norm_thickness(:)) nanstd(norm_thickness(:)) ]

% Get distribution for different Layer1 thickness brackets
% <40 | 40-60 | 60-80 | 80-100
if 0
    loc = layer_thickness(1,:) <= 40; w = layer_thickness(:,loc); figure; hist(w(:),50); title('Layers with spacing less than 40'); w = norm_thickness(:,loc); [nanmean(w(:)) nanstd(w(:))]
    loc = layer_thickness(1,:) > 40 & layer_thickness(1,:) <= 60; w = layer_thickness(:,loc); figure; hist(w(:),50); title('Layers with spacing less than 40'); w = norm_thickness(:,loc); [nanmean(w(:)) nanstd(w(:))]
    loc = layer_thickness(1,:) > 60 & layer_thickness(1,:) <= 80; w = layer_thickness(:,loc); figure; hist(w(:),50); title('Layers with spacing greater than 60 but less than 80'); w = norm_thickness(:,loc); [nanmean(w(:)) nanstd(w(:))]
    loc = layer_thickness(1,:) > 80; w = layer_thickness(:,loc); figure; hist(w(:),50); title('Layers with spacing greater than 60 but less than 80'); w = norm_thickness(:,loc); w = w(abs(w) < 200);[nanmean(w(:)) nanstd(w(:))]
end

min_pwr_data(min_pwr_data == 0) = nan; % Why did I do this?
decay_data(decay_data==0) = nan;
pk_pwr_data(pk_pwr_data == 0) = nan;

norm_pwr = pk_pwr_data;
norm_pwr(norm_pwr == 0) = nan;

norm_pwr = bsxfun( @times, norm_pwr(2:end,:), 1./(norm_pwr(1,:)) );
norm_pwr2 = norm_pwr;
norm_pwr2(norm_pwr2 >10) = nan;

% Get distribution of each layer
lay_pwr_dist_param = [];
lay1 = fitdist(pk_pwr_data(1,:)','Rayleigh');
lay_pwr_dist_param(end+1) = lay1.B;

for iter_idx = 1:size(norm_pwr,1)
    if ~all(isnan( pk_pwr_data(iter_idx,:)) )        
        pd = fitdist( pk_pwr_data(iter_idx,:)','Rayleigh');
        lay_pwr_dist_param(end+1) = pd.B;
    end
end

lay_pwr_dist =[];
for iter_idx = 1:size(norm_pwr,1)
    if ~all(isnan( pk_pwr_data(iter_idx,:)) )        
        lay_pwr_dist(iter_idx).pd = fitdist( pk_pwr_data(iter_idx,:)','Rayleigh');%         
    end
end

pwr_ratio = min_pwr_data./pk_pwr_data;
decay_rates = log(pwr_ratio)./-decay_data;

mean_decay_rate = nanmean(decay_rates,2);
std_decay_rate = nanstd(decay_rates,[],2);
        
mean_pwr_ratios = nanmean(pwr_ratio,2);          
std_pwr_ratios = nanstd(pwr_ratio,[],2);          

out_fn = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/Aug2021_new_images_and_data/nsf_report_data.mat';

save(out_fn,'lay_pwr_dist_param','mean_decay_rate','std_decay_rate','mean_pwr_ratios','std_pwr_ratios');





    
    
    
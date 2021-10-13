

% lora_data_snow_simulator_stat

% Collect layer thickness stat
% Collect power stat

clearvars -except AdditionalPaths gRadar
clc;

data_source = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/greenland_picks_final_2009_2012_reformat/';
year = '2012';
save_path = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/Aug2021_new_images_and_data/lora_data_stat';

data_files = get_filenames(fullfile(data_source,year),'data','20','.mat');
layer_files = get_filenames(fullfile(data_source,year),'layer','20','.mat');

all_data = [];

all_layers = nan(30,1);

all_pow_data = nan(30,1);

for iter = 1: length(data_files)
    
    img = load( data_files{iter} );
    lay = load( layer_files{iter} );
    
    curr_echo = img.data;
    curr_layer = lay.layer;
    
    [Nt,Nx] = size(curr_echo);
    
    curr_layer = logical(curr_layer);
    
    % Layer rangeline
    lay_rlines = nan(30,1);
    pow_data = nan(30,1);
    
    for iter2 = 1:Nx
        curr_rlines = find(curr_layer(:,iter2));       
        
        try
            lay_rlines(1:length(curr_rlines),end+1) = curr_rlines;
            pow_data(1:length(curr_rlines),end+1) = curr_echo(curr_rlines,iter2);
        catch
            %pass
        end
    end
    
    all_layers = [all_layers lay_rlines];
    all_pow_data = [all_pow_data pow_data];
    % all_data = [all_data curr_echo]; % might not need this( RAM issues)
    
end
    
    
all_layers2 = all_layers;
all_layers2( all_layers2 ==0) = nan;
thickness = diff(all_layers2);

norm_thickness = bsxfun(@times, thickness ,(1./thickness(1,:)));

































% interpolate_layers_with_weather_data

% Attempt to fill up missing gaps in layer ground truth


base_path = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/greenland_picks_final_2009_2012_reformat_weather_more_fields';

base_dir = dir(base_path);
rm = ismember( {base_dir.name},{'.','..'}); % remove current and base dir
base_dir(rm) = [];

files_to_check = {};
rows_to_check = {};

pos1 = [362 439 565 424] ;
pos2 = [1132 459 565 424];

for folder_idx = 1 : length(base_dir) 
    curr_path = fullfile( base_path,base_dir(folder_idx).name );
    echo_fns = get_filenames(curr_path,'data','_','.mat');
    
    for echo_idx = 1: length(echo_fns)
        temp = load(echo_fns{echo_idx});
        [~,echo_name] = fileparts(echo_fns{echo_idx});
        
        w_layers = temp.weather_layers;
        
        if all(all(isnan(w_layers))) 
            % skip weather layers that are all NaNs
            continue;
        else
            curr_layer = double(temp.layer.layer);
            
            curr_layer = bsxfun(@times, curr_layer,[1:size(curr_layer,1)]');
            
            flat_layer = [];
            for iter_idx = 1:size(curr_layer,2)
                interim = find(curr_layer(:,iter_idx));
                flat_layer(1:length(interim),iter_idx) = interim;
            end            
        end
        
        h = figure(1); clf;
        imagesc(temp.echo.data); colormap(1-gray);
        hold on; plot(flat_layer');
        set(h,'Position',pos1);
        
        h2 = figure(2); clf;
        imagesc(temp.echo.data); colormap(1-gray);
        hold on; plot(w_layers');
        set(h2,'Position',pos2);
        figure(1); figure(2);
        close all;
        
    end

end

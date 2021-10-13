

tmp = load('/cresis/snfs1/dataproducts/ct_data/snow/2016_Greenland_P3/CSARP_post/CSARP_qlook/20160519_04/Data_20160519_04_196.mat');
tmp_lay = load('/cresis/snfs1/scratch/ibikunle/ct_user_tmp/JSTARRS2021_Sep21/snow/20160519_04/Data_20160519_04_196.mat');


new_lay = nan( size(tmp_lay.twtt,1), length(tmp.GPS_time) );

 for iter_idx3 = 1: size(new_lay,1); 
     if ~all(isnan(tmp_lay.twtt(iter_idx3,:)))
     
%          fprintf('Layer %d \n',iter_idx3);        
         new_lay(iter_idx3,:) = interp1(tmp_lay.gps_time, tmp_lay.twtt(iter_idx3,:), tmp.GPS_time);
     end
 end

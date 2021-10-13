close all
clearvars -except gRadar AdditionalPaths
clc;

param = [];
param.radar_name = 'snow';
param.season_name = '2012_Greenland_P3';
param.out = 'CSARP_post/qlook';
% frms ={'20120330_01_001','20120330_01_003','20120330_01_005','20120330_01_006','20120330_01_019','20120330_01_022',...
%   '20120330_02_002','20120330_02_034','20120330_02_055','20120330_04_001','20120330_04_002','20120330_04_005','20120330_04_025',...
%   '20120330_04_053','20120330_04_058','20120330_04_060','20120330_04_062','20120404_01_047','20120404_01_048','20120404_01_049',...};

img_path0 = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/greenland_picks_final_2009-2012_20140602/2012/data_mat/';
img_path = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/greenland_picks_final_2009_2012_reformat/2012/';
save_path = '/cresis/snfs1/scratch/ibikunle/ct_user_tmp/sim_prep4/Aug2021_new_images_and_data/';

% {reformated frm--> dataproduct frm}:
% {20090401_01_503-->20090401_05_147(lora's dataset);20090501_01_015-->20090501_01_047},
% {20100324_01_001-->20200324_01_001(DC8),20100507_01_031-->20100507_01_019(P3)(lora's dataset);20100507_01_013-->20100507_01_055}
% {20110407_01_002-->20110407__01_005(lora's datasets);20110411_01_043-->20110411_01_171},{20120330_01_006-->20120330_01_049},{20130409_01_031-->20130409_01_326}
% {20140408_01_01-->20140408__01_094},{20150408_03_011-->20150408_03_367},{20160512_03_086-->20160512_03_350}
frms ={};
for frm_idx = 1:15 % Get just 15 frames
%   frms{frm_idx} = sprintf('20110411_01_%03d',frm_idx);
  frms{frm_idx} = sprintf('20120330_04_%03d',frm_idx);
end

% img_path0 = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/greenland_picks_final_2009-2012_20140602/2013/data_mat/';
% img_path = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/greenland_picks_final_2009_2012_reformat/2013/';
% img_path0 = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/Corrected_SEGL_picks_lnm_2009_2017/2011_data_mat/';
% img_path = '/cresis/snfs1/dataproducts/public/data/temp/internal_layers/NASA_OIB_test_files/image_files/Corrected_SEGL_picks_lnm_2009_2017_reformat/2011/';


for idx = 3:length(frms)
    
  % Load Lora reformat data  
  img_fn = sprintf('%sdata_%s.mat',img_path,frms{idx});
  img = load(img_fn);   % reformated data
  [Nt1,Nx1] = size(img.data);
  echo_str = [frms{idx}];
  
  figure(1);clf; 
  imagesc(img.data);colormap(1-gray);
  title(['Lora reformatted ' echo_str],'Interpreter','none');  
  
  
  % Load Lora data before reformating
  img0 = load(img.fn);  % data before reformating  
  if 0
      figure(2);clf; imagesc(img0.data_out);
      imagesc(img.data);colormap(1-gray);
      title(sprintf('Lora reformatted %s',echo_str,'Interpreter','none'));
  end
  
  % Get rlines
  rlines = img.rline:img.rline+size(img.data,2)-1;
  gps_times = img0.time_gps(rlines);
  
  param.start.gps_time = gps_times(1);
  param.stop.gps_time = gps_times(end);
  
  % Load using gps_time
  img1 = load_data_by_gps_time(param);  
  surf_bin = round( interp1(img1.Time, 1:length(img1.Time), img1.Surface) );
  
  % flatten original data
  tmp_data = img1.Data; [Nt,Nx] = size(tmp_data);  
  tmp_data(Nt+1:Nt+1000,:) = 0; % zero pad for shift  
  flat_data = [];
  
  shift = 55 - surf_bin; % Surface now aligned to bin 55 (surface from Lora's data)
  for iter = 1:length(surf_bin)
      flat_data(:,end+1) = circshift(tmp_data(1:11000,iter),shift(iter));
  end
  
  figure(3); imagesc(lp(flat_data));colormap(1-gray);
  title(['CReSIS data ',echo_str],'Interpreter','none');
  ylim([0 Nt1]) ; xlim([1 Nx1]);
   
  linkaxes([1 3]);
  
%   out_fn = sprintf('%simg_%s',save_path,frms{idx});
%   print('-f1',out_fn,'-djpeg','-r300');
%   figure(3);imagesc(lp(img1.Data));colormap(gray);
  pause
%   saveas(3,out_fn);
%   figure(3);clf,plot(img1.GPS_time,img1.Latitude);
%   hold on; plot(img0.time_gps(rlines),img0.lat(rlines),'ro');
%   figure(4);clf,plot(img1.GPS_time,img1.Longitude);
%   hold on; plot(img0.time_gps(rlines),img0.lon(rlines),'ro');
%   param.day_seg = frms{idx}(1:end-4);
%   fn = sprintf('%s/Data_%s.mat',ct_filename_out(param,'qlook'),frms{idx});
%   tmp = load(fn);
%   figure(1);clf;imagesc(lp(tmp.Data));
end
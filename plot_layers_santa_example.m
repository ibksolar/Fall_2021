%% plot_layers_santa_example.m
% clear all
% close all

%% Load raw CReSIS .mat data
load /Users/toverly/nasa/firn/data_local/snow/mat_from_CReSIS/2016/Data_20160519_04_249.mat
% load /Users/toverly/nasa/firn/data_local/snow/mat_from_CReSIS/2016/Data_20160519_04_250.mat
% load /Users/toverly/nasa/firn/data_local/snow/mat_from_CReSIS/2016/Data_20160519_04_251.mat

%% Load layers traced by Santa & the CReSIS 'layers_FLIGHTDATE_SEGMENT.mat' data
% Traced Layers by Santa:
load /Users/toverly/nasa/paden/NASA_LayerTracing/layers_from_Santa/greenland/layers_20160519_04/Data_20160519_04_249.mat
% CReSIS Layer information files from: 
%  https://data.cresis.ku.edu/data/snow/2016_Greenland_P3/CSARP_layer/20160519_04/
load /Users/toverly/nasa/paden/NASA_LayerTracing/layers_from_Santa/greenland/layers_20160519_04/layer_20160519_04.mat
% load /Users/toverly/nasa/paden/NASA_LayerTracing/Santa_Layer_Picks/layers_2016051904/Data_20160519_04_250.mat
% load /Users/toverly/nasa/paden/NASA_LayerTracing/Santa_Layer_Picks/layers_2016051904/Data_20160519_04_251.mat

%% Get length of Row & Column dimensions of radar Data
[len_row,len_col]=size(Data);
 
%% Plot colors
% Color Near cyan
% skyblue=[0.2 0.6 1];
% skyblue=[0 0.8 1]; % also skyblue
% darkblue=[0.2 0 1]; % darker blue 
skyblue=[0.6 0.4 0.8]; % blue purple

%% Figure settings
% Set background color
figure(1)
%--------------------------%
%---  Right 2 of 1x2 ---%
axL=subplot(1,2,1); % UL = Upper Right
hold on;

FigBgColor=[1,1,1];
set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');
%set(gcf,'Interpreter','latex'); NOT WORKING
set(gca,'FontName','Lucida Grande')
set(gca,'YDir','reverse')
title('Raw CReSIS Snow Radar')
%  plot(rho_s_array,k_extArr);
hold on;
% plot(rho_s_array,k_extArr,'d','markersize',20);
%plot(rho_s_array,k_ext_test,'linewidth',3)
% hold on;



xAxis=1:length(len_col);
yAxis_Time=Time(Truncate_Bins(1:len_row)); % fasttime in microseconds
%     yAxisRows=1:length(fasttime);

imgAmpFasttime=imagesc(xAxis,yAxis_Time,Data);
%    imgAmpFasttime=imagesc(amplitudeEdit);
% plot(twtt)
% % % % imagesc(Data,[0 500])
axFasttime=gca;
% % % %    axFasttime=axes('Position',get(yAxisFasttime,'Position'),'YAxisLocation','left','YDir','reverse','YColor','m');
% % % 
% set(axFasttime,'YColor','r','YAxisLocation','right','YDir','reverse');
% % % %    set(axFasttime,'YLabel','fasttime')
set(axFasttime,'XLim',[0 len_col])
set(axFasttime,'YLim',[Time(Truncate_Bins(1)) Time(Truncate_Bins(3845))])

xlabel('Columns','FontSize',36,'Interpreter','latex','FontSmoothing','on')
ylabel('Radar Time ($\mu$ seconds)','FontSize',36,'Interpreter','latex','FontSmoothing','on')
set(gca,'FontSize',24)

set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');


%---  Left 1 of 1x2 ---%
axR=subplot(1,2,2); % UL = Upper Right
hold on;

FigBgColor=[1,1,1];
set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');
%set(gcf,'Interpreter','latex'); NOT WORKING
set(gca,'FontName','Lucida Grande')
set(gca,'YDir','reverse')
title("Image Processed Radar")
%  plot(rho_s_array,k_extArr);
hold on;
% plot(rho_s_array,k_extArr,'d','markersize',20);
%plot(rho_s_array,k_ext_test,'linewidth',3)
hold on;

% imagesc(Data*10e9,[10e8 10e10])
imagesc(Data,[0 125])
% imagesc(lp(Data))
set(gca,'YLim',[1 3845])
set(gca,'XLim',[1 463])
xlabel('Columns','FontSize',36,'Interpreter','latex','FontSmoothing','on')
ylabel('Rows','FontSize',36,'Interpreter','latex','FontSmoothing','on')
set(gca,'FontSize',24)

set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');

%% Figure settings
% Set background color
figure(2)

%--------------------------%
%---  Left 1 of 1x2 ---%
axL=subplot(1,2,1); % UL = Upper Right
hold on;

FigBgColor=[1,1,1];
set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');
%set(gcf,'Interpreter','latex'); NOT WORKING
set(gca,'FontName','Lucida Grande')
set(gca,'YDir','reverse')
title("Raw CReSIS Snow Radar")
%  plot(rho_s_array,k_extArr);
hold on;
% plot(rho_s_array,k_extArr,'d','markersize',20);
%plot(rho_s_array,k_ext_test,'linewidth',3)
hold on;

imagesc(Data,[0 125])


xlabel('Column','FontSize',36,'Interpreter','latex','FontSmoothing','on')
ylabel('Row','FontSize',36,'Interpreter','latex','FontSmoothing','on')
set(gca,'FontSize',24)

set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');

set(gca,'YLim',[400 1300])
set(gca,'XLim',[200 450])

%---  Left 1 of 1x2 ---%
axR=subplot(1,2,2); % UL = Upper Right
hold on;

FigBgColor=[1,1,1];
set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');
%set(gcf,'Interpreter','latex'); NOT WORKING
set(gca,'FontName','Lucida Grande')
set(gca,'YDir','reverse')
title("Image Processed Radar")
%  plot(rho_s_array,k_extArr);
hold on;
% plot(rho_s_array,k_extArr,'d','markersize',20);
%plot(rho_s_array,k_ext_test,'linewidth',3)
hold on;

data400=Data(1:1500,300);
plot(data400,'LineWidth',1.1)
title('Radar waveform at column 300')
view(90,90)

xlabel('Row','FontSize',36,'Interpreter','latex','FontSmoothing','on')
ylabel('Waveform power / column','FontSize',36,'Interpreter','latex','FontSmoothing','on')
set(gca,'FontSize',24)
set(gca,'YLim',[1 250])
set(gca,'XLim',[400 1300])
grid on
% % %% Figure 2
% % figure(3);
% % data100=Data(:,100);
% % plot(data100)
% % title('Radar shot 100, a.k.a. "Trace 100"')
% % view(90,90)


%% Paden Load & Plot example:
%%% ORGINAL
% % % % % Load echogram
% % % % mdata = load_L1B('D:\snow\2016_Greenland_P3\CSARP_qlook\20160519_04\Data_20160519_04_216.mat');
% % % % % Load layers
% % % % surf = layerdata.load_layers(mdata,'','surface');
% % % % layer_names = {'snow_001','snow_002','snow_003'};
% % % % layer_list = cell(size(layer_names));
% % % % [layer_list{:}] = layerdata.load_layers(mdata,'',layer_names{:});
% % % % % Plot
% % % % figure
% % % % imagesc([],mdata.Time,lp(mdata.Data))
% % % % hold on
% % % % plot(surf,'m')
% % % % plot(layer_list{1},'r')
% % % % plot(layer_list{2},'g')
% % % % plot(layer_list{3},'b')
% % % % ylim([2.9e-6 3e-6])

%% Modified Load & Plot example: 
mdata = load_L1B('/Users/toverly/nasa/firn/data_local/snow/mat_from_CReSIS/2016/Data_20160519_04_249.mat');
% Load layers
% surf = layerdata.load_layers(mdata,'','Surface');
surf = mdata.Surface;
layer_names = lyr_name;
layer_list = cell(size(layer_names));
% [layer_list{:}] = layerdata.load_layers(mdata,'',layer_names{:});
[layer_list{:}] = layer_names{:};

% Notes from John
% % master is the echogram
% % layers(lay_idx) is the layer data
% % layers(lay_idx).twtt_ref = interp1(master.GPS_time, layers(lay_idx).gps_time, layers(lay_idx).twtt, 'spline');
% % layers(lay_idx).quality = interp1(master.GPS_time, layers(lay_idx).gps_time, layers(lay_idx).twtt, 'nearest');
% % layers(lay_idx).type = interp1(master.GPS_time, layers(lay_idx).gps_time, layers(lay_idx).twtt, 'nearest');
twtt_interp=[];
twtt_interp_temp=[];
[len_row_twtt,len_col_twtt]=size(twtt);

%NaNs
% twtt_int2 = interp1(gps_time, twtt(8,4:end),GPS_time, 'spline');
% twtt_int8 = interp1(gps_time, twtt(8,4:end),GPS_time, 'spline');
% twtt_int11 = interp1(gps_time, twtt(11,4:end),GPS_time, 'spline');

twtt_int1 = interp1(gps_time, twtt(1,:),GPS_time, 'spline');
% twtt_int2 = interp1(gps_time, twtt(2,:),GPS_time, 'spline');  SKIP! 2 has
% NaN's... unclear why
twtt_int3 = interp1(gps_time, twtt(3,:),GPS_time, 'spline');
twtt_int4 = interp1(gps_time, twtt(4,:),GPS_time, 'spline');
twtt_int5 = interp1(gps_time, twtt(5,:),GPS_time, 'spline');
twtt_int6 = interp1(gps_time, twtt(6,:),GPS_time, 'spline');
twtt_int7 = interp1(gps_time, twtt(7,:),GPS_time, 'spline');
twtt_int8 = interp1(gps_time(4:end), twtt(8,4:end),GPS_time, 'spline'); % has NaN's
twtt_int9 = interp1(gps_time, twtt(9,:),GPS_time, 'spline');
twtt_int10 = interp1(gps_time, twtt(10,:),GPS_time, 'spline');
twtt_int11 = interp1(gps_time(4:end), twtt(11,4:end),GPS_time, 'spline'); % has NaN's
twtt_int12 = interp1(gps_time, twtt(12,:),GPS_time, 'spline');
twtt_int13= interp1(gps_time, twtt(13,:),GPS_time, 'spline');
twtt_int14= interp1(gps_time, twtt(14,:),GPS_time, 'spline');
twtt_int15= interp1(gps_time, twtt(15,:),GPS_time, 'spline');
twtt_int16= interp1(gps_time, twtt(16,:),GPS_time, 'spline');
twtt_int17 = interp1(gps_time, twtt(17,:),GPS_time, 'spline');
twtt_int18 = interp1(gps_time, twtt(18,:),GPS_time, 'spline');
twtt_int19= interp1(gps_time, twtt(19,:),GPS_time, 'spline');
twtt_int20= interp1(gps_time, twtt(20,:),GPS_time, 'spline');
twtt_int21 = interp1(gps_time, twtt(21,:),GPS_time, 'spline');
twtt_int22= interp1(gps_time, twtt(22,:),GPS_time, 'spline');

%% Loop
% for lay_idx=1:len_row_twtt
%     twtt_interp_temp = interp1(gps_time, twtt(lay_idx,~isnan(twtt)),mdata.GPS_time, 'spline');
%     twtt_interp_temp = interp1(gps_time, twtt(5,:),GPS_time, 'spline');
%     twtt_interp=[twtt_interp;twtt_interp_temp];
% %     twtt(lay_idx) = interp1(mdata.GPS_time, gps_time(lay_idx), twtt(lay_idx), 'nearest');
% %     twtt(lay_idx) = interp1(mdata.GPS_time, gps_time(lay_idx), twtt(lay_idx), 'nearest');
% end

% % twtt_interp(1) = interp1(gps_time(1), twtt(:,1),mdata.GPS_time(1), 'spline');
% % quality_interp = interp1(mdata.GPS_time, twtt, gps_time, 'nearest');
% % type_interp = interp1(mdata.GPS_time, twtt, gps_time, 'nearest');

% Plot
figure
imagesc([],mdata.Time,lp(mdata.Data))
hold on
% plot(surf,'m')

plot(twtt_int1,'Color',skyblue,'LineWidth',1.2)
% plot(twtt_int2,'Color',skyblue,'LineWidth',1.2)  % NaN's
plot(twtt_int3,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int4,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int5,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int6,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int7,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int8,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int9,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int10,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int11,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int12,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int13,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int14,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int15,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int15,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int16,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int17,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int18,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int19,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int20,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int21,'Color',skyblue,'LineWidth',1.2)
plot(twtt_int22,'Color',skyblue,'LineWidth',1.2)

ylim([2.94e-6 3.18e-6])

xlabel('Distance (km)','FontSize',36,'Interpreter','latex','FontSmoothing','on')
ylabel('Two-way Travel Time ($\mu$s)','FontSize',36,'Interpreter','latex','FontSmoothing','on')
set(gca,'FontSize',24)

set(gcf,'color',FigBgColor);
set(gcf,'InvertHardCopy','off');

% dimensions in rows/cols, not eventual labels of meters & km
% Surface @ ~130
% colormap('bone')
% legend()

% xticks([1 769 1538 2307 3076 3845])
% % xticklabels({'0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5'})
% xticklabels({'0','1','2','3','4','5'})

% yticks([131 253 379 503 627 752 876 1001 1126])
% yticklabels({'0','1','2','3','4','5','6','7','8'})

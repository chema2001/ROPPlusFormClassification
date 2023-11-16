% *Algorithm to generate Optical Disk Mask of Retinal Images* 
%
%     Developers: José Almeida
%     VSB - Technical University of Ostrava, 2023
% 
%      The input should be a RGB Retinal Image. It uses an
%      morphologic-based method for the automatic recognition of an OD
%      point and then uses an Active Countour approach to generate the
%      mask. The program saves a .MAT file with both binnary OD Mask image
%      and the coordinates for the center point. 


clc; clear all; close all;

addpath('ODExternalFunctions\')

% Image Selection
img_name = "Control028_Serie2_4.jpg";
img_path = "Retinal_Images\Images\" + img_name;
image = im2double(imread(img_path));

segImg = imbinarize(imread("Retinal_Images\Segmentation_Combined\Seg_Control028_Serie2_4.jpg"));


RetCam = 1; % RetCam and ICON images have some differences
metric_th = 0.3;
[mask, center, radii] = OpticalDiskMask(image,RetCam, metric_th);

repRGB = cat(3, 255*segImg, 255*(segImg & ~mask), 255*(segImg & ~mask));
figure; imshow(repRGB); 
%hold on; plot(center(1), center(2), 'b*'); hold off

imgId = split(img_name, '.');
imgId = imgId(1);

matFileName = "ODMask_" + imgId + ".mat";
% save(matFileName, "mask", "center");

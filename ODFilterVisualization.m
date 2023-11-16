% *Visualize the ApplyODSegmentation.m results for a single image* 
%
%     Developers: José Almeida
%     VSB - Technical University of Ostrava, 2023
% 
%      Code used to test the ApplyODSegmentation.m algorithm to a single
%      image in order to stablish the best parameters for each set of
%      images.


clc; clear all; close all;

% Image Selection
img_name = "Control049_Serie2_4.jpg";
img_path = "Retinal_Images\Images\" + img_name;
image = im2double(imread(img_path));

segImg = imread("Retinal_Images\Segmentation_Combined\Seg_" + img_name);
segImg = imbinarize(segImg);

RetCam = 1;
dist_th = 25;
axisRatio_th = 3.5;
area_th = 650;
RetCam = 1;
ODMask = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th);

maskRGB = cat(3, 255*segImg, 255*(segImg & ODMask), 255*(segImg & ODMask));
figure
imshowpair(segImg, maskRGB, 'montage');
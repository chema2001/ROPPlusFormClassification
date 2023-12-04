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
img_name = "Control028_Serie1_2.jpg";
img_path = "Retinal_Images\Images\" + img_name;
image = im2double(imread(img_path));

segImg = imread("Retinal_Images\Segmentation_Combined\JPG\Seg_" + img_name);
segImg = imbinarize(segImg);

dist_th = 30;
axisRatio_th = 4.5;
area_th = 1500;
RetCam = 1;
windowSize = 50;
ODMask = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th, windowSize);

maskRGB = cat(3, 255*segImg, 255*(segImg & ODMask), 255*(segImg & ODMask));
figure
imshowpair(segImg, maskRGB, 'montage');
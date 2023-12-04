clc; clear all; close all;

% Path for the external Filter functions used
addpath('frangi_filter_version2a\');
addpath('jerman_filter\')

img_folder = "Retinal_Images\Images\";
img_name = "Control028_Serie2_5.jpg";
img_path = img_folder + img_name;
image = im2double(imread(img_path));

segImg = segmentRetinalImage(image,1);

%figure; imshowpair(image, segImg, 'montage')

dist_th = 25;
axisRatio_th = 3;
area_th = 1500;
RetCam = 1;
windowSize = 40;
ODMask = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th, windowSize);

%figure; imshowpair(image, ODMask, 'montage')

skel = bwskel(ODMask);

%figure; imshowpair(image, skel, 'montage');
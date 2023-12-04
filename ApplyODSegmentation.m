% *Algorithm to Eliminate Artifacts due to OD in Retinal Blood Vessel Segmentation* 
%
%     Developers: José Almeida
%     VSB - Technical University of Ostrava, 2023
% 
%      This program uses both the original image and the segmentation
%      result. It starts by looking for the most probable location of a OD
%      center in the image. It then creates a window and looks for
%      OD associated structures with specific characteristics that could 
%      have been segmented wrongly and eliminates them.


clc; clear all; close all;

% Path for the external Filter functions used
addpath('frangi_filter_version2a\');
addpath('jerman_filter\')
addpath('ODExternalFunctions\')

% Folder Selection
img_folder = "ICON_Phoenix\P1\";
seg_folder = "ICON_Phoenix\P1\Segmentation_Results\JPG\";
files = dir(img_folder);
image_files = files(contains({files.name}, {'.jpg', '.png', '.bmp', '.tif'}));

for i=1:length(image_files)
    img_name = img_folder + image_files(i).name;
    image = im2double(imread(img_name));
    seg_name = seg_folder + "Seg_" + image_files(i).name;
    segImg = imread(seg_name);
    segImg = imbinarize(segImg);
    
    dist_th = 25;
    axisRatio_th = 3;
    area_th = 1500;
    RetCam = 0;
    windowSize = 40;
    ODMask = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th, windowSize);

    maskRGB = cat(3, 255 * segImg, 255*(segImg & ODMask), 255*(segImg & ODMask));
    ODMaskName = "ICON_Phoenix\P1\ODFiltered1\Visual\OD_" + image_files(i).name;
    imwrite(maskRGB, ODMaskName);

    ODFilteredImageName = "ICON_Phoenix\P1\ODFiltered1\JPG\OD_" + image_files(i).name; 
    imwrite(ODMask, ODFilteredImageName);

    image_id = split(image_files(i).name, '.');
    matOD_name = "ICON_Phoenix\P1\ODFiltered1\MAT\OD_" + image_id(1) + ".mat";
    save(matOD_name, "ODMask");
end
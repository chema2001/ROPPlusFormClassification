clc; clear all; close all;

% Path for the external Filter functions used
addpath('frangi_filter_version2a\');
addpath('jerman_filter\')

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

    ODMask = FilterOpticalDisk_new(image, segImg, 0); % Change the last term accordingly

    maskRGB = cat(3, 255 * segImg, 255*(segImg & ODMask), 255*(segImg & ODMask));
    ODMaskName = "ICON_Phoenix\P1\ODMasks\Visual\OD_" + image_files(i).name;
    imwrite(maskRGB, ODMaskName);

    ODFilteredImageName = "ICON_Phoenix\P1\ODMasks\JPG\OD_" + image_files(i).name; 
    imwrite(ODMask, ODFilteredImageName);

    image_id = split(image_files(i).name, '.');
    matOD_name = "ICON_Phoenix\P1\ODMasks\MAT\OD_" + image_id(1) + ".mat";
    save(matOD_name, "ODMask");
end
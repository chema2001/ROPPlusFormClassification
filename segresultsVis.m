clc; clear all; close all;

image1 = imread("DRIVE_Dataset\training\images\22_training.tif");
gt1 = imread("DRIVE_Dataset\training\1st_manual\22_manual1.tif");
seg1 = imread("DRIVE_Dataset\DRIVE_Segmentation_Results\JPG\Seg_22_training.jpg");

image2 = imread("DRIVE_Dataset\training\images\24_training.tif");
gt2 = imread("DRIVE_Dataset\training\1st_manual\24_manual1.tif");
seg2 = imread("DRIVE_Dataset\DRIVE_Segmentation_Results\JPG\Seg_24_training.jpg");

image3 = imread("DRIVE_Dataset\training\images\33_training.tif");
gt3 = imread("DRIVE_Dataset\training\1st_manual\33_manual1.tif");
seg3 = imread("DRIVE_Dataset\DRIVE_Segmentation_Results\JPG\Seg_33_training.jpg");

image4 = imread("DRIVE_Dataset\training\images\37_training.tif");
gt4 = imread("DRIVE_Dataset\training\1st_manual\37_manual1.tif");
seg4 = imread("DRIVE_Dataset\DRIVE_Segmentation_Results\JPG\Seg_37_training.jpg");

imgArray = {image1 image2 image3 image4; gt1 gt2 gt3 gt4; seg1 seg2 seg3 seg4};
montage(imgArray,'size', [4 3])
ax = gca;
ax.PositionConstraint = "outerposition";
title('Original Image             Ground Truth        Segmentation Result', 'FontSize', 17)
clc; clear all; close all;

% retcam_folder = "Retinal_Images\Images\";
% retcam1 = imread(retcam_folder+"Control033_Serie3_10.jpg");
% retcam2 = imread(retcam_folder+"ROP040_Serie12_7.jpg");
% retcam3 = imread(retcam_folder+"ROP031_Serie2_1.jpg");
% retcam4 = imread(retcam_folder+"ROP031_Serie7_2.jpg");

icon1 = imread("DRIVE_Dataset\training\images\21_training.tif");
icon2 = imread("DRIVE_Dataset\training\images\32_training.tif");
icon3 = imread("DRIVE_Dataset\training\images\31_training.tif");
icon4 = imread("DRIVE_Dataset\training\images\35_training.tif");

seg1 = imbinarize(imread("DRIVE_Dataset\Segmentation_Results\JPG\Seg_21_training.jpg"));
seg2 = imbinarize(imread("DRIVE_Dataset\Segmentation_Results\JPG\Seg_32_training.jpg"));
seg3 = imbinarize(imread("DRIVE_Dataset\Segmentation_Results\JPG\Seg_31_training.jpg"));
seg4 = imbinarize(imread("DRIVE_Dataset\Segmentation_Results\JPG\Seg_35_training.jpg"));

gt1 = imbinarize(imread("DRIVE_Dataset\training\1st_manual\21_manual1.tif"));
gt2 = imbinarize(imread("DRIVE_Dataset\training\1st_manual\32_manual1.tif"));
gt3 = imbinarize(imread("DRIVE_Dataset\training\1st_manual\31_manual1.tif"));
gt4 = imbinarize(imread("DRIVE_Dataset\training\1st_manual\35_manual1.tif"));

R1 = icon1(:,:,1); R1(seg1) = 0;
G1 = icon1(:,:,2); 
B1 = icon1(:,:,3);
fusedImg1 = cat(3, R1,G1,B1);

R2 = icon2(:,:,1); R2(seg2) = 0;
G2 = icon2(:,:,2); 
B2 = icon2(:,:,3);
fusedImg2 = cat(3, R2,G2,B2);

R3 = icon3(:,:,1); R3(seg3) = 0;
G3 = icon3(:,:,2); 
B3 = icon3(:,:,3);
fusedImg3 = cat(3, R3,G3,B3);

R4 = icon4(:,:,1); R4(seg4) = 0;
G4 = icon4(:,:,2); 
B4 = icon4(:,:,3);
fusedImg4 = cat(3, R4,G4,B4);

imgGroup = {icon1, gt1, seg1, fusedImg1, icon2, gt2, seg2, fusedImg2, icon3, gt3, seg3, fusedImg3, icon4, gt4, seg4, fusedImg4};

figure
montage(imgGroup, "Size",[4,4])
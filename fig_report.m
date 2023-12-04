clc; clear all; close all;
folder = "Retinal_Images\Images\";
seg_folder = "Retinal_Images\Segmentation_Combined\JPG\Seg_";
images = ["Control028_Serie2_5.jpg", "Control028_Serie2_1.jpg", "Control028_Serie2_4.jpg", "Control049_Serie2_4.jpg"];

imgGroup = {};

for nr=1:length(images)
    image = im2double(imread(folder+images(nr)));
    segImg = imbinarize(imread(seg_folder+images(nr)));

    RetCam = 1;
    dist_th = 25;
    axisRatio_th = 4.5;
    area_th = 1000;
    squareSide = 35;
    [ODMask, ck_x, ck_y] = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th, squareSide);

    maskRGB = cat(3,  segImg, segImg & ODMask, segImg & ODMask);

    if ck_x > 0
        imgDot = insertShape(image, 'filled-circle', [ck_x, ck_y, 15], Color='green');
    else
        imgDot = image;
    end
    
    R = image(:,:,1); R(ODMask) = 0;
    G = image(:,:,2); R(ODMask) = 1;
    B = image(:,:,3); R(ODMask) = 0;
    fusedImg = cat(3, R,G,B);
    
    
    imgGroup = [imgGroup, imgDot, maskRGB, ODMask, fusedImg];
end


figure
montage(imgGroup)
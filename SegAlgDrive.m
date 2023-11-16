clc; close all; clear all;
Accuracy = [];
Sensitivity = [];
Fmeasure = [];
Precision = []; 
MCC = [];
Dice = [];
Jaccard = [];
Specitivity = [];

addpath('frangi_filter_version2a\');

stage = "training";

if stage=="training"   
    imgInitial = 21;
    imgFinal = 40;
else
    imgInitial = 1;
    imgFinal = 20;
end

for i = imgInitial:imgFinal
    if i < 10
        i = "0" + i;
    end
    img_name = i + "_" + stage;
    img_name_path = "DRIVE_Dataset\"+ stage +"\images\"+ i +"_" + stage + ".tif";
    mask_name = "DRIVE_Dataset\" + stage + "\mask\"+ i +"_" + stage + "_mask.gif";
    image=im2double(imread(img_name_path));
    mask = imread(mask_name);
    mask = imbinarize(mask);
    scrtele = strel('disk', 10);
    mask = imerode(mask, scrtele);

    if stage == "training"
        GT_name = "DRIVE_Dataset\" + stage + "\1st_manual\" + i + "_manual1.tif";
        GT = imread(GT_name);
        GT = imbinarize(GT);
    end

    % LAB Image Enhancement
    LAB = rgb2lab(image);
    L = LAB(:,:,1)/100;
    L= adapthisteq(L, 'NumTiles', [16,16], 'ClipLimit',0.02);
    LAB(:,:,1) = L*100;
    adaptImgRGB = lab2rgb(LAB);
    
    % Green Channel Extraction from the Enhanced Image
    adaptImg = adaptImgRGB(:,:,2);
    
    % Background Normalization
    kernel_size = 30;
    mean_kernel = ones(kernel_size) / (kernel_size^2);
    background_estimate = conv2(adaptImg, mean_kernel, 'same');
    normalized_image = adaptImg - background_estimate;
    normalized_image(~mask) = mean(mean(normalized_image));
    
    % BSGM Filter Enhancement
    kernel = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 2 0 0 0 0 0 0 0;
    0 0 0 0 2 2 2 2 2 2 2 0 0 0 0;
    0 0 0 2 2 1 1 1 1 1 2 2 0 0 0;
    0 0 2 2 1 0 -1 -1 -1 0 1 2 2 0 0;
    0 0 2 1 0 -1 -3 -4 -3 -1 0 1 2 0 0;
    0 0 2 1 -1 -3 -6 -7 -6 -3 -1 1 2 0 0;
    0 2 2 1 -1 -4 -7 -8 -7 -4 -1 1 2 2 0;
    0 0 2 1 -1 -3 -6 -7 -6 -3 -1 1 2 0 0;
    0 0 2 1 0 -1 -3 -4 -3 -1 0 1 2 0 0;
    0 0 2 2 1 0 -1 -1 -1 0 1 2 2 0 0;
    0 0 0 2 2 1 1 1 1 1 2 2 0 0 0;
    0 0 0 0 2 2 2 2 2 2 2 0 0 0 0;
    0 0 0 0 0 0 0 2 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    ];
    gaussian_image = conv2(normalized_image, kernel, 'same'); 
    
    % Modified TopHat Operation
    Sc = strel('rectangle', [1,1]);
    closeImg = imclose(gaussian_image, Sc);
    
    fusedImg = zeros(size(closeImg));
    for j=1:3
        So = strel('disk',j);
        openImg = imopen(closeImg, So);
        tophatImg = gaussian_image - min(gaussian_image,openImg);
        fusedImg = fusedImg + (tophatImg);
    end
    fusedImg = fusedImg ./ 3;
    
    % Apply Frangi and Jerman Filters
    frangiImg = FrangiFilter2D(imcomplement(fusedImg));
 
    % Frangi Result Segmentation using Triangle Threshold
    imageArray = reshape(frangiImg, 1, []);
    tfrangi = triangleThreshold(imageArray,128);
    segImg = imbinarize(frangiImg, tfrangi);
    segImg = bwareaopen(segImg, 100) & mask;

    seg_name =  "DRIVE_Dataset\DRIVE_Segmentation_Results\JPG\Seg_" + img_name + '.jpg';
    imwrite(segImg, seg_name);
    
    matSeg_name = "DRIVE_Dataset\DRIVE_Segmentation_Results\MAT\Seg_" + img_name + '.mat';
    save(matSeg_name, "segImg");

    if stage == "training"
        [acc, sen, f, pre, mcc, dice, jac, spe] = EvaluateImageSegmentationScores(segImg, GT);
        Accuracy(end+1) = acc;
        Sensitivity(end+1) = sen;
        Fmeasure(end+1) = f;
        Precision(end+1) = pre; 
        MCC(end+1) = mcc;
        Dice(end+1) = dice;
        Jaccard(end+1) = jac;
        Specitivity(end+1) = spe;
    end

end


if stage == "training"
    table = [21:40; Accuracy; Sensitivity; Fmeasure; Precision; MCC; Dice; Jaccard; Specitivity]';
    writematrix(table, 'DRIVE_Dataset\DRIVE_Segmentation_Results\scores_drive.csv')
end

if stage == "training"
    acc = mean(Accuracy)
    sen = mean(Sensitivity)
    spe = mean(Specitivity)
end

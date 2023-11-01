% *Retinal Blood Vessel Segmentation Algorithm*
%
%     Developers: José Almeida
%     VSB - Technical University of Ostrava, 2023
%
%   This code allows to visualize each step of the pipeline to one single
%   image.

clc; clear all; close all;

% Path for the external Filter functions used
addpath('frangi_filter_version2a\');
addpath('jerman_filter\')

% Image Selection
img_folder = "ICON_Phoenix\P2\";
img_name = "210410_L_04-12-2021_06-23-31_11.jpg";
img_path = img_folder + img_name;
image = im2double(imread(img_path));
image = imresize(image,[680 680]); % ONLY for Icon Images

% FOV Mask Generation based on the Green Channel
greenImg = image(:,:,2);
mask = imbinarize(greenImg,0.01);
mask = imfill(mask, [100,100]);
scrtele = strel('disk', 10);
mask = imerode(mask, scrtele);

% LAB Image Enhancement
LAB = rgb2lab(image);
L = LAB(:,:,1)/100;
L= adapthisteq(L, 'NumTiles', [16,16], 'ClipLimit',0.02); 
LAB(:,:,1) = L*100;
adaptImgRGB = lab2rgb(LAB);

% Red Channel Extraction from the Enhanced Image
%adaptImg = adaptImgRGB(:,:,2); % For RetCam Images
adaptImg = rgb2gray(adaptImgRGB);
% adaptImg = image(:,:,2); % For Icon Images

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
Sc = strel('rectangle',[1,1]);
closeImg = imclose(gaussian_image, Sc);

fusedImg = zeros(size(closeImg));
for i=1:3
    So = strel('disk',i);
    openImg = imopen(closeImg, So);
    tophatImg = gaussian_image - min(gaussian_image,openImg);
    fusedImg = fusedImg + (tophatImg);
end
fusedImg = fusedImg ./ 3;

% Visualize the Enhancement results
figure
subplot(2,2,1)
imshow(adaptImg,[])
title('CLAHE')
subplot(2,2,2)
imshow(normalized_image, [])
title('BG Normalization')
subplot(2,2,3)
imshow(gaussian_image,[])
title('Gaussian Kernel')
subplot(2,2,4)
imshow(fusedImg,[])
title('TopHat Image')

% Apply Frangi and Jerman Filters
frangiImg = FrangiFilter2D(imcomplement(fusedImg));
jermanImg = vesselness2D(imcomplement(fusedImg), 1:2:7, [10;10] , 0.90);

% Frangi Result Segmentation using Adapted Otsu Method
tfrangi = adaptthresh(frangiImg, 0.08);
binnaryImg1 = imbinarize(frangiImg, tfrangi);
binnaryImg1 = bwareaopen(binnaryImg1, 250) & mask;

% Jerman Result Segmentation using Triangle Threshold
imageArray = reshape(jermanImg, 1, []);
tjerman = triangleThreshold(imageArray,8);
binnaryImg2 = imbinarize(jermanImg,tjerman);
binnaryImg2 = bwareaopen(binnaryImg2, 250);

% Combination of the Frangi and Jerman - For Test porpuses only, not used
jermanImgC = imadjust(jermanImg, [0 1]);
frangiImgC = frangiImg ./ max(max(frangiImg));
combinedImg = 0.3*jermanImgC + 0.7*frangiImgC;

% Final segmentation by combining the two above segmentations
segImg = binnaryImg1 & binnaryImg2;

% Results Visualization
figure
subplot(2,3,1)
imshow(frangiImg,[])
title('Frangi Filter')
subplot(2,3,2)
imshow(jermanImg, [])
title('Jerman Filter')
subplot(2,3,3)
imshow(combinedImg, [])
title('Combined Img')
subplot(2,3,4)
imshow(binnaryImg1)
title('Segmentation of Frangi Filter')
subplot(2,3,5)
imshow(binnaryImg2)
title('Segmentation of Jerman Filter')
subplot(2,3,6)
imshow(segImg)
title('Combined Segmentation')

seg_img = greenImg + 255*segImg;

figure
subplot(1,2,1)
imshow(image)
title('Original Image')
subplot(1,2,2)
imshow(seg_img)
title('Segmentation Result')

% segImg = imresize(segImg, [1240 1240]);
% figure; imshow(segImg)
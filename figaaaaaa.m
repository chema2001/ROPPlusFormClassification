clc; clear all; close all;
% Path for the external Filter functions used
addpath('frangi_filter_version2a\');
addpath('jerman_filter\')

fontSize = 28;

img_folder = "Retinal_Images\Images\";
img_name = "Control028_Serie1_2.jpg";
img_path = img_folder + img_name;
image = im2double(imread(img_path));

greenImg = image(:,:,2);
mask = imbinarize(greenImg,0.01);
mask = imfill(mask, [100,100]);
scrtele = strel('disk', 10);
mask = imerode(mask, scrtele);
mask(1:5, :) = 0;
mask(end-5:end, :) = 0;

figure;
subplot(1,2,1)
imshow(image)
title('RetCam3 Image', 'FontSize',35)
subplot(1,2,2)
imshow(mask)
title('Generated FOV mask','FontSize',35)

% figure;
% subplot(1,3,1)
% imshow(image)
% title('DRIVE Image Example', 'FontSize', fontSize)
% subplot(1,3,2)
% imshow(mask)
% title('Provided FOV Mask', 'FontSize', fontSize)
% subplot(1,3,3)
% imshow(GT)
% title('Provided Manual Annotation (GT)', 'FontSize', fontSize)

LAB = rgb2lab(image);
L = LAB(:,:,1)/100;
L= adapthisteq(L, 'NumTiles', [16,16], 'ClipLimit',0.02);
LAB(:,:,1) = L*100;
adaptImgRGB = lab2rgb(LAB);

figure;
subplot(1,2,1)
imshow(image)
title('Original Image', 'FontSize', 35)
subplot(1,2,2)
imshow(adaptImgRGB)
title('CIELAB Enhancement Result', 'FontSize', 35)

% figure; 
% subplot(1,3,1)
% imshow(adaptImgRGB(:,:,1))
% title('Red Channel', 'FontSize', fontSize)
% subplot(1,3,2)
% imshow(adaptImgRGB(:,:,2))
% title('Green Channel', 'FontSize', fontSize)
% subplot(1,3,3)
% imshow(adaptImgRGB(:,:,3))
% title('Blue Channel', 'FontSize', fontSize)


adaptImg = adaptImgRGB(:,:,2);

% Background Normalization
kernel_size = 30;
mean_kernel = ones(kernel_size) / (kernel_size^2);
background_estimate = conv2(adaptImg, mean_kernel, 'same');
normalized_image = adaptImg - background_estimate;
normalized_image(~mask) = mean(mean(normalized_image));

figure;
subplot(1,3,1)
imshow(adaptImg)
title('Green Channel', 'FontSize', fontSize)
subplot(1,3,2)
imshow(background_estimate, [])
title('Background Estimation', 'FontSize', fontSize)
subplot(1,3,3)
imshow(normalized_image, [])
title('Normalized Image', 'FontSize', fontSize)

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
Sc = strel('rectangle',[2,2]);
closeImg = imclose(gaussian_image, Sc);

fusedImg = zeros(size(closeImg));
for j=1:3
    So = strel('disk',j);
    openImg = imopen(closeImg, So);
    tophatImg = gaussian_image - min(gaussian_image,openImg);
    fusedImg = fusedImg + (tophatImg);
end
fusedImg = fusedImg ./ 3;

figure;
subplot(1,3,1)
imshow(normalized_image, [])
title('Normalized Image', 'FontSize', fontSize)
subplot(1,3,2)
imshow(gaussian_image, [])
title('BSGMF Result', 'FontSize', fontSize)
subplot(1,3,3)
imshow(fusedImg,[])
title('Modified Top-Hat Result', 'FontSize', fontSize)

% Apply Frangi and Jerman Filters
frangiImg = FrangiFilter2D(imcomplement(fusedImg));
jermanImg = vesselness2D(imcomplement(fusedImg), 1:2:7, [10;10], 0.9);
%frangiImg = imadjust(frangiImg);
%jermanImg = imadjust(jermanImg);

figure;
subplot(1,3,1)
imshow(fusedImg ,[])
title('Modified Top-Hat Result', 'FontSize', fontSize)
subplot(1,3,2)
imshow(frangiImg, [])
title('Frangi Filter Result', 'FontSize',fontSize)
subplot(1,3,3)
imshow(jermanImg)
title('Jerman Filter Result', 'FontSize',fontSize)

frangiImg = imadjust(frangiImg);
jermanImg = imadjust(jermanImg);
combinedImg = 0.7*frangiImg + 0.3*jermanImg;
imageArray = reshape(combinedImg, 1, []);
tcombined = triangleThreshold(imageArray, 4);
binnaryImg3 = imbinarize(combinedImg, tcombined);
binnaryImg3 = bwareaopen(binnaryImg3, 250) & mask;
% 
% figure;
% subplot(1,3,1)
% imshow(image)
% title('Original Image', 'FontSize', fontSize)
% subplot(1,3,2)
% imshow(segImg)
% title('Obtained Segmented Image', 'FontSize', fontSize)
% subplot(1,3,3)
% imshow(GT)
% title('Ground Thruth','FontSize', fontSize)

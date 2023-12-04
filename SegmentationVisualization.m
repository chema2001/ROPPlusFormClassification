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
img_folder = "Retinal_Images\Images\";
img_name = "Control028_Serie2_5.jpg";
img_path = img_folder + img_name;
image = im2double(imread(img_path));
%image = imresize(image,[680 680]); % ONLY for Icon Images

% FOV Mask Generation based on the Green Channel
greenImg = image(:,:,2);
mask = imbinarize(greenImg,0.01);
mask = imfill(mask, [100,100]);
scrtele = strel('disk', 10);
mask = imerode(mask, scrtele);
mask(1:5, :) = 0;
mask(end-5:end, :) = 0;

% LAB Image Enhancement
LAB = rgb2lab(image);
L = LAB(:,:,1)/100;
L= adapthisteq(L, 'NumTiles', [16,16], 'ClipLimit',0.02); 
LAB(:,:,1) = L*100;
adaptImgRGB = lab2rgb(LAB);

% Red Channel Extraction from the Enhanced Image
adaptImg = adaptImgRGB(:,:,2); 
% adaptImg = image(:,:,2); 

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
Sc = strel('rectangle',[2,2]);
closeImg = imclose(gaussian_image, Sc);

fusedImg = zeros(size(closeImg));
for i=1:3
    So = strel('disk',i);
    openImg = imopen(closeImg, So);
    tophatImg = gaussian_image - min(gaussian_image,openImg);
    fusedImg = fusedImg + (tophatImg);
end
fusedImg = fusedImg ./ 3;

% Apply Frangi and Jerman Filters
frangiImg = FrangiFilter2D(imcomplement(fusedImg));
jermanImg = vesselness2D(imcomplement(fusedImg), 1:2:7, [10;10], 0.90);

% Frangi Result Segmentation using Adapted Otsu Method
imageArray = reshape(frangiImg, 1, []);
tfrangi = triangleThreshold(imageArray,32);
binnaryImg1 = imbinarize(frangiImg, tfrangi);
binnaryImg1 = bwareaopen(binnaryImg1, 250) & mask;

% Jerman Result Segmentation using Triangle Threshold
imageArray = reshape(jermanImg, 1, []);
tjerman = triangleThreshold(imageArray,4);
binnaryImg2 = imbinarize(jermanImg,tjerman);
binnaryImg2 = bwareaopen(binnaryImg2, 300);

frangiImg = imadjust(frangiImg);
jermanImg = imadjust(jermanImg);
combinedImg = 0.7*frangiImg + 0.3*jermanImg;
imageArray = reshape(combinedImg, 1, []);
tcombined = triangleThreshold(imageArray, 4);
binnaryImg3 = imbinarize(combinedImg, tcombined);
binnaryImg3 = bwareaopen(binnaryImg3, 200) & mask;

% Final segmentation by combining the two above segmentations
segImg = binnaryImg3;


% segImg = imresize(segImg, [1240 1240]);
figure; imshowpair(frangiImg, segImg, 'montage')


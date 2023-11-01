clc; clear all; close all;

% Path for the external Filter functions used
addpath('frangi_filter_version2a\');
addpath('jerman_filter\')

% Folder Selection
img_folder = "Retinal_Images\Images\";
seg_folder = "Retinal_Images\Segmentation_Results\JPG\";
files = dir(img_folder);
image_files = files(contains({files.name}, {'.jpg', '.png', '.bmp', '.tif'}));

metric_th = 0.3;
axisRatio_th = 3.5;
area_th = 650;

for i=1:length(image_files)
    img_name = img_folder + image_files(i).name;
    image = im2double(imread(img_name));
    seg_name = seg_folder + "Seg_" + image_files(i).name;
    segImg = imread(seg_name);
    
    segImg = imbinarize(segImg);
    [BW,~] = createYellowMask(image);

    %parametry pro BF
    sigmas = 5;        %  spatial gaussian kernel 
    sigmar100 = 100;    %  range gaussian kernel for BF
    w = 6*round(sigmas)+1;
    tol = 0.01;
    
    [r, s, ~]=size(image);
    
    model_RG(:,:,1)= image (:,:,1); %Green and red channel extraction
    model_RG(:,:,2)= image (:,:,2);
    model_RG(:,:,3)= zeros (r,s);
    
    model_RG=rgb2gray(model_RG); % Monochromatic conversion
    
    hist=adapthisteq(model_RG); %Histogram equalization
    
    Img=imadjust(hist,[0.1; 0.8],[0; 0.9]); %Intensity transformation
    
    f=double(Img);
    
    L = 0;
    [M, N]=computeTruncation(f, sigmar100, w, tol);
    [f100, ~] = computeDivergence(f, f, sigmas,sigmar100,L,w,N,M);
    Img = double(f100);
    
    Img = double(Img(:,:,1)); %format double
    
    %Definition of center of initial contour
    greenImg = image(:,:,2);
    mask = imbinarize(greenImg,0.01);
    mask = imfill(mask, [100,100]);
    mask(1:10, :) = 0;
    mask(end-10:end, :) = 0;
    scrtele = strel('disk', 50);
    mask = imerode(mask, scrtele) & BW;
    
    redChannel = im2double(image(:,:,1));
%     kernel_size = 30;
%     mean_kernel = ones(kernel_size) / (kernel_size^2);
%     background_estimate = conv2(redChannel, mean_kernel, 'same');
%     normalized_image = redChannel - background_estimate;
%     normalized_image(~mask) = mean(mean(normalized_image));
    
    gSize = 70;
    gaussian1 = fspecial('Gaussian', gSize, 15);
    gaussian2 = fspecial('Gaussian', gSize, 17);
    dog = gaussian1 - gaussian2;
    filtImg = conv2(redChannel, dog, 'same');
    filtImg(~mask) = mean(mean(filtImg(mask)));
    filtImg = imadjust(filtImg);
    
    [c,r,metric] = imfindcircles(filtImg,[10 40], 'EdgeThreshold',0.25);
    
    [xLength, yLength] = size(filtImg);
    squareSide = 50;
    windowMask = segImg;
    radii_th = 25;
    target = [0,0];
    idx = -1;
    
    if ~isempty(c) 
        if metric(1) > metric_th
            c = c(1,:);

            if c(2) > squareSide
                windowMask(1:(c(2)-squareSide),:) = 0;
            end
            if c(2) < xLength - squareSide
                windowMask((c(2)+squareSide):end,:) = 0;
            end
            
            if c(1) > squareSide
                windowMask(:,1:(c(1)-squareSide)) = 0;
            end
            if c(1) < yLength - squareSide
                windowMask(:,(c(1)+squareSide):end) = 0;
            end

            L = bwlabel(windowMask);
            stats = regionprops(L,"Centroid", "MajorAxisLength", "MinorAxisLength", "Area");
            centroids = cat(1, stats.Centroid);
            majorAxis = cat(1, stats.MajorAxisLength);
            minorAxis = cat(1, stats.MinorAxisLength);
            areaStat = cat(1, stats.Area);
            
            for j=1:size(centroids,1)
                distx = centroids(j,1) - c(1);
                disty = centroids(j,2) - c(2);
                dist = sqrt(distx^2 + disty^2);
                axisRatio = majorAxis(j)/minorAxis(j);
                areaObj = areaStat(j);
                
                if dist < radii_th && axisRatio < axisRatio_th && areaObj < area_th
                    radii_th = dist;
                    idx = j;
                end
            end
 
         end
    end

    ODMask = segImg;
    if idx > -1   
        ODMask(L==idx) = 0;
    end

    maskRGB = cat(3, 255 * segImg, 255*(segImg & ODMask), 255*(segImg & ODMask));
    ODMaskName = "Retinal_Images\ODMasks_new\Visual\OD_" + image_files(i).name;
    imwrite(maskRGB, ODMaskName);

    ODFilteredImageName = "Retinal_Images\ODMasks_new\Visual\OD_" + image_files(i).name; 
    imwrite(ODMask, ODFilteredImageName);

    image_id = split(image_files(i).name, '.');
    matOD_name = "Retinal_Images\ODMasks_new\MAT\OD_" + image_id(1) + ".mat";
    save(matOD_name, "ODMask");
end
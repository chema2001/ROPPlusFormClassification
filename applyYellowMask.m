clc; clear all; close all;

%Folder Selection

% img_folder = "Retinal_Images\Images\";
% seg_folder = "Retinal_Images\Segmentation_Results\JPG\";
% files = dir(img_folder);
% image_files = files(contains({files.name}, {'.jpg', '.png', '.bmp', '.tif'}));
% 
% for i=1:length(image_files)
%     disp("Image " + i + "/" + length(image_files))
%     img_name = img_folder + image_files(i).name;
%     image = imread(img_name);
%     seg_name = seg_folder + "Seg_" + image_files(i).name;
%     segImg = imread(seg_name);
%     segImg = imbinarize(segImg);
% 
%     ymMask = createYellowMask_new(image);
%     
%     newMask = ~ymMask & segImg;
% 
%     maskRGB = cat(3, 255 * segImg, 255*(segImg & newMask), 255*(segImg & newMask));
%     YMMaskName = "Retinal_Images\TestYellowMask\YM_" + image_files(i).name;
%     imwrite(maskRGB, YMMaskName);
% end
 
img = imread("Retinal_Images\Images\ROP040_Serie11_1.jpg");
segImg = imbinarize(imread("Retinal_Images\Segmentation_Results\JPG\Seg_ROP040_Serie12_9.jpg"));

LAB = rgb2lab(img);
L = LAB(:,:,2)/100;
L= adapthisteq(L);
LAB(:,:,2) = L*100;
adaptImgRGB = lab2rgb(LAB);
figure; imshow(adaptImgRGB)


% [BW, rgbmask] = createYellowMask_new(img);
% 
% newMask = ~BW & segImg;
% maskRGB = cat(3, 255 * segImg, 255*(segImg & newMask), 255*(segImg & newMask));
% imshowpair(segImg, maskRGB, 'montage')
% 
% figure; imshow(rgbmask)
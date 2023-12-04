    
    image = im2double(imread("DRIVE_Dataset\training\images\40_training.tif"));
    mask_name = "DRIVE_Dataset\training\mask\40_training_mask.gif";
    mask = imread(mask_name);
    mask = imbinarize(mask);
    scrtele = strel('disk', 12);
    mask = imerode(mask, scrtele);

    GT_name = "DRIVE_Dataset\training\1st_manual\40_manual1.tif";
    GT = imread(GT_name);
    GT = imbinarize(GT);

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
    Sc = strel('rectangle', [2,2]);
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
    tfrangi = triangleThreshold(imageArray,100);
    %tfrangi = adaptthresh(frangiImg, 0.08);
    binnaryImg1 = imbinarize(frangiImg, tfrangi);
    binnaryImg1 = bwareaopen(binnaryImg1, 100) & mask;


    jermanImg = vesselness2D(imcomplement(fusedImg), 1:2:7, [10;10], 0.90);
    imageArray = reshape(jermanImg, 1, []);
    tjerman = triangleThreshold(imageArray,6);
    binnaryImg2 = imbinarize(jermanImg,tjerman);
    binnaryImg2 = bwareaopen(binnaryImg2, 100);
    
    frangiImg = imadjust(frangiImg);
    jermanImg = imadjust(jermanImg);
    combinedImg = 0.6*frangiImg + 0.4*jermanImg;
    imageArray = reshape(combinedImg, 1, []);
    tcombined = triangleThreshold(imageArray, 8);
    binnaryImg3 = imbinarize(combinedImg, tcombined);
    binnaryImg3 = bwareaopen(binnaryImg3, 100) & mask;

    imGroup= {GT, binnaryImg1, binnaryImg2, binnaryImg3};
    figure
    montage(imGroup, 'size', [1 4])


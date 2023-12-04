function segImg = segmentRetinalImage(image, filter, bins, cielabenh)
    
    [L,W] = size(image(:,:,1));
    rsz = 0;

    if max(L,W) > 700
        rsz = 1;
        ratio = L/W;
        image = imresize(image, [680 680*ratio]);
    end

    greenImg = image(:,:,2);
    mask = imbinarize(greenImg,0.01);
    mask = imfill(mask, [100,100]);
    scrtele = strel('disk', 10);
    mask = imerode(mask, scrtele);
    mask(1:5, :) = 0;
    mask(end-5:end, :) = 0;
    

    if cielabenh
        % LAB Image Enhancement
        LAB = rgb2lab(image);
        L = LAB(:,:,1)/100;
        L= adapthisteq(L, 'NumTiles', [16,16], 'ClipLimit',0.02);
        LAB(:,:,1) = L*100;
        adaptImgRGB = lab2rgb(LAB);
    
        % Red Channel Extraction from the Enhanced Imageg
        adaptImg = adaptImgRGB(:,:,2);
    else
        adaptImg = image(:,:,2);
    end

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
    for j=1:3
        So = strel('disk',j);
        openImg = imopen(closeImg, So);
        tophatImg = gaussian_image - min(gaussian_image,openImg);
        fusedImg = fusedImg + (tophatImg);
    end
    fusedImg = fusedImg ./ 3;

    % Apply Frangi and Jerman Filters
    frangiImg = FrangiFilter2D(imcomplement(fusedImg));
    jermanImg = vesselness2D(imcomplement(fusedImg), 1:2:7, [10;10], 0.90);
    
    bins = str2num(bins);
    if strcmp(filter,'frangi')
        % Frangi Result Segmentation using Adapted Otsu Method
        imageArray = reshape(frangiImg, 1, []);
        tfrangi = triangleThreshold(imageArray,bins);
        segImg = imbinarize(frangiImg, tfrangi);
        segImg = bwareaopen(segImg, 200) & mask;
    elseif strcmp(filter,'jerman')
        % Jerman Result Segmentation using Triangle Threshold
        imageArray = reshape(jermanImg, 1, []);
        tjerman = triangleThreshold(imageArray,bins);
        segImg = imbinarize(jermanImg,tjerman);
        segImg = bwareaopen(segImg, 200);
    elseif strcmp(filter,'combined')
        % Combined Enhancement
        frangiImg = imadjust(frangiImg);
        jermanImg = imadjust(jermanImg);
        combinedImg = 0.7*frangiImg + 0.3*jermanImg;
        imageArray = reshape(combinedImg, 1, []);
        tcombined = triangleThreshold(imageArray, bins);
        segImg = imbinarize(combinedImg, tcombined);
        segImg = bwareaopen(segImg, 200) & mask;
    end
    
    if rsz
        segImg = imresize(segImg, [L W]);
    end

end
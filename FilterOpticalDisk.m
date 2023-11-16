function newMask = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th)

    if ~RetCam
        image = imresize(image, [640 640]);
        segImg = imresize(segImg, [640 640]);
    end

    
    %Definition of center of initial contour
    greenImg = image(:,:,2);
    mask = imbinarize(greenImg,0.01);
    mask = imfill(mask, [100,100]);
    mask(1:10, :) = 0;
    mask(end-10:end, :) = 0;
    scrtele = strel('disk', 100);
    mask = imerode(mask, scrtele);

    if ~RetCam
        greenChannel = im2double(image(:,:,2));

        kernel_size = 30;
        mean_kernel = ones(kernel_size) / (kernel_size^2);
        background_estimate = conv2(greenChannel, mean_kernel, 'same');
        normalized_image = greenChannel - background_estimate;
        normalized_image(~mask) = mean(mean(normalized_image));
    
        DoGInput = normalized_image;
    else
        redChannel = im2double(image(:,:,1));

        DoGInput = redChannel;
    end

    gSize = 70;
    gaussian1 = fspecial('Gaussian', gSize, 15);
    gaussian2 = fspecial('Gaussian', gSize, 17);
    dog = gaussian1 - gaussian2;
    filtImg = conv2(DoGInput, dog, 'same');
    filtImg(~mask) = mean(mean(filtImg(mask)));
    filtImg = imadjust(filtImg);
   
  
    [c,~,metric] = imfindcircles(filtImg,[10 40], 'EdgeThreshold',0.25);

    [xLength, yLength] = size(filtImg);
    
    metric_th = 0.25;
    squareSide = 50;
    min_area = 50;

    idx = -1;
    foundObject = 0;

    if ~isempty(c)      
        for k=1:length(metric)
            if metric(k) > metric_th && ~foundObject
                windowMask = segImg;
                se = strel('rectangle', [2 2]);
                windowMask = imdilate(windowMask, se);
                %figure; imshow(filtImg); hold on; plot(c(:,1), c(:,2), 'b*'); hold off
                %figure; imshow(segImg); hold on; plot(c(center,1), c(center,2), 'r*'); hold off
                
                ck = c(k,:);
                if ck(2) > squareSide
                    windowMask(1:(ck(2)-squareSide),:) = 0;
                end
                if ck(2) < xLength - squareSide
                    windowMask((ck(2)+squareSide):end,:) = 0;
                end
                
                if ck(1) > squareSide
                    windowMask(:,1:(ck(1)-squareSide)) = 0;
                end
                if ck(1) < yLength - squareSide
                    windowMask(:,(ck(1)+squareSide):end) = 0;
                end
                
                L = bwlabel(windowMask);
                stats = regionprops(L,"Centroid", "MajorAxisLength", "MinorAxisLength", "Area");
                centroids = cat(1, stats.Centroid);
                majorAxis = cat(1, stats.MajorAxisLength);
                minorAxis = cat(1, stats.MinorAxisLength);
                areaStat = cat(1, stats.Area);
                
                for i=1:size(centroids,1)
                    distx = centroids(i,1) - ck(1);
                    disty = centroids(i,2) - ck(2);
                    dist = sqrt(distx^2 + disty^2);
                    axisRatio = majorAxis(i)/minorAxis(i);
                    areaObj = areaStat(i);
            
                    if dist < dist_th && axisRatio < axisRatio_th && areaObj < area_th && areaObj > min_area
                        dist_th = dist;
                        idx = i;
                        foundObject = 1;
                    end
                end
                clear stats;
            end
        end
    end
    
    newMask = segImg;
    if idx > -1   
        newMask(L==idx) = 0;
    end

    if ~RetCam
        newMask = imresize(newMask, [1240 1240]);
    end
end
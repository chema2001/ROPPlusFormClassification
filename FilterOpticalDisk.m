function [newMask,ck_x,ck_y] = FilterOpticalDisk(image, segImg, RetCam, dist_th, axisRatio_th, area_th, squareSide)

    if ~RetCam
        image = imresize(image, [640 640]);
        segImg = imresize(segImg, [640 640]);
    end

    
    %Definition of center of initial contour
    greenImg = image(:,:,2);
    mask = imbinarize(greenImg, 0.01);
    mask = imfill(mask, [100,100]);
    mask(1:10, :) = 0;
    mask(end-10:end, :) = 0;
    scrtele = strel('disk', 35);
    mask = imerode(mask, scrtele);

    if ~RetCam
        greenChannel = im2double(image(:,:,2));
        
        DoGInput = greenChannel;
    else
        redChannel = im2double(image(:,:,1));

        DoGInput = redChannel;
    end

    gSize = 70;
    gaussian1 = fspecial('Gaussian', gSize, 13);
    gaussian2 = fspecial('Gaussian', gSize, 17);
    dog = gaussian1 - gaussian2;
    filtImg = conv2(DoGInput, dog, 'same');
    filtImg(~mask) = mean(mean(filtImg(mask)));
    filtImg = imadjust(filtImg);

    [c,~,metric] = imfindcircles(filtImg,[10 100], 'EdgeThreshold',0.2);

    [xLength, yLength] = size(filtImg);
    
    metric_th = 0.15;
    min_area = 300;

    idx = -1;
    foundObject = 0;

    ck_x = -1;
    ck_y = -1;
    
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
        ck_x = ck(1);
        ck_y = ck(2);
    elseif ~isempty(c)
        ck_x = c(1,1);
        ck_y = c(1,2);
    else
        ck_x = -1;
        ck_y = -1;
    end

    if ~RetCam
        newMask = imresize(newMask, [1240 1240]);
    end
end
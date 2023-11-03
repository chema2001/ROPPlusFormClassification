function [mask, center, radii] = OpticalDiskMask(i, RetCam, metric_th)

    %parametry pro BF
    sigmas = 5;        %  spatial gaussian kernel 
    sigmar100 = 100;    %  range gaussian kernel for BF
    w = 6*round(sigmas)+1;
    tol = 0.01;

    [r, s, ~]=size(i);
    
    model_RG(:,:,1)= i (:,:,1); %Green and red channel extraction
    model_RG(:,:,2)= i (:,:,2);
    model_RG(:,:,3)= zeros (r,s);
    
    model_RG=rgb2gray(model_RG); % Monochromatic conversion
    %model_RG(~segImg) = mean(model_RG(:));
    
    hist=adapthisteq(model_RG); %Histogram equalization
    
    Img=imadjust(hist,[0.1; 0.8],[0; 0.9]); %Intensity transformation
    
    f=double(Img);

    L = 0;
    [M, N]=computeTruncation(f, sigmar100, w, tol);
    [f100, ~] = computeDivergence(f, f, sigmas,sigmar100,L,w,N,M);
    Img = double(f100);
    
    Img = double(Img(:,:,1)); %format double
    
    %Definition of center of initial contour
    greenImg = i(:,:,2);
    mask = imbinarize(greenImg,0.01);
    mask = imfill(mask, [100,100]);
    mask(1:10, :) = 0;
    mask(end-10:end, :) = 0;
    scrtele = strel('disk', 50);
    mask = imerode(mask, scrtele);

    if ~RetCam
        greenChannel = im2double(i(:,:,2));

        kernel_size = 30;
        mean_kernel = ones(kernel_size) / (kernel_size^2);
        background_estimate = conv2(greenChannel, mean_kernel, 'same');
        normalized_image = greenChannel - background_estimate;
        normalized_image(~mask) = mean(mean(normalized_image));
    
        DoGInput = normalized_image;
    else
        [BW,~] = createYellowMask(i);
        mask = mask & BW;

        redChannel = im2double(i(:,:,1));

        DoGInput = redChannel;
    end

    gSize = 70;
    gaussian1 = fspecial('Gaussian', gSize, 15);
    gaussian2 = fspecial('Gaussian', gSize, 17);
    dog = gaussian1 - gaussian2;
    filtImg = conv2(DoGInput, dog, 'same');
    filtImg(~mask) = mean(mean(filtImg(mask)));
    filtImg = imadjust(filtImg);

    [centers,radiis,metric] = imfindcircles(filtImg,[10 40], 'EdgeThreshold',0.25);
    mask = zeros(size(redChannel));
    center = [];
    radii =[];
    if ~isempty(centers) 
        if metric(1) > metric_th
            xt = centers(1,1);
            yt = centers(1,2);
            
            a=2.5; % Size of horizontal axis of initial active contour in the shape of ellipse
            b=5.5;  % Size of vertical axis of initial active contour in the shape of ellipse
            NumIter = 300; %iterations
            timestep=.05; %time step
            mu=.05/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
            sigma = 10;%size of kernel
            epsilon = .01;
            c0 = 2.6; % the constant value 
            lambda1=1.01;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
            lambda2=1.05;%inner weight
            %if lambda1>lambda2; tend to inflate
            %if lambda1<lambda2; tend to deflate
            nu = 0.0005*255*255;%length term
            alf = 25;%data term weight
            
            %Define segmentation based on the active contour model
            [Height, Wide] = size(Img);
            [xx, yy] = meshgrid(1:Wide,1:Height);
            phi = (sqrt(((xx - xt).^2/a + (yy - yt).^2/b )) - 10); % initial active contour defintion
            phi = sign(phi).*c0;
            
            Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
            ONE=ones(size(Img));
            KONE = imfilter(ONE,Ksigma,'replicate');  
            KI = imfilter(Img,Ksigma,'replicate');  
            KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
            %pause(0.02)
            
            % Compute segmentation over set iterations
            for iter = 1:NumIter
                phi =evolution_LGD(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
            end
            
            [r, s]=size(phi);
            
            for i=1:r
                for j=1:s
                    if phi(i,j)<0
                        bin(i,j)=0;
                    else
                        bin(i,j)=1;
                    end
                end
            end
            
            mask=1-bin;
            center = centers(1,:);
            radii = radiis(1);
        end 
    end
end


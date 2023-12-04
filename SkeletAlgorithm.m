clc; clear all; close all;

% Folder Selection
seg_folder = "ICON_Phoenix\P4\Segmentation_Results\MAT\";
files = dir(seg_folder);
seg_files = files(contains({files.name}, {'.mat'}));

for i=1:length(seg_files)
    segResult = load(seg_folder + seg_files(i).name);
    segResult = segResult.segImg; %Change accordingly
    skel = bwskel(segResult);
    
    % Save the result of the Segmentation
    seg_name = split(seg_files(i).name, '.');
    skel_name =  "ICON_Phoenix\P4\Skeletonization_Results\JPG\Skel_" + seg_name(1) + ".jpg";
    imwrite(skel, skel_name);

    matSkel_name = "ICON_Phoenix\P4\Skeletonization_Results\MAT\Skel_" + seg_name(1) + ".mat";
    save(matSkel_name, "skel");
end
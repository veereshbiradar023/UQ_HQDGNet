function [normalized_vol, lung_mask] = preprocess_ct_volume(raw_volume, hu_min, hu_max)
% PREPROCESS_CT_VOLUME Preprocesses thoracic CT volume.
%   Applies Hounsfield Unit (HU) clamping, intensity normalization into [0, 1],
%   and morphological lung parenchyma segmentation.
%
%   Syntax:
%       [normalized_vol, lung_mask] = preprocess_ct_volume(raw_volume, hu_min, hu_max)
%
%   Inputs:
%       raw_volume - 3D matrix of CT attenuation values in HU
%       hu_min     - Minimum window threshold (default: -1000 HU)
%       hu_max     - Maximum window threshold (default: +400 HU)
%
%   Outputs:
%       normalized_vol - Normalized 3D volume in range [0, 1]
%       lung_mask      - Binary 3D mask of segmented lung parenchyma

    if nargin < 2, hu_min = -1000.0; end
    if nargin < 3, hu_max = 400.0; end

    % 1. Hounsfield Unit Clamping and Linear Normalization
    clamped = max(min(raw_volume, hu_max), hu_min);
    normalized_vol = (clamped - hu_min) / (hu_max - hu_min);

    % 2. Morphological Lung Segmentation
    % Lung parenchyma exhibits attenuation < -320 HU
    thresh_val = (-320.0 - hu_min) / (hu_max - hu_min);
    binary_air = normalized_vol < thresh_val;

    % Perform 3D morphological opening and closing to remove airways & noise
    se = strel('sphere', 2);
    lung_mask = imopen(binary_air, se);
    lung_mask = imclose(lung_mask, se);

    % Retain two largest 3D connected components (left and right lungs)
    cc = bwconncomp(lung_mask);
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [~, idx] = sort(numPixels, 'descend');
    
    clean_mask = false(size(lung_mask));
    n_keep = min(2, length(idx));
    for i = 1:n_keep
        clean_mask(cc.PixelIdxList{idx(i)}) = true;
    end
    lung_mask = clean_mask;
end

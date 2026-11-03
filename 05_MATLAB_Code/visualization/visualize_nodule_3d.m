function visualize_nodule_3d(patch, center_coords, save_path)
% VISUALIZE_NODULE_3D Renders Multi-Planar Reconstruction (MPR) and 3D isosurface.
%
%   Inputs:
%       patch         - 3D matrix representing local CT subvolume (D x H x W)
%       center_coords - [z, y, x] center of nodule candidate
%       save_path     - Output image path (optional)

    if nargin < 2 || isempty(center_coords)
        center_coords = round(size(patch) / 2);
    end

    cz = center_coords(1);
    cy = center_coords(2);
    cx = center_coords(3);

    fig = figure('Position', [100, 100, 1000, 800], 'Color', 'w');

    % 1. Axial slice
    subplot(2, 2, 1);
    imshow(squeeze(patch(cz, :, :)), []);
    colormap(gca, 'gray');
    title(sprintf('Axial Slice (Z = %d)', cz), 'FontSize', 12, 'FontWeight', 'bold');
    hold on; plot(cx, cy, 'r+', 'MarkerSize', 12, 'LineWidth', 2);

    % 2. Coronal slice
    subplot(2, 2, 2);
    imshow(squeeze(patch(:, cy, :)), []);
    colormap(gca, 'gray');
    title(sprintf('Coronal Slice (Y = %d)', cy), 'FontSize', 12, 'FontWeight', 'bold');
    hold on; plot(cx, cz, 'r+', 'MarkerSize', 12, 'LineWidth', 2);

    % 3. Sagittal slice
    subplot(2, 2, 3);
    imshow(squeeze(patch(:, :, cx)), []);
    colormap(gca, 'gray');
    title(sprintf('Sagittal Slice (X = %d)', cx), 'FontSize', 12, 'FontWeight', 'bold');
    hold on; plot(cy, cz, 'r+', 'MarkerSize', 12, 'LineWidth', 2);

    % 4. 3D Isosurface rendering
    subplot(2, 2, 4);
    iso_thresh = 0.55; % Soft-tissue nodule threshold in [0, 1]
    p = patch(isosurface(patch, iso_thresh));
    isonormals(patch, p);
    set(p, 'FaceColor', [0.85, 0.3, 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.85);
    camlight; lighting phong; view(3); axis tight equal;
    grid on;
    title('3D Nodule Morphology Isosurface', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X (voxels)'); ylabel('Y (voxels)'); zlabel('Z (voxels)');

    sgtitle('UQ-HQDGNet 3D Pulmonary Nodule Multi-Planar Characterization', 'FontSize', 14, 'FontWeight', 'bold');

    if nargin >= 3 && ~isempty(save_path)
        exportgraphics(fig, save_path, 'Resolution', 300);
        fprintf('Saved visualization to %s\n', save_path);
    end
end

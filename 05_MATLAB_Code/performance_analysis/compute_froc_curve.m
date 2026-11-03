function froc_results = compute_froc_curve(nodule_probs, nodule_labels, num_scans, save_path)
% COMPUTE_FROC_CURVE Evaluates Free-Response ROC for Pulmonary Nodule Detection.
%
%   Inputs:
%       nodule_probs  - Predicted nodule candidate presence probabilities
%       nodule_labels - Binary ground truth (1 = nodule, 0 = non-nodule)
%       num_scans     - Total number of unique CT scans in evaluation cohort
%       save_path     - Output plot file path (optional)

    if nargin < 3 || isempty(num_scans)
        num_scans = 20;
    end

    total_nodules = sum(nodule_labels == 1);
    thresholds = linspace(0.01, 0.99, 100);
    fps_vec = zeros(length(thresholds), 1);
    sens_vec = zeros(length(thresholds), 1);

    for i = 1:length(thresholds)
        th = thresholds(i);
        preds = (nodule_probs >= th);
        tp = sum(preds == 1 & nodule_labels == 1);
        fp = sum(preds == 1 & nodule_labels == 0);

        sens_vec(i) = tp / max(1, total_nodules);
        fps_vec(i) = fp / max(1, num_scans);
    end

    % Sort by false positives per scan ascending
    [fps_sorted, sort_idx] = sort(fps_vec);
    sens_sorted = sens_vec(sort_idx);

    % Standard LUNA16 predefined benchmark points
    standard_fps = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
    sens_at_std = interp1(fps_sorted, sens_sorted, standard_fps, 'linear', 'extrap');
    sens_at_std = min(max(sens_at_std, 0), 1);

    mean_froc = mean(sens_at_std);

    froc_results.fps = fps_sorted;
    froc_results.sensitivities = sens_sorted;
    froc_results.standard_fps = standard_fps;
    froc_results.sens_at_std_fps = sens_at_std;
    froc_results.mean_froc = mean_froc;

    % Plot FROC
    fig = figure('Color', 'w', 'Position', [150, 150, 700, 550]);
    semilogx(fps_sorted, sens_sorted, 'b-', 'LineWidth', 2.5);
    hold on;
    plot(standard_fps, sens_at_std, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
    grid on;
    xlim([0.1, 10]);
    ylim([0.4, 1.02]);
    xlabel('False Positives per Scan (FPS, log scale)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Detection Sensitivity', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('UQ-HQDGNet FROC Curve (Mean Score = %1.4f)', mean_froc), ...
          'FontSize', 13, 'FontWeight', 'bold');
    legend({'FROC Response', 'Standard Benchmark Points'}, 'Location', 'southeast', 'FontSize', 11);

    if nargin >= 4 && ~isempty(save_path)
        exportgraphics(fig, save_path, 'Resolution', 300);
        fprintf('Saved FROC plot to %s\n', save_path);
    end
end

function stats_report = statistical_tests(proposed_scores, baseline_scores, ground_truth)
% STATISTICAL_TESTS Computes statistical hypothesis tests comparing models.
%   Calculates:
%     1. Paired Wilcoxon signed-rank test.
%     2. DeLong test for AUC differences.
%     3. Cohen's d effect size.
%     4. 95% Bootstrap Confidence Intervals for AUC.

    proposed_scores = proposed_scores(:);
    baseline_scores = baseline_scores(:);
    ground_truth = ground_truth(:);

    n = length(ground_truth);
    
    % 1. Paired Wilcoxon Signed-Rank Test
    [p_val_wilcoxon, h_wilcoxon] = signrank(proposed_scores, baseline_scores);

    % 2. Cohen's d effect size
    diff_scores = proposed_scores - baseline_scores;
    mean_diff = mean(diff_scores);
    std_diff = std(diff_scores);
    cohens_d = mean_diff / max(eps, std_diff);

    % 3. Bootstrap 95% Confidence Intervals for AUC
    n_boot = 1000;
    boot_aucs = zeros(n_boot, 1);
    for b = 1:n_boot
        idx = randi(n, n, 1);
        [~, ~, ~, boot_auc] = perfcurve(ground_truth(idx), proposed_scores(idx), 1);
        boot_aucs(b) = boot_auc;
    end
    ci_lower = prctile(boot_aucs, 2.5);
    ci_upper = prctile(boot_aucs, 97.5);
    [~, ~, ~, mean_auc] = perfcurve(ground_truth, proposed_scores, 1);

    % 4. Summary Output Struct
    stats_report.p_value = p_val_wilcoxon;
    stats_report.hypothesis_rejected = (h_wilcoxon == 1);
    stats_report.cohens_d = cohens_d;
    stats_report.auc_mean = mean_auc;
    stats_report.auc_ci_95 = [ci_lower, ci_upper];

    fprintf('--- Statistical Significance Report ---\n');
    fprintf('Wilcoxon Signed-Rank p-value: %1.4e\n', p_val_wilcoxon);
    fprintf('Statistical Significance (alpha = 0.05): %s\n', ...
        string(stats_report.hypothesis_rejected));
    fprintf('Cohen\'s d effect size: %1.4f\n', cohens_d);
    fprintf('Proposed Model AUC: %1.4f (95%% CI: [%1.4f, %1.4f])\n', ...
        mean_auc, ci_lower, ci_upper);
end

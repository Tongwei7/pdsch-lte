function render_dataset_report(directory)
    % RENDER_DATASET_REPORT 对照接收 PRB 功率、发送占用真值和 CRS 信道估计。
    %   directory 为已验收批次目录，按 manifest.csv 选代表捕获。
    %   三行依次是连续分配深衰落、真实空闲间隔、同频叠加。
    %   功率为未标定数字量，图只用于数据质量检查。
    manifest = readtable(fullfile(directory, 'manifest.csv'), 'TextType', 'string');
    tags = {'deep_contiguous', 'true_gap', 'cochannel'};
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1400 1050]);
    for row = 1:3
        found = find(strcmp(manifest.scenario, tags{row}), 1);
        if isempty(found)
            continue
        end
        a = load(fullfile(directory, char(manifest.path(found))));
        realPart = reshape(a.rx_real, size(a.rx_real, 1), 14, []);
        imagPart = reshape(a.rx_imag, size(a.rx_imag, 1), 14, []);
        K = size(realPart, 1);
        S = size(realPart, 3);
        N = K / 12;
        power = realPart.^2 + imagPart.^2;
        rbPower = reshape(mean(mean(reshape(power, 12, N, 14, S), 1), 3), N, S);
        subplot(3, 3, (row - 1) * 3 + 1);
        imagesc(0:S - 1, 0:N - 1, 10 * log10(rbPower + realmin('single')));
        axis xy;
        colorbar;
        title([strrep(tags{row}, '_', ' ') ': received PRB power']);
        xlabel('Aligned subframe');
        ylabel('PRB (zero based)');
        subplot(3, 3, (row - 1) * 3 + 2);
        imagesc(0:S - 1, 0:N - 1, a.label_prb, [0 1]);
        axis xy;
        title('Target PDSCH true occupancy');
        xlabel('Aligned subframe');
        ylabel('PRB (zero based)');
        hh = a.crs_h_real.^2 + a.crs_h_imag.^2;
        hh = reshape(mean(mean(hh, 4), 2), K, S);
        hh = reshape(mean(reshape(hh, 12, N, S), 1), N, S);
        subplot(3, 3, (row - 1) * 3 + 3);
        imagesc(0:S - 1, 0:N - 1, 10 * log10(hh + realmin('single')));
        axis xy;
        colorbar;
        title('Received CRS channel power estimate');
        xlabel('Aligned subframe');
        ylabel('PRB (zero based)');
    end
    folder = fullfile(directory, 'reports');
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    print(f, fullfile(folder, 'observations_and_truth.png'), '-dpng', '-r120');
    close(f);
end

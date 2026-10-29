function summary = lte_dataset_generate(configPath, workerIndex, workerCount)
    % LTE_DATASET_GENERATE 生成选定 LTE 小区的 PDSCH 占用训练数据。
    %   summary = lte_dataset_generate(configPath, workerIndex, workerCount)
    %
    %   configPath：该批次 JSON 配置路径。
    %   workerIndex：从 0 开始的进程编号；workerCount：本批次进程总数。
    %   summary：本进程完成、拒绝、失败的捕获数量和耗时（秒）。
    %
    %   流程：标准发送映射 -> 传播/小区叠加 -> 接收同步/FFT -> 标签对齐。
    %   输入特征只来自接收观测，发送端真值只用于监督标签与审计。
    %   本程序读取本地合成 SIB 资产，不访问原始实测 IQ 或 Y 盘。

    %% 批次初始化与分工
    if nargin < 2
        workerIndex = 0;
    end
    if nargin < 3
        workerCount = 1;
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    cfg = jsondecode(fileread(configPath));
    if ~isfield(cfg, 'generator_revision')
        cfg.generator_revision = 'development-unversioned';
    end
    assert(workerIndex >= 0 && workerIndex < workerCount);
    assert(exist('ltePDSCH', 'file') ~= 0, 'LTE Toolbox is required');
    out = char(cfg.output_root);
    if ~exist(out, 'dir')
        mkdir(out);
    end
    for d = {'shards', 'audit', 'status', 'rejected', 'iq'}
        if ~exist(fullfile(out, d{1}), 'dir')
            mkdir(fullfile(out, d{1}));
        end
    end
    jobs = cfg.jobs;
    done = 0;
    rejected = 0;
    failed = 0;
    started = tic;
    % 按任务序号取模分工，每份捕获仅由一个进程写入。
    for j = 1:numel(jobs)
        if mod(j - 1, workerCount) ~= workerIndex
            continue
        end
        job = jobs(j);
        id = char(job.id);
        marker = fullfile(out, 'status', [id '.json']);
        % 仅续用同一版本的完成记录，其他情况重新生成。
        if exist(marker, 'file')
            old = struct();
            try
                old = jsondecode(fileread(marker));
            catch
            end
            if isfield(old, 'state') && ...
                strcmp(old.state, 'complete') && ...
                isfield(old, 'generator_revision') && ...
                strcmp(old.generator_revision, cfg.generator_revision)
                done = done + 1;
                continue
            end
        end
        fprintf('START %s scenario=%s seed=%d\n', id, job.scenario, job.seed);
        try
            r = generateCapture(job, cfg, root);
            r.id = id;
            r.split = job.split;
            r.group_id = job.group_id;
            r.generator_revision = cfg.generator_revision;
            % 同步/标签对齐不可靠时拒绝，不能改成全零负样本。
            if ~r.alignment_valid
                r.state = 'rejected';
                rejected = rejected + 1;
                writejson(fullfile(out, 'rejected', [id '.json']), r);
            else
                r.state = 'complete';
                done = done + 1;
            end
            writejson(marker, r);
        catch err
            % failed 是程序执行错误，与接收条件造成的 rejected 分开记录。
            failed = failed + 1;
            r = struct('id', id, 'state', 'failed', 'identifier', err.identifier, 'message', ...
                err.message, ...
                'stack', err.stack, 'split', job.split, 'group_id', job.group_id, ...
                'generator_revision', cfg.generator_revision);
            writejson(marker, r);
            fprintf(2, 'FAILED %s: %s\n%s\n', id, err.message, getReport(err, 'extended', ...
                'hyperlinks', 'off'));
        end
        summary = struct('worker', workerIndex, 'workers', workerCount, 'completed', done, ...
            'rejected', rejected, ...
            'failed', failed, 'elapsed_seconds', toc(started), 'last_job', id);
        writejson(fullfile(out, sprintf('worker_%02d.json', workerIndex)), summary);
        if isfield(cfg, 'stop_on_failure') && cfg.stop_on_failure && failed > 0
            error('dataset:Failed', 'Acceptance job failed');
        end
    end
    summary = struct('worker', workerIndex, 'workers', workerCount, 'completed', done, ...
        'rejected', rejected, ...
        'failed', failed, 'elapsed_seconds', toc(started));
    writejson(fullfile(out, sprintf('worker_%02d.json', workerIndex)), summary);
end

% 单次捕获包括额外的同步余量；最终只导出 nkeep 个完整子帧。
function result = generateCapture(job, cfg, root)
    % GENERATECAPTURE 完成一次发送、传播、接收、标签对齐和保存。
    rng(double(job.seed), 'twister');
    % 时间：1 子帧=1 ms；额外 40 ms 用于任意捕获起点和接收同步。
    % fs/wideFs 单位为 Hz，宽带叠加采用 61.44 MS/s，最终采集采用 30.72 MS/s。
    nkeep = double(cfg.keep_subframes);
    nsf = nkeep + 40;
    fs = 30720000;
    wideFs = 2 * fs;
    enb = struct('NDLRB', double(job.nrb), 'NCellID', double(job.pci), 'CellRefP', 2, ...
        'DuplexMode', 'FDD', 'CyclicPrefix', 'Normal', 'Ng', 'One', 'PHICHDuration', ...
        'Normal', ...
        'NFrame', double(job.start_sfn), 'NSubframe', 0, 'CFI', 2);
    [tx, truth, grants, base] = makeCell(enb, nsf, job, root, true);
    oi = lteOFDMInfo(enb);
    nativeFs = oi.SamplingRate;
    [target, channelInfo] = propagate(tx, job.channel, nativeFs, double(job.seed) + 17);
    target = resample(target, wideFs, nativeFs);
    maxN = size(target, 1);
    y = target;
    interferers = struct('PCI', {}, 'NDLRB', {}, 'relative_center_hz', {}, ...
        'desired_to_interferer_db', {}, ...
        'channel', {}, 'time_offset_samples_at_wide_fs', {});
    %% 同频与邻频小区叠加
    % 每个邻区独立生成标准 LTE 信号并传播。邻区数据不进入目标 truth 标签。
    % 相对时延以 wideFs 的采样点为单位；SIR 按目标与邻区平均功率设定。
    if job.cochannel
        other = enb;
        other.NCellID = mod(enb.NCellID + 215, 504);
        if other.NCellID == enb.NCellID
            other.NCellID = mod(other.NCellID + 1, 504);
        end
        jj = job;
        jj.seed = double(job.seed) + 10001;
        jj.scenario = 'interferer';
        jj.channel = 'EVA';
        rng(jj.seed, 'twister');
        [w, ~, ~, ~] = makeCell(other, nsf, jj, root, false);
        [z, ci] = propagate(w, 'EVA', nativeFs, jj.seed + 17);
        z = resample(z, wideFs, nativeFs);
        shift = double(job.interferer_delay_samples);
        z = [zeros(shift, 1); z];
        z = fitlength(z, maxN);
        z = z .* exp(2j * pi * double(job.interferer_cfo_hz) * (0:maxN - 1)' / wideFs);
        z = z * sqrt(mean(abs(target).^2) / max(mean(abs(z).^2), ...
            eps)) * 10^(-double(job.sir_db) / 20);
        y = y + z;
        interferers = struct('PCI', other.NCellID, 'NDLRB', other.NDLRB, 'relative_center_hz', ...
            0, ...
            'desired_to_interferer_db', job.sir_db, 'channel', ci, ...
            'time_offset_samples_at_wide_fs', shift);
    end
    if job.adjacent
        % 复现实测发现的部分邻载波：10 MHz、相对中心约 -14.4 MHz。
        other = enb;
        other.NDLRB = 50;
        other.NCellID = 492;
        jj = job;
        jj.seed = double(job.seed) + 20001;
        jj.scenario = 'interferer';
        jj.channel = 'EPA';
        rng(jj.seed, 'twister');
        [w, ~, ~, ~] = makeCell(other, nsf, jj, root, false);
        [z, ci] = propagate(w, 'EPA', 15360000, jj.seed + 17);
        z = resample(z, wideFs, 15360000);
        z = fitlength(z, maxN);
        z = z .* exp(-2j * pi * 14400000 * (0:maxN - 1)' / wideFs);
        z = z * sqrt(mean(abs(target).^2) / max(mean(abs(z).^2), ...
            eps)) * 10^(-double(job.sir_db) / 20);
        y = y + z;
        interferers(end + 1) = struct('PCI', 492, 'NDLRB', 50, 'relative_center_hz', -14400000, ...
            'desired_to_interferer_db', job.sir_db, 'channel', ci, ...
            'time_offset_samples_at_wide_fs', 0);
    end
    % 先在较宽采样带宽叠加，再抗混叠降采样；带外邻载波不会绕回另一侧。
    center = double(job.center_hz);
    t = (0:size(y, 1) - 1)' / wideFs;
    y = y .* exp(2j * pi * (center + double(job.cfo_hz)) * t);
    y = resample(y, 1, 2);
    target = resample(target, 1, 2);
    rng(double(job.seed) + 30001, 'twister');
    %% 噪声与捕获模拟
    % SNR 定义为目标时域宽带平均功率/复数白噪声方差，不是逐 RE SINR。
    noiseVar = mean(abs(target).^2) * 10^(-double(job.snr_db) / 10);
    y = y + sqrt(noiseVar / 2) * (randn(size(y)) + 1j * randn(size(y)));
    cut = round(double(job.capture_cut_ms) * fs / 1000);
    y = y(cut + 1:end);
    % 数字幅度为合成接收缩放，没有 RF 标定，不能解释为 dBm。
    digitalScale = double(job.digital_gain) * .001 / sqrt(max(mean(abs(y).^2), eps));
    y = y * digitalScale;
    result = struct('alignment_valid', false, 'scenario', job.scenario, 'seed', job.seed, ...
        'files', {{}}, ...
        'capture_sample_rate_hz', fs, 'source_subframes', nsf, 'reason', '', ...
        'capture_digital_rms', sqrt(mean(abs(y).^2)), 'capture_papr_db', ...
        10 * log10(max(abs(y).^2) / max(mean(abs(y).^2), eps)));
    [rx, front] = receive(y, enb.NCellID, nkeep);
    if ~front.ok
        result.reason = front.reason;
        result.frontend = front;
        return
    end
    %% 发送真值与接收坐标对齐
    % 接收机先独立同步，此处再用已知发送时间检查帧映射；SFN 允许正常回绕。
    first = mod(front.first_sfn - enb.NFrame, 1024) * 10;
    timingError = (cut + front.full_offset * fs / front.fs) / fs - first * .001;
    if first + nkeep > nsf || abs(timingError) > .00015 || front.nrb ~= enb.NDLRB
        result.reason = 'Receiver/transmitter frame mapping mismatch';
        result.frontend = front;
        result.timing_error_seconds = timingError;
        return
    end
    K = enb.NDLRB * 12;
    idx = first + (1:nkeep);
    % truth 记录实际映射的数据 RE：深衰落不改变标签，CRS 等正常孔洞为 0。
    % 拆成 [12 子载波, PRB, 14 符号, 子帧]，按 PRB/时隙求数据占用并集。
    label_re = uint8(truth(:, :, idx)); %#ok<NASGU>
    label_prb = uint8(reshape(any(any(reshape(label_re, 12, enb.NDLRB, 14, nkeep), 1), 3), ...
        enb.NDLRB, nkeep)); %#ok<NASGU>
    label_prb_slot = zeros(enb.NDLRB, 2, nkeep, 'uint8');
    for slot = 1:2
        q = label_re(:, (slot - 1) * 7 + (1:7), :);
        label_prb_slot(:, slot, :) = uint8(reshape(any(any(reshape(q, 12, enb.NDLRB, 7, ...
            nkeep), 1), 3), enb.NDLRB, 1, nkeep));
    end
    assert(isequal(label_prb, reshape(max(label_prb_slot, [], 2), enb.NDLRB, nkeep)), ...
        'Slot/PRB label mismatch');
    % AI 约定轴序 [K,14,R,S]：K 为有效子载波，R=1 为实际接收支路，S 为子帧数。
    rx_real = single(real(reshape(rx, K, 14, 1, nkeep)));
    rx_imag = single(imag(reshape(rx, K, 14, 1, nkeep))); %#ok<NASGU>
    subframe_index = int64(first:first + nkeep - 1); %#ok<NASGU>
    public_excluded = front.public_excluded;
    public_valid = front.public_valid; %#ok<NASGU>
    % CRS 数组为 [K,14,1,2,S]，最后的 2 对应参考信号端口，不是两路接收天线。
    % 这是接收估计，不是所有传输模式下理想的 PDSCH 数据层等效信道。
    crs_h_real = single(real(front.h));
    crs_h_imag = single(imag(front.h)); %#ok<NASGU>
    crs_noise_variance = single(front.noise);
    cfi_estimate = uint8(front.cfi);
    cfi_hard_mismatch = single(front.cfi_error); %#ok<NASGU>
    front = rmfield(front, {'h', 'public_excluded', 'public_valid'});
    shard = fullfile(char(cfg.output_root), 'shards', [char(job.id) '.mat']);
    %% 训练分块、审计真值和可选 IQ 分开落盘
    % -v7 兼容当前 scipy.io.loadmat；临时文件写完后再替换正式文件。
    % audit 中真实 RNTI、分配和信道设定只用于核查，禁止作为部署模型输入。
    temporary = [shard '.partial.mat'];
    save(temporary, 'rx_real', 'rx_imag', 'subframe_index', 'label_re', 'label_prb', ...
        'label_prb_slot', ...
        'public_excluded', 'public_valid', 'crs_h_real', 'crs_h_imag', 'crs_noise_variance', ...
        'cfi_estimate', 'cfi_hard_mismatch', '-v7');
    movefile(temporary, shard, 'f');
    audit = struct('dataset_version', 'lte-pdsch-occupancy-v1', 'reader_schema', ...
        'pdsch-occupancy-proposal-v1', ...
        'matlab_release', version('-release'), 'generator_revision', ...
        cfg.generator_revision, 'job', job, 'target_enb', enb, 'common_config', ...
        base, ...
        'target_channel', channelInfo, 'interferers', interferers, 'frontend', front, ...
        'first_transmitted_subframe', first, 'mapping_timing_error_seconds', ...
        timingError, ...
        'oracle_wideband_noise_variance_before_gain', noiseVar, ...
        'applied_digital_scale', digitalScale, 'grants', grants, ...
        'truth_source', ...
        'ltePDSCHIndices for all actually transmitted target grants; union across antenna ports', ...
        'snr_definition', ...
        'Target average time-domain wideband power divided by complex AWGN variance, before digital gain', ...
        'limitations', ...
        {{'Synthetic scheduler and MCS distribution are assumptions, not recovered measured business statistics', ...
        'CRS estimates are not ideal PDSCH equivalent channels for every transmission mode', ...
        'No full core network, dynamic RRC or closed-loop HARQ ACK simulator'}});
    writejson(fullfile(char(cfg.output_root), 'audit', [char(job.id) '.json']), audit);
    if job.save_iq
        iq_real = single(real(y));
        iq_imag = single(imag(y));
        sample_rate_hz = fs; %#ok<NASGU>
        save(fullfile(char(cfg.output_root), 'iq', [char(job.id) '.mat']), 'iq_real', ...
            'iq_imag', 'sample_rate_hz', '-v7');
    end
    result.alignment_valid = true;
    result.files = {['shards/' char(job.id) '.mat']};
    result.subframes = nkeep;
    result.windows = floor(nkeep / 10);
    result.nrb = enb.NDLRB;
    result.positive_prb_fraction = mean(label_prb(:));
    result.frontend = front;
    result.grants = numel(grants);
    result.common_pdsch = sum([grants.rnti] == 65535);
    result.timing_error_seconds = timingError;
end

function [wave, truth, grants, base] = makeCell(enb, nsf, job, root, isTarget)
    % MAKECELL 构造一个小区的连续标准网格及真实数据映射。
    %   truth：[12*NDLRB,14,nsf]；grants：实际成功发送的 DCI/PRB/MCS 记录。
    %   isTarget 用于区分目标普通业务缺席和邻区仍然正常发送的情况。
    N = enb.NDLRB;
    K = 12 * N;
    % RMC 仅负责空业务条件下的公共信号；用户 DCI/PDSCH 在后面显式生成。
    rmc = lteRMCDL('R.3');
    fields = fieldnames(enb);
    for f = 1:numel(fields)
        rmc.(fields{f}) = enb.(fields{f});
    end
    rmc.PDSCH.TxScheme = 'TxDiversity';
    rmc.PDSCH.NLayers = 2;
    rmc.TotSubframes = nsf;
    rmc.PDSCH.PRBSet = (0:N - 1)';
    rmc.OCNGPDSCHEnable = 'Off';
    rmc.OCNGPDCCHEnable = 'Off';
    if isfield(rmc, 'SIB')
        rmc.SIB.Enable = 'Off';
    end
    rng(double(job.seed), 'twister');
    cfiValues = ones(1, nsf);
    cfiValues(rand(1, nsf) > .55) = 2;
    cfiValues(rand(1, nsf) > .94) = 3;
    % RMC 按绝对 SFN/子帧编号索引 CFI，需将捕获相对序列旋转到对应位置。
    rmc.CFI = zeros(1, nsf);
    rmc.CFI(mod(enb.NFrame * 10 + (0:nsf - 1), nsf) + 1) = cfiValues;
    [~, grid] = lteRMCDLTool(rmc, []);
    truth = false(K, 14, nsf);
    grants = struct('subframe', {}, 'sfn', {}, 'rnti', {}, 'mcs', {}, 'tbs', {}, 'scheme', {}, ...
        'nlayers', {}, 'prbs', {}, 'dci', {}, 'pdcch_format', {}, ...
        'pdcch_bit_interval', {}, 'rv', {});
    % 分别统计控制容量不足、编码容量不足和资源冲突导致的未发送候选。
    controlRejected = 0;
    capacityRejected = 0;
    collisionRejected = 0;
    % 加载有效 ASN.1 合成 SIB1/SIB2，系统信息 PDSCH 也属于占用正标签。
    sib = load(fullfile(root, 'reference', 'synthetic_sib1.mat'));
    sibBits = double(sib.sib1_bits(:));
    si = load(fullfile(root, 'reference', sprintf('synthetic_sib2_%d.mat', N)));
    siBits = double(si.sib2_bits(:));
    template = [];
    hold = 0;
    ueRnti = randperm(59901, 6) + 99;
    ndi = false(6, 8);
    for si = 0:nsf - 1
        enb.NSubframe = mod(si, 10);
        enb.NFrame = mod(double(job.start_sfn) + floor(si / 10), 1024);
        enb.CFI = cfiValues(si + 1);
        sg = grid(:, si * 14 + (1:14), :);
        occupied = false(N, 2);
        pi = ltePDCCHInfo(enb);
        pbits = -ones(pi.MTot, 1);
        pending = struct('rnti', {}, 'mcs', {}, 'prbs', {}, 'dci', {}, 'common', {}, 'scheme', ...
            {}, 'layers', {}, 'pmi', {}, 'ue', {}, 'payload', {});
        % 公共系统信息优先排入队列，使用 SI-RNTI=65535 和 QPSK。
        % SIB1 调度与 SIB2 发送周期在本版合成配置中互相对应。
        sendSib1 = enb.NSubframe == 5 && mod(enb.NFrame, 2) == 0;
        sendSib2 = enb.NSubframe == 0 && mod(enb.NFrame, 16) == 0;
        if sendSib1 || sendSib2
            payload = sibBits;
            if sendSib2
                payload = siBits;
            end
            count = min(8, N);
            st = mod(floor(si / 20) * 7, N - count + 1);
            d = struct('DCIFormat', 'Format1A', 'AllocationType', 0, 'Allocation', ...
                struct('RIV', riv(N, st, count)), ...
                'TPCPUCCH', 1, 'NewData', 0, 'RV', ...
                mod(ceil(1.5 * mod(floor(enb.NFrame / 2), 4)), 4));
            ts = double(lteTBS(3, 0:26));
            mi = find(ts >= numel(payload), 1) - 1;
            d.ModCoding = mi;
            if sendSib2
                d.RV = 0;
            end
            pending(end + 1) = struct('rnti', 65535, 'mcs', mi, 'prbs', (st:st + count - 1)', ...
                'dci', d, 'common', true, 'scheme', 'TxDiversity', 'layers', ...
                2, 'pmi', 0, 'ue', 0, 'payload', payload);
        end
        % 分配模板可持续多个子帧，但每次发送仍生成新传输块；没有 HARQ 反馈闭环。
        if hold <= 0
            [template, hold] = schedule(N, char(job.scenario), ueRnti, job);
        end
        hold = hold - 1;
        % target_absent 只关闭目标普通业务，公共信号和系统信息仍保留。
        if ~(strcmp(job.scenario, 'target_absent') && isTarget)
            for u = 1:numel(template)
                a = template(u);
                d = a.dci;
                harq = mod(si, 8);
                ndi(a.ue, harq + 1) = ~ndi(a.ue, harq + 1);
                if strcmp(d.DCIFormat, 'Format2')
                    d.HARQNo = harq;
                    d.NewData1 = double(ndi(a.ue, harq + 1));
                    d.NewData2 = d.NewData1;
                else
                    d.HARQNo = harq;
                    d.NewData = double(ndi(a.ue, harq + 1));
                end
                a.dci = d;
                pending(end + 1) = a; %#ok<AGROW>
            end
        end
        for u = 1:numel(pending)
            a = pending(u);
            prbs = a.prbs;
            if size(prbs, 2) == 1
                prbs = repmat(prbs, 1, 2);
            end
            busy = occupied(sub2ind([N 2], prbs + 1, repmat(1:2, size(prbs, 1), 1)));
            if any(busy(:))
                collisionRejected = collisionRejected + 1;
                continue
            end
            d = a.dci;
            [d, db] = lteDCI(enb, d);
            % 从 DCI 反算 PRB，核对控制内容与实际映射位置。
            recovered = lteDCIResourceAllocation(enb, d);
            if size(recovered, 2) == 1
                recovered = repmat(recovered, 1, 2);
            end
            assert(isequal(sort(prbs, 1), sort(double(recovered), 1)), ...
                'DCI allocation and actual PRB set differ');
            ue = struct('RNTI', a.rnti, 'PDCCHFormat', 1, 'NDLRB', N);
            if a.common
                ue.PDCCHFormat = 2;
            end
            candidates = ltePDCCHSpace(enb, ue);
            if a.common
                candidates = ltePDCCHSpace(enb, struct('PDCCHFormat', 2));
            end
            % 在合法 PDCCH 搜索空间中选空闲 CCE；没有位置时不发送该条 PDSCH。
            chosen = [];
            for c = 1:size(candidates, 1)
                bitRange = candidates(c, 1):candidates(c, 2);
                if all(pbits(bitRange) == -1)
                    chosen = bitRange;
                    break
                end
            end
            if isempty(chosen)
                controlRejected = controlRejected + 1;
                continue
            end
            pd = struct('RNTI', a.rnti, 'TxScheme', a.scheme, 'NLayers', a.layers, 'PMISet', ...
                a.pmi, ...
                'PRBSet', prbs, 'RV', 0, 'NTurboDecIts', 8);
            if a.common
                pd.Modulation = 'QPSK';
                pd.RV = d.RV;
                tbs = double(lteTBS(3, a.mcs));
                bits = [a.payload; zeros(tbs - numel(a.payload), 1)];
            else
                [itbs, modulation] = lteMCS(a.mcs, 'PDSCH');
                if iscell(modulation)
                    modulation = modulation{1};
                end
                pd.Modulation = modulation;
                tbs = double(lteTBS(size(prbs, 1), itbs));
                if strcmp(a.scheme, 'SpatialMux') && a.layers == 2
                    pd.Modulation = {modulation, modulation};
                    pd.RV = [0 0];
                    tbs = [tbs tbs];
                    bits = {randi([0 1], tbs(1), 1), randi([0 1], tbs(2), 1)};
                else
                    bits = randi([0 1], tbs, 1);
                end
            end
            [ind, info] = ltePDSCHIndices(enb, pd, prbs);
            if any(tbs ./ double(info.G) > .94)
                capacityRejected = capacityRejected + 1;
                continue
            end
            assert(all(sg(ind(:)) == 0), 'PDSCH collision with an existing common/data RE');
            cw = lteDLSCH(enb, pd, info.G, bits);
            symbols = ltePDSCH(enb, pd, cw);
            sg(ind) = symbols;
            pbits(chosen) = lteDCIEncode(ue, db);
            % 将端口平面索引投影到二维 RE 坐标，合并所有目标发射端口的数据映射。
            mask = false(K, 14);
            mask(mod(double(ind(:)) - 1, K * 14) + 1) = true;
            truth(:, :, si + 1) = truth(:, :, si + 1) | mask;
            occupied(sub2ind([N 2], prbs + 1, repmat(1:2, size(prbs, 1), 1))) = true;
            % 验收中理想反解实际发送符号，检查 CRC 和传输块；不代表真实接收可无误解码。
            if isfield(job, 'verify_payload') && job.verify_payload && ~strcmp(a.scheme, ...
                'SpatialMux')
                soft = ltePDSCHDecode(enb, pd, symbols);
                [decoded, crc] = lteDLSCHDecode(enb, pd, tbs, soft);
                assert(all(crc == 0), 'Noiseless transport block CRC failed');
                assert(isequal(double(decoded{1}), double(bits)), 'Decoded transport differs');
            end
            grants(end + 1) = struct('subframe', si, 'sfn', enb.NFrame, 'rnti', a.rnti, 'mcs', ...
                a.mcs, ...
                'tbs', tbs, 'scheme', a.scheme, 'nlayers', a.layers, 'prbs', ...
                prbs, 'dci', d, ...
                'pdcch_format', ue.PDCCHFormat, 'pdcch_bit_interval', ...
                [chosen(1) chosen(end)] - 1, 'rv', pd.RV); %#ok<AGROW>
        end
        ps = ltePDCCH(enb, pbits);
        sg(ltePDCCHIndices(enb)) = ps;
        grid(:, si * 14 + (1:14), :) = sg;
    end
    enb.NSubframe = 0;
    enb.NFrame = double(job.start_sfn);
    [wave, ~] = lteOFDMModulate(enb, grid);
    base = struct('CFI', cfiValues, 'control_grants_rejected', controlRejected, ...
        'capacity_grants_rejected', capacityRejected, 'overlap_grants_rejected', ...
        collisionRejected, ...
        'scheduler', 'open-loop synthetic multi-UE scheduler; new transport blocks RV0, no feedback-driven retransmissions');
end

function [a, hold] = schedule(N, scenario, rntis, job)
    % SCHEDULE 返回合成调度列表 a 和模板持续子帧数 hold。
    %   N 为系统 PRB 数；所有 PRB 坐标从 0 开始。业务比例为覆盖设计。
    a = struct('rnti', {}, 'mcs', {}, 'prbs', {}, 'dci', {}, 'common', {}, 'scheme', {}, ...
        'layers', {}, 'pmi', {}, 'ue', {}, 'payload', {});
    hold = randi([1 8]);
    if strcmp(scenario, 'target_absent')
        return
    end
    if rand < .12
        return
    end
    % 对照连续分配遇衰落与真实空闲间隔，避免网络只学会“填洞”。
    if ismember(scenario, {'deep_contiguous', 'true_gap'})
        span = randi([max(8, floor(N * .3)), max(9, floor(N * .75))]);
        st = randi([0 N - span]);
        gap = max(1, round(span * .15));
        left = floor((span - gap) / 2);
        sets = {(st:st + span - 1)'};
        if strcmp(scenario, 'true_gap')
            sets = {(st:st + left - 1)', (st + left + gap:st + span - 1)'};
        end
        hold = randi([3 10]);
    else
        edges = sort(randperm(N - 1, min(5, N - 1)));
        edges = [0 edges N];
        sets = {};
        for k = 1:numel(edges) - 1
            if rand < double(job.load_probability) && edges(k + 1) > edges(k)
                sets{end + 1} = (edges(k):edges(k + 1) - 1)'; %#ok<AGROW>
            end
        end
    end
    if isfield(job, 'distributed') && job.distributed
        sets = {(0:min(11, N - 1))'};
    end
    for u = 1:numel(sets)
        pr = sets{u};
        if isempty(pr)
            continue
        end
        mcs = randi([0 26]);
        d = struct('DCIFormat', 'Format1A', 'AllocationType', 0, 'Allocation', struct('RIV', ...
            riv(N, pr(1), numel(pr))), ...
            'ModCoding', mcs, 'RV', 0, 'TPCPUCCH', 0, 'NewData', 1, 'HARQNo', 0);
        scheme = 'TxDiversity';
        layers = 2;
        pmi = 0;
        if isfield(job, 'spatial_mux') && job.spatial_mux && mod(u, 2) == 1
            % 两个 CRS 端口、秩 2 码本空间复用和两个码字；这里发送两路独立数据流。
            d = lteDCI(struct('NDLRB', N, 'CellRefP', 2, 'DuplexMode', 'FDD'), ...
                struct('DCIFormat', 'Format2', 'AllocationType', 0));
            ng = numel(d.Allocation.Bitmap);
            P = ceil(N / ng);
            groups = unique(floor(pr / P) + 1);
            d.Allocation.Bitmap(:) = '0';
            d.Allocation.Bitmap(groups) = '1';
            pr = double(lteDCIResourceAllocation(struct('NDLRB', N, 'CellRefP', 2), d));
            d.ModCoding1 = mcs;
            d.ModCoding2 = mcs;
            d.NewData1 = 1;
            d.NewData2 = 1;
            d.RV1 = 0;
            d.RV2 = 0;
            d.HARQNo = 0;
            d.SwapFlag = 0;
            d.PrecodingInfo = 2;
            scheme = 'SpatialMux';
            layers = 2;
            pmi = 0;
        elseif isfield(job, 'distributed') && job.distributed && numel(sets) == 1
            % 用标准映射器计算两个时隙的 PRB，分布式分配的两列可以不同。
            d.AllocationType = 1;
            d.Allocation.Gap = 0;
            try
                pr = double(lteDCIResourceAllocation(struct('NDLRB', N, 'CellRefP', 2), d));
                if isempty(pr) || any(pr(:) >= N)
                    continue
                end
            catch
                continue
            end
        end
        a(end + 1) = struct('rnti', rntis(u), 'mcs', mcs, 'prbs', pr, 'dci', d, 'common', false, ...
            'scheme', scheme, 'layers', layers, 'pmi', pmi, 'ue', u, 'payload', ...
            []); %#ok<AGROW>
    end
end

function n = riv(N, start, count)
    % RIV 将连续资源块起点/长度编码成 LTE 资源指示值。
    %   start 从 0 开始，count 是块数；两种表达对应短/长连续分配。
    if count - 1 <= floor(N / 2)
        n = N * (count - 1) + start;
    else
        n = N * (N - count + 1) + N - 1 - start;
    end
end

function [y, info] = propagate(x, name, fs, seed)
    % PROPAGATE 将双发射端口波形传播为一路接收观测，并返回模型设置。
    %   fs 为 Hz；双径时延另存为采样点数和秒，多普勒为 Hz。
    if strcmp(name, 'Off')
        h = [1; .4j];
        y = x * h / sqrt(sum(abs(h).^2));
        info = struct('profile', 'Off', 'doppler_hz', 0);
        return
    end
    % 双径在时域叠加，物理地产生频率凹陷；没有直接抹除资源网格。
    if strcmp(name, 'TwoRay')
        rng(seed, 'twister');
        delay = round(fs * (.25 + rand * 3.5) * 1e-6);
        phase = 2 * pi * rand;
        h = [1; -exp(1j * phase)] / sqrt(2);
        z = x(:, 1) + exp(.7j) * x(:, 2);
        z = z / sqrt(2);
        y = filter([h(1); zeros(delay - 1, 1); h(2)], 1, z);
        info = struct('profile', 'Physical two-ray notch stress model', 'delay_samples', ...
            delay, 'delay_seconds', delay / fs, ...
            'path_coefficients_real', real(h), 'path_coefficients_imag', imag(h), ...
            'doppler_hz', 0);
        return
    end
    % 多普勒单位 Hz；5/70/300 是本版覆盖设定，不是实测速度的唯一反演。
    fd = 5;
    if strcmp(name, 'EVA')
        fd = 70;
    elseif strcmp(name, 'ETU')
        fd = 300;
    end
    model = struct('NRxAnts', 1, 'MIMOCorrelation', 'Low', 'DelayProfile', name, ...
        'DopplerFreq', fd, ...
        'SamplingRate', fs, 'Seed', seed, 'InitTime', 0, 'ModelType', 'GMEDS', ...
        'NTerms', 16, 'NormalizeTxAnts', 'On', 'NormalizePathGains', 'On', ...
        'InitPhase', 'Random');
    [y, ~] = lteFadingChannel(model, x);
    info = model;
end

function [g, r] = receive(y, targetPCI, nkeep)
    % RECEIVE 对选定目标小区进行捕获，返回对齐后的复数网格 g 和接收记录 r。
    %   targetPCI 是选定的小区标识；没有传入真实发射中心、频偏或时间偏移。
    % 粗载波候选来自实测场景族，每个候选都要通过接收同步与重复 PBCH 验证。
    attempts = struct('coarse_hz', {}, 'ok', {}, 'reason', {});
    g = [];
    for coarse = [0 600000 -500000]
        [g, r] = tryReceive(y, coarse, targetPCI, nkeep);
        attempts(end + 1) = struct('coarse_hz', coarse, 'ok', r.ok, 'reason', r.reason); %#ok<AGROW>
        if r.ok
            r.acquisition_attempts = attempts;
            return
        end
    end
    r.acquisition_attempts = attempts;
end

function [g, r] = tryReceive(y, coarse, targetPCI, nkeep)
    % TRYRECEIVE 尝试一个粗载波位置，执行频偏估计、同步、广播与全带宽 FFT。
    %   coarse 单位 Hz；先在 1.92 MS/s 窄带恢复公共配置，再处理实际带宽。
    fs = 30720000;
    z = y .* exp(-2j * pi * coarse * (0:numel(y) - 1)' / fs);
    small = resample(z, 1, 16);
    enb = struct('NDLRB', 6, 'NCellID', targetPCI, 'CellRefP', 2, 'NSubframe', 0, 'NFrame', 0, ...
        'DuplexMode', 'FDD', 'CyclicPrefix', 'Normal', 'Ng', 'One', 'PHICHDuration', ...
        'Normal');
    r = struct('ok', false, 'reason', '', 'coarse_center_hz', coarse);
    g = [];
    if numel(small) < 19200 * 3
        r.reason = 'Insufficient capture';
        return
    end
    cfo = lteFrequencyOffset(enb, small);
    small = lteFrequencyCorrect(enb, small, cfo);
    [pci, off] = lteCellSearch(enb, small(1:min(end, 19200 * 3)), ...
        struct('MaxCellCount', 1, 'CellIDs', targetPCI, 'SSSDetection', ...
        'PostFFT'));
    if isempty(pci) || pci ~= targetPCI
        r.reason = 'Target PSS/SSS not acquired';
        return
    end
    cec = struct('PilotAverage', 'UserDefined', 'FreqWindow', 9, 'TimeWindow', 9, ...
        'InterpType', 'cubic', ...
        'InterpWindow', 'Centred', 'InterpWinSize', 1, 'Reference', 'CellRS');
    % 至少两帧一致的 PBCH 才接受带宽/帧号配置；单帧偶然通过不作依据。
    decoded = [];
    match = struct([]);
    for f = 0:min(7, floor((numel(small) - off) / 19200) - 1)
        enb.NSubframe = 0;
        sg = lteOFDMDemodulate(enb, small(off + f * 19200 + (1:1920)));
        [h, n] = lteDLChannelEstimate(enb, cec, sg);
        [br, bh] = lteExtractResources(ltePBCHIndices(enb), sg, h);
        [~, ~, nf, bits, ports] = ltePBCHDecode(enb, br, bh, n);
        if ports ~= 2
            continue
        end
        m = lteMIB(bits);
        if ~ismember(m.NDLRB, [50 75 100])
            continue
        end
        record = struct('first_sfn', mod(double(m.NFrame) + double(nf) - f, 1024), 'nrb', ...
            double(m.NDLRB), 'ports', double(ports), 'ng', m.Ng, 'duration', m.PHICHDuration);
        if isempty(match)
            match = record;
            decoded = record.first_sfn;
        elseif record.first_sfn == match.first_sfn && record.nrb == match.nrb
            decoded(end + 1) = record.first_sfn;
        end %#ok<AGROW>
    end
    if numel(decoded) < 2
        r.reason = 'Fewer than two consistent PBCH frames';
        return
    end
    enb.NDLRB = match.nrb;
    enb.Ng = match.ng;
    enb.PHICHDuration = match.duration;
    enb.NFrame = match.first_sfn;
    oi = lteOFDMInfo(enb);
    full = resample(z, oi.SamplingRate, fs);
    full = lteFrequencyCorrect(enb, full, cfo);
    offset = lteDLFrameOffset(enb, full(1:min(end, round(.035 * oi.SamplingRate))));
    if abs(offset / oi.SamplingRate - off / 1920000) > .00015
        r.reason = 'Full/narrow frame offsets disagree';
        return
    end
    need = round(nkeep * .001 * oi.SamplingRate);
    if offset + need > numel(full)
        r.reason = 'Insufficient aligned subframes';
        return
    end
    g = lteOFDMDemodulate(enb, full(offset + (1:need)));
    K = 12 * match.nrb;
    r.h = complex(zeros(K, 14, 1, 2, nkeep, 'single'));
    r.noise = zeros(1, nkeep);
    r.cfi = zeros(1, nkeep);
    r.cfi_error = ones(1, nkeep);
    r.public_excluded = zeros(K, 14, nkeep, 'uint8');
    r.public_valid = zeros(K, 14, nkeep, 'uint8');
    for si = 0:nkeep - 1
        enb.NSubframe = mod(si, 10);
        enb.NFrame = mod(match.first_sfn + floor(si / 10), 1024);
        sg = g(:, si * 14 + (1:14));
        [h, n] = lteDLChannelEstimate(enb, cec, sg);
        r.h(:, :, :, :, si + 1) = single(h);
        r.noise(si + 1) = n;
        [pc, ph] = lteExtractResources(ltePCFICHIndices(enb), sg, h);
        soft = ltePCFICHDecode(enb, pc, ph, n);
        enb.CFI = lteCFIDecode(soft);
        r.cfi(si + 1) = enb.CFI;
        r.cfi_error(si + 1) = mean((soft(:) > 0) ~= logical(lteCFI(enb)));
        % 按实际恢复的公共参数建立已知排除位置，未知位置仍为未知。
        known = false(K, 14);
        for fn = {@lteCellRSIndices, @ltePSSIndices, @lteSSSIndices, @ltePBCHIndices}
            q = fn{1}(enb);
            known(mod(double(q(:)) - 1, K * 14) + 1) = true;
        end
        % PCFICH 没有独立 CRC；硬码字差异足够小时才提供控制区先验。
        % 公共图仅作输入特征；不在这里用真值修图，也不改写标签。
        if r.cfi_error(si + 1) <= .125
            known(:, 1:enb.CFI) = true;
        end
        r.public_excluded(:, :, si + 1) = uint8(known);
        r.public_valid(:, :, si + 1) = uint8(known);
    end
    r.ok = true;
    r.nrb = match.nrb;
    r.first_sfn = match.first_sfn;
    r.fs = oi.SamplingRate;
    r.full_offset = offset;
    r.narrow_offset = off;
    r.estimated_residual_cfo_hz = cfo;
    r.consistent_pbch = numel(decoded);
    r.target_pci = pci;
    r.public_source = 'Received PSS/SSS and CRC-checked MIB; PCFICH hard-codeword check <=0.125 (no CRC)';
end

function x = fitlength(x, n)
    % FITLENGTH 在尾部补零或截断，使待叠加波形具有相同采样点数。
    if size(x, 1) < n
        x = [x; zeros(n - size(x, 1), size(x, 2))];
    else
        x = x(1:n, :);
    end
end

function writejson(path, value)
    % WRITEJSON 以 UTF-8 写临时 JSON，关闭后再替换正式进度/审计文件。
    tmp = [path '.partial'];
    f = fopen(tmp, 'w', 'n', 'UTF-8');
    assert(f >= 0, 'Cannot open output');
    cl = onCleanup(@()fclose(f));
    fprintf(f, '%s', jsonencode(value));
    clear cl;
    movefile(tmp, path, 'f');
end

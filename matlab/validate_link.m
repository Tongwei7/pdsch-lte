function report = validate_link
    % VALIDATE_LINK 检查 50/75/100 RB 下的编码、预编码及理想反向解码。
    %   发射分集和双层空间复用各检查一次，共六种组合。
    %   这里直接反解发送符号，不模拟真实传播；CRC 通过不能证明一路接收
    %   能恢复两路独立数据，也不代表 AI 检测精度。
    report = struct('passed', false, 'checks', {{}});
    for nrb = [50 75 100]
        enb = struct('NDLRB', nrb, 'NCellID', 241, 'CellRefP', 2, 'NSubframe', 1, 'CFI', 2, ...
            'CyclicPrefix', 'Normal', 'DuplexMode', 'FDD', 'Ng', 'One', ...
            'PHICHDuration', 'Normal');
        for scheme = {'TxDiversity', 'SpatialMux'}
            pd = struct('RNTI', 100, 'TxScheme', scheme{1}, 'NLayers', 2, 'PMISet', 0, ...
                'PRBSet', (5:14)', 'Modulation', 'QPSK', 'RV', 0, 'NTurboDecIts', 8);
            bits = randi([0 1], 256, 1);
            tbs = 256;
            if strcmp(scheme{1}, 'SpatialMux')
                pd.Modulation = {'QPSK', 'QPSK'};
                pd.RV = [0 0];
                bits = {bits, bits};
                tbs = [256 256];
            end
            [~, info] = ltePDSCHIndices(enb, pd, pd.PRBSet);
            cw = lteDLSCH(enb, pd, info.G, bits);
            sym = ltePDSCH(enb, pd, cw);
            soft = ltePDSCHDecode(enb, pd, sym);
            [~, crc] = lteDLSCHDecode(enb, pd, tbs, soft);
            assert(all(crc == 0));
            report.checks{end + 1} = sprintf('%dRB %s coder/precoder inverse CRC0', nrb, scheme{1});
        end
    end
    report.passed = true;
    disp(jsonencode(report));
end

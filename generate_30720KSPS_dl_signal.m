close all

NID1 = 69;
NID2 = 2;
NID = 3*NID1 + NID2;

% Generated by MATLAB(R) 9.11 (R2021b) and 5G Toolbox 2.3 (R2021b).
% Generated on: 23-Oct-2022 18:15:34

%% Generating Downlink waveform
% Downlink configuration
cfgDL = nrDLCarrierConfig;
cfgDL.Label = 'Carrier1';
cfgDL.FrequencyRange = 'FR1';
cfgDL.ChannelBandwidth = 20;
cfgDL.NCellID = NID;
cfgDL.NumSubframes = 90;
cfgDL.WindowingPercent = 0;
cfgDL.SampleRate = [];
cfgDL.CarrierFrequency = 0;

%% SCS specific carriers
scscarrier = nrSCSCarrierConfig;
scscarrier.SubcarrierSpacing = 15;
scscarrier.NSizeGrid = 106;
scscarrier.NStartGrid = 3;

cfgDL.SCSCarriers = {scscarrier};

%% Bandwidth Parts
bwp = nrWavegenBWPConfig;
bwp.BandwidthPartID = 1;
bwp.Label = 'BWP1';
bwp.SubcarrierSpacing = 15;
bwp.CyclicPrefix = 'normal';
bwp.NSizeBWP = 106;
bwp.NStartBWP = 3;

cfgDL.BandwidthParts = {bwp};

%% Synchronization Signals Burst
ssburst = nrWavegenSSBurstConfig;
ssburst.BlockPattern = 'Case A';
ssburst.TransmittedBlocks = ones([1 4]);
ssburst.Period = 20;
ssburst.NCRBSSB = [];
ssburst.KSSB = 0;
ssburst.DataSource = 'MIB';
ssburst.DMRSTypeAPosition = 2;
ssburst.CellBarred = false;
ssburst.IntraFreqReselection = false;
ssburst.PDCCHConfigSIB1 = 0;
ssburst.SubcarrierSpacingCommon = 15;
ssburst.Enable = true;
ssburst.Power = 0;

cfgDL.SSBurst = ssburst;

%% CORESET and Search Space Configuration
% CORESET 1
coreset1 = nrCORESETConfig;
coreset1.CORESETID = 0;
coreset1.Label = 'CORESET0';
coreset1.FrequencyResources = ones([1 8]);
coreset1.Duration = 2;
coreset1.CCEREGMapping = 'interleaved';
coreset1.REGBundleSize = 6;
coreset1.InterleaverSize = 2;
coreset1.ShiftIndex = 0;

% CORESET 2
coreset2 = nrCORESETConfig;
coreset2.CORESETID = 1;
coreset2.Label = 'CORESET1';
coreset2.FrequencyResources = ones([1 8]);
coreset2.Duration = 2;
coreset2.CCEREGMapping = 'interleaved';
coreset2.REGBundleSize = 6;
coreset2.InterleaverSize = 2;
coreset2.ShiftIndex = 0;

cfgDL.CORESET = {coreset1,coreset2};

% Search Spaces
searchspace = nrSearchSpaceConfig;
searchspace.SearchSpaceID = 1;
searchspace.Label = 'SearchSpace1';
searchspace.CORESETID = 1;
searchspace.SearchSpaceType = 'ue';
searchspace.StartSymbolWithinSlot = 0;
searchspace.SlotPeriodAndOffset = [1 0];
searchspace.Duration = 1;
searchspace.NumCandidates = [8 8 4 2 1];

cfgDL.SearchSpaces = {searchspace};

%% PDCCH Instances Configuration
pdcch = nrWavegenPDCCHConfig;
pdcch.Enable = true;
pdcch.Label = 'PDCCH1';
pdcch.Power = 0;
pdcch.BandwidthPartID = 1;
pdcch.SearchSpaceID = 1;
pdcch.AggregationLevel = 8;
pdcch.AllocatedCandidate = 1;
pdcch.SlotAllocation = 0;
pdcch.Period = 1;
pdcch.Coding = true;
pdcch.DataBlockSize = 20;
pdcch.DataSource = 'PN9-ITU';
pdcch.RNTI = 1;
pdcch.DMRSScramblingID = 2;
pdcch.DMRSPower = 0;

cfgDL.PDCCH = {pdcch};

%% PDSCH Instances Configuration
pdsch = nrWavegenPDSCHConfig;
pdsch.Enable = true;
pdsch.Label = 'PDSCH1';
pdsch.Power = 0;
pdsch.BandwidthPartID = 1;
pdsch.Modulation = 'QPSK';
pdsch.NumLayers = 1;
pdsch.MappingType = 'A';
pdsch.ReservedCORESET = [];
pdsch.SymbolAllocation = [0 14];
pdsch.SlotAllocation = 0:9;
pdsch.Period = 10;
pdsch.PRBSet = 90:105;
pdsch.VRBToPRBInterleaving = false;
pdsch.VRBBundleSize = 2;
pdsch.NID = 1;
pdsch.RNTI = 1;
pdsch.Coding = true;
pdsch.TargetCodeRate = 0.513671875;
pdsch.TBScaling = 1;
pdsch.XOverhead = 0;
pdsch.RVSequence = [0 2 3 1];
pdsch.DataSource = 'PN9-ITU';
pdsch.DMRSPower = 0;
pdsch.EnablePTRS = false;
pdsch.PTRSPower = 0;

% PDSCH Reserved PRB
pdschReservedPRB = nrPDSCHReservedConfig;
pdschReservedPRB.PRBSet = [];
pdschReservedPRB.SymbolSet = [];
pdschReservedPRB.Period = [];

pdsch.ReservedPRB = {pdschReservedPRB};

% PDSCH DM-RS
pdschDMRS = nrPDSCHDMRSConfig;
pdschDMRS.DMRSConfigurationType = 1;
pdschDMRS.DMRSReferencePoint = 'CRB0';
pdschDMRS.DMRSTypeAPosition = 2;
pdschDMRS.DMRSAdditionalPosition = 0;
pdschDMRS.DMRSLength = 1;
pdschDMRS.CustomSymbolSet = [];
pdschDMRS.DMRSPortSet = [];
pdschDMRS.NIDNSCID = [];
pdschDMRS.NSCID = 0;
pdschDMRS.NumCDMGroupsWithoutData = 2;

pdsch.DMRS = pdschDMRS;

% PDSCH PT-RS
pdschPTRS = nrPDSCHPTRSConfig;
pdschPTRS.TimeDensity = 1;
pdschPTRS.FrequencyDensity = 2;
pdschPTRS.REOffset = '00';
pdschPTRS.PTRSPortSet = [];

pdsch.PTRS = pdschPTRS;

cfgDL.PDSCH = {pdsch};

%% CSI-RS Instances Configuration
csirs = nrWavegenCSIRSConfig;
csirs.Enable = false;
csirs.Label = 'CSIRS1';
csirs.Power = 0;
csirs.BandwidthPartID = 1;
csirs.CSIRSType = 'nzp';
csirs.CSIRSPeriod = 'on';
csirs.RowNumber = 3;
csirs.Density = 'one';
csirs.SymbolLocations = 0;
csirs.SubcarrierLocations = 0;
csirs.NumRB = 52;
csirs.RBOffset = 0;
csirs.NID = 0;

cfgDL.CSIRS = {csirs};

% Generation
[waveform,info] = nrWaveformGenerator(cfgDL);

Fs = info.ResourceGrids(1).Info.SampleRate; 								 % Specify the sample rate of the waveform in Hz

%% Visualize
% Spectrum Analyzer
spectrum = dsp.SpectrumAnalyzer('SampleRate', Fs);
spectrum(waveform);
release(spectrum);

% convert waveform to 32 bit integer 
waveform = round(waveform * 2^31);
fileHandle = fopen('30720KSPS_dl_signal.sigmf-data','w');
fwrite(fileHandle, reshape([real(waveform) imag(waveform)]',1,[]), 'int32');
fclose(fileHandle);


%%
%%
%% verify waveform with code from matlab examples
L_max = 8; % 4, 8 or 64
nrbSSB = 20;
scsSSB = 15;
nSlot = 0;
rxSampleRate = Fs;

peak_value = zeros(3,1);
peak_index = zeros(3,1);
pssIndices = (636-63):1:(636+63);
carrierConfig = nrCarrierConfig('NSizeGrid', 106);
for current_NID2 = [0 1 2]
    slotGrid = nrResourceGrid(carrierConfig, 1);
    slotGrid = slotGrid(:,1); % only 1 symbol
    slotGrid(pssIndices, 1) = nrPSS(current_NID2);
    refWaveform = nrOFDMModulate(carrierConfig, slotGrid);
    refWaveform = refWaveform(161:end); % remove CP

    temp = xcorr(waveform, refWaveform);
    corr = temp(size(waveform):end);
    corr = sum(abs(corr),2);
    [peak_value(current_NID2+1), peak_index(current_NID2+1)] = max(corr);
    subplot(3,1,current_NID2+1);
    plot(corr);
end
[x, detected_NID2] = max(peak_value);
detected_NID2 = detected_NID2-1;
assert(detected_NID2 == NID2)

%%
refGrid = zeros([nrbSSB*12 2]);
refGrid(nrPSSIndices,2) = nrPSS(NID2);
% Timing estimation. This is the timing offset to the OFDM symbol prior to
% the detected SSB due to the content of the reference grid
if false
    timingOffset = nrTimingEstimate(waveform,nrbSSB,scsSSB,nSlot,refGrid,'SampleRate',rxSampleRate);
else
    % use same SSB as in python code
    mu = 0;
    cp1 = round(5.2e-6*Fs * 2^(-mu));
    cp2 = round(4.7e-6*Fs * 2^(-mu));
    timingOffset = 35264 - 2048 - cp1 - cp2;
end

% Synchronization, OFDM demodulation, and extraction of strongest SS block
rxGrid = nrOFDMDemodulate(waveform(1+timingOffset:end,:),nrbSSB,scsSSB,nSlot,'SampleRate',rxSampleRate);
rxGrid = rxGrid(:,2:5,:);

% Extract the received SSS symbols from the SS/PBCH block
sssIndices = nrSSSIndices;
sssRx = nrExtractResources(sssIndices,rxGrid);

% Correlate received SSS symbols with each possible SSS sequence
sssEst = zeros(1,336);
for NID1 = 0:335
    ncellid = (3*NID1) + NID2;
    sssRef = nrSSS(ncellid);
    sssEst(NID1+1) = sum(abs(mean(sssRx .* conj(sssRef),1)).^2);
end
% Plot SSS correlations
figure;
stem(0:335,sssEst,'o');
title('SSS Correlations (Frequency Domain)');
xlabel('$N_{ID}^{(1)}$','Interpreter','latex');
ylabel('Magnitude');
axis([-1 336 0 max(sssEst)*1.1]);

% Determine NID1 by finding the strongest correlation
detected_NID1 = find(sssEst==max(sssEst)) - 1;

%%
detected_ncellid = (3*detected_NID1) + detected_NID2;
% Calculate PBCH DM-RS indices
dmrsIndices = nrPBCHDMRSIndices(detected_ncellid);

% Perform channel estimation using DM-RS symbols for each possible DM-RS
% sequence and estimate the SNR
dmrsEst = zeros(1,8);
for ibar_SSB = 0:7

    refGrid = zeros([240 4]);
    refGrid(dmrsIndices) = nrPBCHDMRS(detected_ncellid,ibar_SSB);
    [hest,nest] = nrChannelEstimate(rxGrid,refGrid,'AveragingWindow',[0 1]);
    dmrsEst(ibar_SSB+1) = 10*log10(mean(abs(hest(:).^2)) / nest);

end

% Plot PBCH DM-RS SNRs
figure;
stem(0:7,dmrsEst,'o');
title('PBCH DM-RS SNR Estimates');
xlabel('$\overline{i}_{SSB}$','Interpreter','latex');
xticks(0:7);
ylabel('Estimated SNR (dB)');
axis([-1 8 min(dmrsEst)-1 max(dmrsEst)+1]);

% Record ibar_SSB for the highest SNR
ibar_SSB = find(dmrsEst==max(dmrsEst)) - 1;

%%
refGrid = zeros([nrbSSB*12 4]);
refGrid(dmrsIndices) = nrPBCHDMRS(detected_ncellid,ibar_SSB);
refGrid(sssIndices) = nrSSS(detected_ncellid);
[hest,nest,hestInfo] = nrChannelEstimate(rxGrid,refGrid,'AveragingWindow',[0 1]);
%%
disp(' -- PBCH demodulation and BCH decoding -- ')

% Extract the received PBCH symbols from the SS/PBCH block
[pbchIndices,pbchIndicesInfo] = nrPBCHIndices(detected_ncellid);
pbchRx = nrExtractResources(pbchIndices,rxGrid);

% Configure 'v' for PBCH scrambling according to TS 38.211 Section 7.3.3.1
% 'v' is also the 2 LSBs of the SS/PBCH block index for L_max=4, or the 3
% LSBs for L_max=8 or 64.
if L_max == 4
    v = mod(ibar_SSB,4);
else
    v = ibar_SSB;
end
ssbIndex = v;

% PBCH equalization and CSI calculation
pbchHest = nrExtractResources(pbchIndices,hest);
[pbchEq,csi] = nrEqualizeMMSE(pbchRx,pbchHest,nest);
Qm = pbchIndicesInfo.G / pbchIndicesInfo.Gd;
csi = repmat(csi.',Qm,1);
csi = reshape(csi,[],1);

% Plot received PBCH constellation after equalization
figure;
plot(pbchEq,'o');
xlabel('In-Phase'); ylabel('Quadrature')
title('Equalized PBCH Constellation');
m = max(abs([real(pbchEq(:)); imag(pbchEq(:))])) * 1.1;
axis([-m m -m m]);

% PBCH demodulation, does also descrambling
pbchBits = nrPBCHDecode(pbchEq,detected_ncellid,v,nest);

% generate scrambling sequence just for debugging, TS38.11 7.3.3.1
E = 864; 
scrambling_seq = nrPBCHPRBS(detected_ncellid,v,E);

% Calculate RMS PBCH EVM
pbchRef = nrPBCH(pbchBits<0,detected_ncellid,v);
evm = comm.EVM;
pbchEVMrms = evm(pbchRef,pbchEq);

% Display calculated EVM
disp([' PBCH RMS EVM: ' num2str(pbchEVMrms,'%0.3f') '%']);

%%
% Apply CSI
pbchBits = pbchBits .* csi;

% Perform BCH decoding including rate recovery, polar decoding, and CRC
% decoding. PBCH descrambling and separation of the BCH transport block
% bits 'trblk' from 8 additional payload bits A...A+7 is also performed:
%   A ... A+3: 4 LSBs of system frame number
%         A+4: half frame number
% A+5 ... A+7: for L_max=64, 3 MSBs of the SS/PBCH block index
%              for L_max=4 or 8, A+5 is the MSB of subcarrier offset k_SSB
polarListLength = 8;
[scrblk,crcBCH,trblk,sfn4lsb,nHalfFrame,msbidxoffset] = ...
    nrBCHDecode(pbchBits,polarListLength,L_max,detected_ncellid);
%% 
% manual polar decoding and descrambling
if true
    iBIL = false;
    iIL = true;
    crcLen = 24;
    nMax = 9;
    A = 32;
    P = 24;
    K = A+P;
    N = get_3GPP_N(K,E,9);
    decIn = nrRateRecoverPolar(pbchBits,K,N,iBIL);
    decBits = nrPolarDecode(decIn,K,E,polarListLength,nMax,iIL,crcLen);
    %crc = decBits(end-23:end);
    [blk,err1] = nrCRCDecode(decBits,'24C');
    if err1
        disp(' BCH CRC is not zero.');
        return
    end
    if srcblk ~= blk(1:32)
        disp(' scrambled payload is not correct.');
        return
    end

    % descramble according to TS38.212 7.1.2
    if L_max == 4 || L_max == 8
        M = A-3;
    else
        M = A-6;
    end
    tmp_seq = nrPBCHPRBS(detected_ncellid,0,length(blk)*100);
    G = [16 23 18 17 8 30 10 6 24 7 0 5 3 2 1 4 9 11 12 13 14 15 19 20 21 22 25 26 27 28 29 31] + 1;
    PBCH_SFN_2ND_LSB_G = 10;
    PBCH_SFN_3RD_LSB_G = 9;
    v = 2*blk(PBCH_SFN_3RD_LSB_G) + blk(PBCH_SFN_2ND_LSB_G);
    n = v*M;
    scrambling_seq2 = tmp_seq(n+1:n+A);
    scrambling_seq3 = zeros(A,1);
    j = 1;
    for i = 1:A
        % SFN_2ND_LSB = 10, SFN_3RD_LSB = 9, half_frame_index= 11
        if i == G(11) || i == G(PBCH_SFN_2ND_LSB_G) || i == G(PBCH_SFN_3RD_LSB_G)  
            scrambling_seq3(i) = 0;
        else
            scrambling_seq3(i) = scrambling_seq2(j);
            j = j+1;
        end
    end
    tmp_seq2 = bitxor(blk, scrambling_seq3);
    if all(~tmp_seq2) || sum(tmp_seq2) < 10
        disp('manual descrambling works!')
    end

    tmp_seq3 = tmp_seq2(G);
    ber = comm.ErrorRate;
    errStats = ber(double(tmp_seq3(1:24)), double(trblk));
    disp([' Bit Error Rate: ' num2str(errStats(1))]);
end

%%
% Display the BCH CRC
disp([' BCH CRC: ' num2str(crcBCH)]);

% Stop processing MIB and SIB1 if BCH was received with errors
if crcBCH
    disp(' BCH CRC is not zero.');
    return
end

% Use 'msbidxoffset' value to set bits of 'k_SSB' or 'ssbIndex', depending
% on the number of SS/PBCH blocks in the burst
if (L_max==64)
    ssbIndex = ssbIndex + (bit2int(msbidxoffset,3) * 8);
    k_SSB = 0;
else
    k_SSB = msbidxoffset * 16;
end

% Displaying the SSB index
disp([' SSB index: ' num2str(ssbIndex)]);
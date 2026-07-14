function [HtruncReal,opt] = helperCSINetGenerateData(numSamples,channel,carrier,opt)
%helperCSINetGenerateData Generate 5G NR channel estimates
%   HEST = helperCSIGenerateData(N,CH,CR,OPT) generates N channel estimate
%   frames based on the 5G NR carrier, CR, and channel, CH. HEST is an
%   array with size [Ndelay Ntx Niq Nrx], where Ndelay is the maximum
%   delay, Ntx is the number of transmit antennas, Niq is 2 for in-phase
%   and quadrature components, and Nrx is the number of receive antennas.
%   OPT is a structure with various configuration parameters. Each file
%   contains one frame of channel estimates.

%   Copyright 2024-2025 The MathWorks, Inc.

% Calculate dependent parameters
release(channel)
channelInfo = info(channel);
if isa(channel,"nrCDLChannel")
  Nrx = channelInfo.NumOutputSignals; % Number of Rx antennas
  % Make sure that this is high enough for nrPerfectChannelEstimate to return
  % the full number of symbols worth of channel estimates
  opt.ChannelSampleDensity = 64*4;
else
  Nrx = channelInfo.NumReceiveAntennas;
end

waveInfo = nrOFDMInfo(carrier);
channel.SampleRate = waveInfo.SampleRate;

numSubCarriers = carrier.NSizeGrid*12; % 12 subcarriers per RB
Tdelay = 1/(numSubCarriers*carrier.SubcarrierSpacing*1e3);
opt.MaxDelay = round((channel.DelaySpread/Tdelay)*opt.TruncationFactor/2)*2;

opt.NumSlotsPerFrame = 1;
opt.Preprocess = true;
opt.ResetChannelPerFrame = true;
opt.Normalization = false;
opt.Verbose = true;

% Calculate the number of frames required to generate these training
% samples. Each slot has Nrx training channel estimates.
numSlots = ceil(numSamples/Nrx);
numFrames = ceil(numSlots/opt.NumSlotsPerFrame);

if ~helperCSIValidateDataFiles(numFrames,channel,carrier,opt)
  [~,HtruncReal] = helperCSIGenerateData(numFrames,channel,carrier,opt);
else
  disp("Data exists. Skipping generation.")
  disp("Loading data from file(s)...")
  % Create a signalDataStore object to access the data. The signal
  % datastore uses individual files for each data point.
  sds = signalDatastore( ...
    fullfile(opt.DataDir,"processed",opt.DataFilePrefix+"_*"));
  % Load data into memory
  HtruncRealCell = readall(sds);
  HtruncReal = cat(5,HtruncRealCell{:});
  disp("Data loaded.")
end

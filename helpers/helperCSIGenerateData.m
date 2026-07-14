function [Hest,Hestpp] = helperCSIGenerateData(numFrames,channel,carrier,opt)
%helperCSIGenerateData Generate 5G NR channel estimates
%   [HEST,Hestpp] = helperCSIGenerateData(N,CH,CR,OPT) generates N channel
%   estimate frames based on the 5G NR carrier, CR, and channel, CH. HEST
%   is an array with size [Nsc Nsym Nrx Ntx]. OPT is a structure with
%   various configuration parameters. Each file contains one channel
%   estimate.

%   Copyright 2024-2025 The MathWorks, Inc.

% Calculate dependent parameters
subcarrierPerRB = 12;
opt.Nsc = carrier.NSizeGrid*subcarrierPerRB;
channelInfo = info(channel);
if isa(channel,"nrCDLChannel")
  opt.Ntx = channelInfo.NumInputSignals;  % Number of Tx antennas
  opt.Nrx = channelInfo.NumOutputSignals; % Number of Rx antennas
  if isfield(opt,"ChannelSampleDensity")
    % Make sure that this is high enough for nrPerfectChannelEstimate to
    % return the full number of symbols worth of channel estimates
    channel.SampleDensity = opt.ChannelSampleDensity;
  else
    % If not specified, use default value
    channel.SampleDensity = 64;
  end
else
  opt.Ntx = channelInfo.NumTransmitAntennas;
  opt.Nrx = channelInfo.NumReceiveAntennas;
end

numSlotsPerFrame = opt.NumSlotsPerFrame;
% if numFrames == 1
%   % If only one frame is enough, adjust slots per frame
%   numSlotsPerFrame = numSlots;
% end

symbolsPerSlot = carrier.SymbolsPerSlot;
waveInfo = nrOFDMInfo(carrier);
opt.SamplesPerSlot = ...
  sum(waveInfo.SymbolLengths(1:symbolsPerSlot));
channel.SampleRate = waveInfo.SampleRate;

if opt.SaveData
  fileNamePrefix = opt.DataFilePrefix;
end

% Decide on number of parallel workers and frames per worker
numWorkers = 1;
if opt.UseParallel
  if exist('gcp','file')
    pool = gcp;
    if ~isempty(pool)
      numWorkers = pool.NumWorkers;
    end
  end
  maxNumWorkers = inf;
else
  maxNumWorkers = 0;
end

Nsc = opt.Nsc;
preprocess = opt.Preprocess;
if preprocess
  % Setup truncation factor and max delay
  Tdelay = 1/(Nsc*carrier.SubcarrierSpacing*1e3);
  rmsTauSamples = channel.DelaySpread/Tdelay;
  opt.MaxDelay = round((rmsTauSamples)*opt.TruncationFactor/2)*2;
end

if opt.SaveData
  chEstFileNameBase = fullfile(pwd,opt.DataDir,fileNamePrefix);
  if ~exist(fullfile(pwd,opt.DataDir),"file")
    mkdir(fullfile(pwd,opt.DataDir))
  end
  if opt.Preprocess
    processedFileNameBase = ...
      fullfile(pwd,opt.DataDir,"processed",fileNamePrefix+"_processed");
    if ~exist(fullfile(pwd,opt.DataDir,"processed"),"file")
      mkdir(fullfile(pwd,opt.DataDir,"processed"))
    end
  else
    processedFileNameBase = [];
  end
else
  chEstFileNameBase = [];
  processedFileNameBase = [];
end

Ntx = opt.Ntx;
Nrx = opt.Nrx;
if preprocess
  isFrequencySpatial = strcmp(opt.DataDomain,"Frequency-Spatial");
  Hestpp = zeros(opt.MaxDelay,Ntx,2,Nrx,numSlotsPerFrame*numFrames,"single");
else
  Hestpp = [];
  isFrequencySpatial = [];
end

saveData = opt.SaveData;
Hest = zeros(Nsc,numSlotsPerFrame*symbolsPerSlot,Nrx,Ntx,numFrames,"single");
if opt.Verbose
  disp("Starting CSI data generation")
  disp(numWorkers + " worker(s) running")
end
tStart = tic;
dataQ = parallel.pool.DataQueue;
%for p=1:numFrames
parfor (p=1:numFrames,maxNumWorkers)
  channelLocal = clone(channel);
  % Channel estimate
  H = helperCSIChannelEstimate(numSlotsPerFrame,channelLocal,carrier,opt);
  Hest(:,:,:,:,p) = H;
  if saveData
    saveDataToFile(chEstFileNameBase+"_"+p,H);
  end

  if preprocess
    % Preprocessed channel estimate
    Hpp = ...
      helperCSIPreprocessChannelEstimate(H,opt);
    Hestpp(:,:,:,:,p) = Hpp;
    if saveData
      saveDataToFile(processedFileNameBase+"_"+p,single(Hpp));
    end
  end

  send(dataQ,1)
  if opt.Verbose && ~mod(p,100)
    numCompleted = dataQ.QueueLength;
    t = seconds(toc(tStart));
    t.Format = "hh:mm:ss";
    fprintf("%s - %2.0f%% Completed\n",t,100*numCompleted/numFrames)
  end
end
if opt.Verbose
  t = seconds(toc(tStart));
  t.Format = "hh:mm:ss";
  fprintf("%s - %2.0f%% Completed\n",t,100)
end

if saveData
  dataFiles = dir(chEstFileNameBase+"*.mat");
  save(fullfile(opt.DataDir,"info"),'carrier','channel','dataFiles')
end
end

function saveDataToFile(fileName,H)
%saveDataToFile Save data to file
%   saveDataToFile(FNAME,H) saves the channel estimate, H, in file FNAME. 
%   Save function must be outside of the parfor-loop.
save(fileName,'H')
end

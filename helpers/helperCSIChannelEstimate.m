function estChannelGrid = helperCSIChannelEstimate(slotsPerFrame,channel,carrier,opt)
%helperCSIChannelEstimate Channel estimate
%   CHEST = helperCSIChannelEstimate(N,CR,CH) generates one frame of
%   perfect channel estimate that contains N slots per frame using CH
%   channel object. CR is the 5G NR carrier parameters object.

%   Copyright 2024-2025 The MathWorks, Inc.

symPerSlot = carrier.SymbolsPerSlot;      % Number OFDM symbols per slot
if slotsPerFrame < carrier.SlotsPerFrame
  channel.NumTimeSamples = opt.SamplesPerSlot*slotsPerFrame;
  estChannelGrid = chanEst(channel,carrier,opt.ZeroTimingOffset);
else
  channel.NumTimeSamples = opt.SamplesPerSlot*carrier.SlotsPerFrame;

  totalNumSyms = symPerSlot*ceil(slotsPerFrame/carrier.SlotsPerFrame)*carrier.SlotsPerFrame;
  estChannelGrid = complex(zeros(opt.Nsc,totalNumSyms,opt.Nrx,opt.Ntx));
  for subFrame = 0:ceil(slotsPerFrame/carrier.SlotsPerFrame)-1
    % Perfect channel estimate
    Hest = chanEst(channel,carrier,opt.ZeroTimingOffset);
    idx=(subFrame*carrier.SlotsPerFrame)*carrier.SymbolsPerSlot+1:(subFrame+1)*carrier.SlotsPerFrame*carrier.SymbolsPerSlot;
    estChannelGrid(:,idx,:,:) = Hest;
  end
end

if opt.ResetChannelPerFrame
  reset(channel)
end
end

function hEst = chanEst(channel,carrier,zeroTimingOffset)
[pathGains, sampleTimes] = channel();

pathFilters = getPathFilters(channel);
if zeroTimingOffset
  % Perfect timing sync
  offset = 0;
else
  offset = nrPerfectTimingEstimate(pathGains, pathFilters);
end

% Perfect channel estimate
hEst = nrPerfectChannelEstimate(carrier, pathGains, ...
  pathFilters, offset, ...
  sampleTimes);
end

%Gerar dataset para o treino do autoencoder
clear; clc; close all;

nSizeGrid = 52;
autoEncOpt.SubcarrierSpacing = 15;

%Criar um objeto nrCarrierConfig para configurar os parâmetros da carrier

carrier = nrCarrierConfig;
autoEncOpt.NSizeGrid = nSizeGrid;
carrier.SubcarrierSpacing = autoEncOpt.SubcarrierSpacing;
waveInfo = nrOFDMInfo(carrier);

%Configurar ocanal MIMO

autoEncOpt.TxAntennaSize = [2 2 2 1 1];   % rows, columns, polarization, panels
%autoEncOpt.TxAntennaSize = [8 2 2 1 1];   % rows, columns, polarization, panels
autoEncOpt.RxAntennaSize = [2 1 2 1 1];   % rows, columns, polarization, panels
%autoEncOpt.RxAntennaSize = [4 1 2 1 1];   % rows, columns, polarization, panels
autoEncOpt.MaxDoppler = 5;                % Hz
autoEncOpt.RMSDelaySpread = 300e-9;       % s
numSubCarriers = carrier.NSizeGrid*12; % 12 subcarriers per RB

%Delay profile

autoEncOpt.DelayProfile = "CDL-C"; % CDL-A, CDL-B, CDL-C, CDL-D, CDL-D, CDL-E

%Criar o objeto nrCDLchannel e configurar os parâmetros
samplesPerSlot = ...
  sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

channel = nrCDLChannel;
channel.DelayProfile = autoEncOpt.DelayProfile;
channel.DelaySpread = autoEncOpt.RMSDelaySpread;     % s
channel.MaximumDopplerShift = autoEncOpt.MaxDoppler; % Hz
channel.RandomStream = "Global stream";
channel.TransmitAntennaArray.Size = autoEncOpt.TxAntennaSize;
channel.ReceiveAntennaArray.Size = autoEncOpt.RxAntennaSize;
channel.ChannelFiltering = false;        % No filtering for 
                                         % perfect estimate
channel.NumTimeSamples = samplesPerSlot; % 1 slot worth of samples
channel.SampleRate = waveInfo.SampleRate;


%Simular ocanal
[pathGains,sampleTimes] = channel();
pathFilters = getPathFilters(channel);


autoEncOpt.ZeroTimingOffset = true;
if autoEncOpt.ZeroTimingOffset
  % Perfect timing sync
  offset = 0;
else
  offset = nrPerfectTimingEstimate(pathGains, pathFilters);
end
Hest = nrPerfectChannelEstimate(carrier, pathGains, ...
                                pathFilters, offset, ...
                                sampleTimes);
reset(channel);


% Get dimensions of Channel estimate
[nSub,nS,nRx,nTx] = size(Hest);

helperPlotChannelResponse(Hest);

Hmean = squeeze(mean(Hest,2));

Hmean = permute(Hmean,[1 3 2]);

Hdft2 = fft2(Hmean);


Tdelay = 1/(numSubCarriers*carrier.SubcarrierSpacing*1e3);
rmsTauSamples = channel.DelaySpread/Tdelay;
maxTruncationFactor = floor(numSubCarriers/rmsTauSamples);


autoEncOpt.TruncationFactor = 10;
autoEncOpt.MaxDelay = round((channel.DelaySpread/Tdelay)*autoEncOpt.TruncationFactor/2)*2;

midPoint = floor(nSub/2);
lowerEdge = midPoint - (nSub-autoEncOpt.MaxDelay)/2 + 1;
upperEdge = midPoint + (nSub-autoEncOpt.MaxDelay)/2;
Htemp = Hdft2([1:lowerEdge-1 upperEdge+1:end],:,:);


autoEncOpt.DataDomain = "Frequency-Spatial";
switch autoEncOpt.DataDomain
    case "Delay-Angle"
        Htrunc = Htemp;
    case "Frequency-Spatial"
        Htrunc = ifft2(Htemp);
end
HtruncReal = cat(3, real(Htrunc), imag(Htrunc));
size(HtruncReal)


helperPlotCSIFeedbackPreprocessingSteps(Hmean(:,:,1), ...
                                        Hdft2(:,:,1), Htemp(:,:,1), ...
                                        Htrunc(:,:,1), nSub, ...
                                        nTx, ...
                                        autoEncOpt.MaxDelay, ...
                                        autoEncOpt.DataDomain);


numSamples = 20000;


%autoEncOpt.UseParallel = false;
autoEncOpt.UseParallel = true;

autoEncOpt.SaveData = true;
autoEncOpt.DataDir = "DataBig8Tx_New_20mil";
autoEncOpt.DataFilePrefix = "CH_est";




HtruncReal = helperCSINetGenerateData(numSamples,channel,carrier,autoEncOpt);

[maxDelay,nTx,Niq,nRx,Nframes] = size(HtruncReal);


HtruncReal = reshape(HtruncReal,maxDelay,nTx,Niq,nRx*Nframes);
[maxDelay,nTx,Niq,Nsamples] = size(HtruncReal);


figure
subplot(1,2,1)
imagesc(HtruncReal(:,:,1,1,1))
xlabel("Transmit Antennas")
ylabel("Compressed Subcarriers")
title("In-phase")
subplot(1,2,2)
imagesc(HtruncReal(:,:,2,1,1))
xlabel("Transmit Antennas")
ylabel("Compressed Subcarriers")
title("Quadrature")




meanVal = mean(HtruncReal,"all");
stdVal = std(HtruncReal,[],"all");

%Separar dados de treino, validação e teste

N = size(HtruncReal, 4);
numTrain = floor(N*10/15); %66.67% dos dados
numVal = floor(N*3/15); % 20% dos dados
numTest = floor(N*2/15); %13% dos dados

%Normalize the data 


targetStd = 1;
HTReal = (HtruncReal(:,:,:,1:numTrain)-meanVal) ...
   /stdVal;
HVReal = (HtruncReal(:,:,:,numTrain+(1:numVal))-meanVal) ...
   /stdVal;
HTestReal = (HtruncReal(:,:,:,numTrain+numVal+(1:numTest))-meanVal) ...
   /stdVal;
 autoEncOpt.MeanVal = meanVal;
 autoEncOpt.StdValue = stdVal*1.5;
 %autoEncOpt.StdValue = stdVal;
 autoEncOpt.TargetSTDValue = targetStd;

autoEncOpt.MaxDelay = maxDelay;
autoEncOpt.NumTx = nTx;
autoEncOpt.Niq = Niq;


%Salvar o dataset com a divisão e dados normalizados
save("CSI_DatasetNovoBig8Tx_20mil.mat","HTReal","HVReal","HTestReal","autoEncOpt","-v7.3");



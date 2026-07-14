function HppReal = helperCSIPreprocessChannelEstimate(Hest,opt)
%helperCSIPreprocessChannelEstimate Preprocess channel estimate
%   HPPREAL = helperCSIPreprocessChannelEstimate(HEST,MAXDELAY,FREQTX)
%   preprocesses the channel estimate, HEST, by decimating it using 2D-FFT
%   and 2D-IFFT. The resulting channel estimate has MAXDELAY samples. If
%   FREQTX is true, the samples are in the frequency-spatial domain. If
%   FREQTX is false, the samples are in the delay-angle domain.
%
%   HEST is the channel estimate array with dimensions
%   Nsc-by-Nsym-Ntx-by-Nrx, where Nsc is number of subcarriers, Nsym is the
%   number of symbols (must be an integer number of slots), Ntx is the
%   number of transmit antennas, and Nrx is the number of receive antennas.
%
%   Preprocessed channel estimate, HPPREAL, is a real valued array with
%   dimensions MAXDELAY-by-Ntx-by-Niq-by-Nrx-by-Nslot, where Niq is 2
%   (in-phase and quadrature values) and Nslot is the number of slots
%   (Nsym/14).

%   Copyright 2024-2025 The MathWorks, Inc.

maxDelay = opt.MaxDelay;

[Nsc,Nsym,Nrx,Ntx] = size(Hest);
midPoint = floor(Nsc/2);
lowerEdge = midPoint - (Nsc-maxDelay)/2 + 1;
upperEdge = midPoint + (Nsc-maxDelay)/2;

% Average over symbols (one slot)
symbolsPerSlot = 14;
Nslot = Nsym/symbolsPerSlot;
H = reshape(Hest,Nsc,Nslot,symbolsPerSlot,Nrx,Ntx);
H = mean(H,3);
H = permute(H, [1 5 4 2 3]);

% Decimate over subcarriers using 2-D FFT for each Rx antenna
HppReal = zeros(maxDelay,Ntx,2,Nrx,Nslot,'like',real(H(1)));
for slot=1:Nslot
  Hdft2 = fft2(H(:,:,:,slot));
  Htemp = Hdft2([1:lowerEdge-1 upperEdge+1:end],:,:);
  if strcmp(opt.DataDomain,"Frequency-Spatial")
    Htrunc = ifft2(Htemp);
  else
    Htrunc = Htemp;
  end
  if opt.Normalization
    meanVal = opt.MeanVal;
    stdVal = opt.StdValue;
    targetSTD = opt.TargetSTDValue;
    HppReal(:,:,1,:,slot) = (real(Htrunc) - meanVal) / stdVal * targetSTD + 0.5;
    HppReal(:,:,2,:,slot) = (imag(Htrunc) - meanVal) / stdVal * targetSTD + 0.5;
  else
    HppReal(:,:,1,:,slot) = real(Htrunc);
    HppReal(:,:,2,:,slot) = imag(Htrunc);
  end
end
end
% function codeword = helperCSINetEncode(encNet, Hest, opt)
% %helperCSINetEncode Compress channel estimates with encoder neural network
% %   CW = helperCSINetEncode(ENC,HEST,OPT) compresses channel estimates,
% %   HEST, using the encoder neural network, ENC. OPT is the autoencoder
% %   options structure.
% %
% %   See also CSICompressionAutoencoderExample, helperCSINetDecode.
% 
% %   Copyright 2022-2023 The MathWorks, Inc.
% 
% arguments
%     encNet (1,1) dlnetwork
%     Hest (:,:,:,:,:) {double, single}
%     opt (1,1) struct
% end
% 
% [nSub,nSym,nRx,nTx,N] = size(Hest);
% 
% assert(nSub == opt.NumSubcarriers, ...
%     sprintf("Number of subcarriers (%d) in Hest does not match autoencoder's expected number of subcarriers (%d).",nSub,opt.NumSubcarriers))
% assert(nSym == opt.NumSymbols, ...
%     sprintf("Number of symbols (%d) in Hest does not match autoencoder's expected number of symbols (%d).",nSym,opt.NumSymbols))
% assert(nTx == opt.NumTxAntennas, ...
%     sprintf("Number of Tx antennas (%d) in Hest does not match autoencoder's expected number of Tx antennas (%d).",nTx,opt.NumTxAntennas))
% 
% dummy = predict(encNet,zeros(opt.MaxDelay,nTx,2));
% codeword = zeros(nRx,N,length(dummy),'single');
% for n=1:N
%   HtruncReal = helperCSIPreprocessChannelEstimate(Hest(:,:,:,:,n),opt);
%   codeword(:,n,:) = predict(encNet,HtruncReal);
% end
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%____________________________________________________________%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function codeword = helperCSINetEncode(encNet, Hest, opt, nVar)
%helperCSINetEncode Compress channel estimates with encoder neural network
%   CW = helperCSINetEncode(ENC,HEST,OPT) compresses channel estimates,
%   HEST, using the encoder neural network, ENC. OPT is the autoencoder
%   options structure.
%
%   CW = helperCSINetEncode(ENC,HEST,OPT,NVAR) additionally uses the noise
%   variance NVAR to compute the estimated SNR and embed it as a third
%   input channel when the network was trained with SNR conditioning
%   (opt.SNRConditioned = true).
%
%   See also CSICompressionAutoencoderExample, helperCSINetDecode.

%   Copyright 2022-2023 The MathWorks, Inc.
%   Modified: SNR-conditioned input support

arguments
    encNet (1,1) dlnetwork
    Hest (:,:,:,:,:) {double, single}
    opt (1,1) struct
    nVar double = []                    % <<< NOVO argumento opcional
end

[nSub,nSym,nRx,nTx,N] = size(Hest);

assert(nSub == opt.NumSubcarriers, ...
    sprintf("Number of subcarriers (%d) in Hest does not match autoencoder's expected number of subcarriers (%d).",nSub,opt.NumSubcarriers))
assert(nSym == opt.NumSymbols, ...
    sprintf("Number of symbols (%d) in Hest does not match autoencoder's expected number of symbols (%d).",nSym,opt.NumSymbols))
assert(nTx == opt.NumTxAntennas, ...
    sprintf("Number of Tx antennas (%d) in Hest does not match autoencoder's expected number of Tx antennas (%d).",nTx,opt.NumTxAntennas))

% =====================================================================
%  SNR CONDITIONING: Determinar se a rede espera 3 canais de entrada
% =====================================================================
snrConditioned = isfield(opt, 'SNRConditioned') && opt.SNRConditioned;

if snrConditioned
    % Estimar SNR a partir do canal e da variância de ruído
    if ~isempty(nVar) && nVar > 0
        signalPower = mean(abs(Hest(:)).^2);
        snrEstimate_dB = 10 * log10(signalPower / nVar);
    else
        % Fallback: se não tiver estimativa de ruído, assumir SNR alto
        snrEstimate_dB = 30;
    end

    % Normalizar com o mesmo fator usado no treino
    snrNormFactor = opt.SNRNormFactor;  % Salvo no .mat da rede (tipicamente 40)

    % Clipar para o range de treino (evitar extrapolação extrema)
    snrNorm = max(-5/snrNormFactor, min(1.0, snrEstimate_dB / snrNormFactor));

    % Dummy para obter tamanho da saída (agora com 3 canais)
    dummy = predict(encNet, zeros(opt.MaxDelay, nTx, 3));
else
    % Comportamento original (2 canais)
    dummy = predict(encNet, zeros(opt.MaxDelay, nTx, 2));
end

codeword = zeros(nRx, N, length(dummy), 'single');

for n = 1:N
    % Pré-processamento original: [maxDelay x nTx x 2 x nRx]
    HtruncReal = helperCSIPreprocessChannelEstimate(Hest(:,:,:,:,n), opt);

    if snrConditioned
        % Adicionar 3º canal (mapa constante de SNR normalizado)
        % HtruncReal tem shape [maxDelay x nTx x 2 x nRx]
        % Precisamos iterar por antena Rx como o código original faz
        [d1, d2, ~, nRxLocal] = size(HtruncReal);
        snrChannel = snrNorm * ones(d1, d2, 1, nRxLocal, 'like', HtruncReal);
        HtruncReal = cat(3, HtruncReal, snrChannel);  % [maxDelay x nTx x 3 x nRx]
    end

    codeword(:,n,:) = predict(encNet, HtruncReal);
end

end






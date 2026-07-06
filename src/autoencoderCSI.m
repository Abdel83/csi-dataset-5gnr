clear; clc; close all;


%% =====================================================================
%  CARREGAR DADOS
%  =====================================================================

%load("CSI_DatasetNovoBig8Tx_20mil.mat");     % 20mil amostras
load("CSI_DatasetNovoBig32Tx_20mil.mat");     % 20mil amostras




[maxDelay, nTx, Niq, Nsamples] = size(HTReal);
numTest = size(HTestReal, 4);

autoEncOpt = struct();
autoEncOpt.UseParallel = true;

meanV = mean(HTReal, 'all');
stdV  = std(HTReal, 0, 'all');

% Normalizar para média 0 e desvio 1
HTClean    = (HTReal    - meanV) / stdV;
HVClean    = (HVReal    - meanV) / stdV;
HTestClean = (HTestReal - meanV) / stdV;

%% =====================================================================
%  FAIXA DE SNR PARA TREINO
%  =====================================================================

minSNR = -5;
maxSNR = 40;

% Valor para normalizar SNR para a rede (mapeia para ~[0, 1])
snrNormFactor = maxSNR;  

%% =====================================================================
%  PREPARAR DADOS DE TREINO RUIDOSOS
%  =====================================================================

numTrain = size(HTClean, 4);


 % Entrada terá 3 canais: [real, imag, SNR_map]
 HTNoisy    = zeros(maxDelay, nTx, 3, numTrain);
 SNR_values_train = zeros(1, numTrain);  % guardar para loss ponderada (C4)

    
rng(42);

for i = 1:numTrain
    
    currentSNR = minSNR + (maxSNR - minSNR) * rand();
    
    SNR_values_train(i) = currentSNR;
    
    % Gerar ruído
    noisePower = 10^(-currentSNR / 10);
    noiseStd   = sqrt(noisePower / 2);
    noiseBlock = randn(maxDelay, nTx, 2) * noiseStd;
    
    % Canais 1-2: dados + ruído
    HTNoisy(:,:,1:2,i) = HTClean(:,:,:,i) + noiseBlock;
    
    % Canal 3: mapa constante com SNR normalizado
    snrNorm = currentSNR / snrNormFactor;
    HTNoisy(:,:,3,i) = snrNorm * ones(maxDelay, nTx);
  
end

%% =====================================================================
%  PREPARAR DADOS DE VALIDAÇÃO RUIDOSOS
%  =====================================================================

numVal = size(HVClean, 4);

HVNoisy    = zeros(maxDelay, nTx, 3, numVal);
SNR_values_val = zeros(1, numVal);


for i = 1:numVal
  
    currentSNR = minSNR + (maxSNR - minSNR) * rand();
   
    SNR_values_val(i) = currentSNR;
    
    noisePower = 10^(-currentSNR / 10);
    noiseStd   = sqrt(noisePower / 2);
    noiseBlock = randn(maxDelay, nTx, 2) * noiseStd;
    
    HVNoisy(:,:,1:2,i) = HVClean(:,:,:,i) + noiseBlock;

    snrNorm = currentSNR / snrNormFactor;
    HVNoisy(:,:,3,i) = snrNorm * ones(maxDelay, nTx);
    
end

%% =====================================================================
%  CONFIGURAR autoEncOpt
%  =====================================================================

if exist('numSubCarriers', 'var')
    autoEncOpt.NumSubcarriers = numSubCarriers;
elseif ~isfield(autoEncOpt, 'NumSubcarriers')
    warning('NumSubcarriers não encontrado. Usando padrão 624.');
    autoEncOpt.NumSubcarriers = 624;
end

if ~exist('numSymbols', 'var')
    numSymbols = 14;
end
autoEncOpt.NumSymbols    = numSymbols;
autoEncOpt.NumTxAntennas = nTx;
autoEncOpt.maxDelay      = maxDelay;
autoEncOpt.MaxDelay      = maxDelay;
autoEncOpt.DataDomain    = "Frequency-Spatial";
autoEncOpt.Normalization = true;
autoEncOpt.MeanVal       = meanV;
autoEncOpt.StdValue      = stdV * 1.5;  
autoEncOpt.TargetSTDValue = 1;
autoEncOpt.SNRNormFactor   = snrNormFactor;
autoEncOpt.InputChannels   = 3; 

fprintf('Entrada da rede: [%d x %d x %d]\n', maxDelay, nTx, autoEncOpt.InputChannels);

%% =====================================================================
%  ARQUITETURA DA REDE
%  =====================================================================

nInputCh = autoEncOpt.InputChannels;  %  3 (SNR-conditioned)
inputSize = [maxDelay, nTx, nInputCh];
nLinear   = maxDelay * nTx * 2;       % Saída sempre [maxDelay x nTx x 2]
nEncoded  = 64;

lgraph = layerGraph();

% ==========================================================
%  ENCODER
% ==========================================================

lgraph = addLayers(lgraph, imageInputLayer(inputSize, ...
    "Normalization","none","Name","Enc_Input"));

% Conv Inicial: aceita nInputCh canais (2 ou 3)
lgraph = addLayers(lgraph, [
    convolution2dLayer([3 3], 16, "Padding","same","Name","Enc_Conv")
    batchNormalizationLayer("Epsilon",0.001,"Name","Enc_BN")
    leakyReluLayer(0.3,"Name","Enc_leakyRelu")
]);
lgraph = connectLayers(lgraph,"Enc_Input","Enc_Conv");

% Bloco Residual 1 (16 filtros)
lgraph = addLayers(lgraph, [
    convolution2dLayer([3 3],16,"Padding","same","Name","Enc_Res1_Conv1")
    batchNormalizationLayer("Epsilon",0.001,"Name","Enc_Res1_BN1")
    leakyReluLayer(0.3,"Name","Enc_Res1_lRelu1")
    convolution2dLayer([3 3],16,"Padding","same","Name","Enc_Res1_Conv2")
    batchNormalizationLayer("Epsilon",0.001,"Name","Enc_Res1_BN2")
]);
lgraph = addLayers(lgraph, additionLayer(2,"Name","Enc_Res1_Add"));
lgraph = addLayers(lgraph, leakyReluLayer(0.3,"Name","Enc_Res1_Out"));

lgraph = connectLayers(lgraph,"Enc_leakyRelu","Enc_Res1_Conv1");
lgraph = connectLayers(lgraph,"Enc_Res1_BN2","Enc_Res1_Add/in1");
lgraph = connectLayers(lgraph,"Enc_leakyRelu","Enc_Res1_Add/in2");
lgraph = connectLayers(lgraph,"Enc_Res1_Add","Enc_Res1_Out");

% Conv transição 16 -> 32
lgraph = addLayers(lgraph, [
    convolution2dLayer([3 3],32,"Padding","same","Name","Enc_Conv2")
    batchNormalizationLayer("Epsilon",0.001,"Name","Enc_BN2")
    leakyReluLayer(0.3,"Name","Enc_leakyRelu2")
]);
lgraph = connectLayers(lgraph,"Enc_Res1_Out","Enc_Conv2");

% Bloco Residual 2 (32 filtros, multi-scale)
lgraph = addLayers(lgraph, [
    convolution2dLayer([1 1],16,"Padding","same","Name","Enc_Res2_Conv1x1")
    batchNormalizationLayer("Epsilon",0.001,"Name","Enc_Res2_BN1")
    leakyReluLayer(0.3,"Name","Enc_Res2_lRelu1")
    convolution2dLayer([3 3],32,"Padding","same","Name","Enc_Res2_Conv3x3")
    batchNormalizationLayer("Epsilon",0.001,"Name","Enc_Res2_BN2")
]);
lgraph = addLayers(lgraph, additionLayer(2,"Name","Enc_Res2_Add"));
lgraph = addLayers(lgraph, leakyReluLayer(0.3,"Name","Enc_Res2_Out"));

lgraph = connectLayers(lgraph,"Enc_leakyRelu2","Enc_Res2_Conv1x1");
lgraph = connectLayers(lgraph,"Enc_Res2_BN2","Enc_Res2_Add/in1");
lgraph = connectLayers(lgraph,"Enc_leakyRelu2","Enc_Res2_Add/in2");
lgraph = connectLayers(lgraph,"Enc_Res2_Add","Enc_Res2_Out");

% Gargalo: Flatten -> FC -> BN
lgraph = addLayers(lgraph, [
    flattenLayer("Name","Enc_flatten")
    fullyConnectedLayer(nEncoded,"Name","Enc_FC")
    batchNormalizationLayer("Name","Enc_Sigmoid")
]);
lgraph = connectLayers(lgraph,"Enc_Res2_Out","Enc_flatten");

% ==========================================================
%  DECODER
% ==========================================================

lgraph = addLayers(lgraph, [
    fullyConnectedLayer(nLinear,"Name","Dec_FC")
    functionLayer(@(x)dlarray(reshape(x,maxDelay,nTx,2,[]),'SSCB'), ...
            "Formattable",true,"Acceleratable",true,"Name","Dec_Reshape")
    ]);
lgraph = connectLayers(lgraph,"Enc_Sigmoid","Dec_FC");
    
autoencoderNet = dlnetwork(lgraph);
autoencoderNet = helperCSINetAddResidualLayers(autoencoderNet, "Dec_Reshape");
    
autoencoderNet = addLayers(autoencoderNet, ...
    convolution2dLayer([3 3],2,"Padding","same","Name","Dec_Conv"));
autoencoderNet = connectLayers(autoencoderNet,"leakyRelu_2_3","Dec_Conv");


% Visualizar
figure
plot(autoencoderNet)
title('CSI Compression Autoencoder (SNR-Conditioned)')
analyzeNetwork(autoencoderNet)

%% =====================================================================
%  OPÇÕES DE TREINO
%  =====================================================================

miniBatchSize = 128;

options = trainingOptions("adam", ...
    InitialLearnRate = 3e-4, ...
    LearnRateSchedule = "piecewise", ...
    LearnRateDropPeriod = 40, ...
    LearnRateDropFactor = 0.5, ...
    Epsilon = 1e-7, ...
    MaxEpochs = 500, ...%200,500
    MiniBatchSize = miniBatchSize, ...
    L2Regularization = 1e-5, ...
    Shuffle = "every-epoch", ...
    ValidationData = {HVNoisy, HVClean}, ...
    ValidationFrequency = 100, ...
    Metrics = "rmse", ...
    Verbose = true, ...
    ValidationPatience = Inf, ...
    OutputNetwork = "best-validation-loss", ...
    ExecutionEnvironment = "auto", ...
    Plots = 'training-progress');

%% =====================================================================
%  LOSS FUNCTION
%  =====================================================================

lossFunc = @(x, t) nmseLossdB(x, t);


%% =====================================================================
%  TREINO
%  =====================================================================

[net, trainInfo] = trainnet(HTNoisy, HTClean, autoencoderNet, lossFunc, options);

% Salvar
savedOptions = options;
savedOptions.ValidationData = [];


save("dCSITrainedNetwork_" + "_" ...
    + string(datetime("now","Format","dd_MM_HH_mm")), ...
    'net','trainInfo','autoEncOpt','savedOptions');

%% =====================================================================
%  AVALIAÇÃO POR FAIXA DE SNR
%  =====================================================================
%  Em vez de testar só em um SNR, avaliar em múltiplos SNRs
%  para visualizar exatamente onde o modelo melhora/piora

testSNRs = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40];

fprintf('\n========== AVALIAÇÃO POR FAIXA DE SNR ==========\n');
fprintf('%6s | %12s | %12s\n', 'SNR', 'Mean rho', 'Mean NMSE');
fprintf('-------|--------------|-------------\n');

rho_per_snr  = zeros(length(testSNRs), 1);
nmse_per_snr = zeros(length(testSNRs), 1);

for s = 1:length(testSNRs)
    TestSNR = testSNRs(s);
    TestNoisePow = 10^(-TestSNR / 10);
    
    % Gerar entrada ruidosa de teste
    HTestNoisy_raw = HTestClean + randn(size(HTestClean)) * sqrt(TestNoisePow / 2);
    
    
    % Adicionar 3º canal com SNR normalizado
    snrMap = (TestSNR / snrNormFactor) * ones(maxDelay, nTx, 1, numTest);
    HTestNoisy_input = cat(3, HTestNoisy_raw, snrMap);
    
    
    % Predição
    HTestRealHat_Norm = predict(net, HTestNoisy_input);
    HTestRealHat = (HTestRealHat_Norm * stdV) + meanV;
    
    rho  = zeros(numTest, 1);
    nmse = zeros(numTest, 1);
    
    for n = 1:numTest
        in  = HTestReal(:,:,1,n) + 1i * HTestReal(:,:,2,n);
        out = HTestRealHat(:,:,1,n) + 1i * HTestRealHat(:,:,2,n);
        
        n1 = sqrt(sum(conj(in) .* in, 'all'));
        n2 = sqrt(sum(conj(out) .* out, 'all'));
        aa = abs(sum(conj(in) .* out, 'all'));
        rho(n) = aa / (n1 * n2);
        
        mse_val = mean(abs(in - out).^2, 'all');
        nmse(n) = 10 * log10(mse_val / mean(abs(in).^2, 'all'));
    end
    
    rho_per_snr(s)  = mean(rho);
    nmse_per_snr(s) = mean(nmse);
    
    fprintf('%4ddB | %12.5f | %10.2f dB\n', TestSNR, mean(rho), mean(nmse));
end

% Gráfico de desempenho por SNR
figure
tiledlayout(2,1)
nexttile
plot(testSNRs, rho_per_snr, 'o-', 'LineWidth', 1.5)
grid on; xlabel('SNR (dB)'); ylabel('\rho');
title('Correlação Média por SNR')

nexttile
plot(testSNRs, nmse_per_snr, 'o-', 'LineWidth', 1.5)
grid on; xlabel('SNR (dB)'); ylabel('NMSE (dB)');
title('NMSE Médio por SNR')


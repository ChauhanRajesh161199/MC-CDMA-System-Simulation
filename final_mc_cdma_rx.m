clc;
clear;
close all;

%% System Parameters
noOfUsers = 4;
noOfInfoBitsPerUser = 10000;

% OFDM & Physical Layer
Nfft = 256;
cpLength = 16;
guardLeft  = 8;
guardRight = 8;
usedSubcarriers = Nfft - guardLeft - guardRight;
spreadingFactor = 4;
noOfOfdmSymbolsPerFrame = 4;
bitspersymbol = 2; % QPSK

% Channel Coder
ConstraintLength = 3;
GeneratorPolynomials = [7 5];
codeRate = 1/length(GeneratorPolynomials);
trellis = poly2trellis(ConstraintLength, GeneratorPolynomials);

% Frame Structure & Synchronization
agcField = repmat([1 -1],1,50);
agcSamLen = length(agcField);
barker11 = [1  1  1 -1 -1 -1 1 -1 -1  1 -1]; 
barkerSync = repmat(barker11,1,4);    
barkerSyncLen = length(barkerSync);
trainingSeq = repmat([1 -1],1,50);
trainingLen = length(trainingSeq);
frameGuard=zeros(1,30);
frameGuardLen = length(frameGuard);
% Total no of QPSK symbols per frame
numOfQpskSymbolsPerFrame = agcSamLen+trainingLen+barkerSyncLen+ ...
                            (Nfft+cpLength)*noOfOfdmSymbolsPerFrame+frameGuardLen; 

% AGC Settings
targetPower = 1;      
tolerance   = 1e-3;   
maxIter     = 20;

% Derived Capacity Variables
qpskSymbolsPerOfdmSymbol = usedSubcarriers / spreadingFactor;
infoBitsPerOfdmSymbols = qpskSymbolsPerOfdmSymbol * bitspersymbol * codeRate;
infoBitsPerFrame = infoBitsPerOfdmSymbols * noOfOfdmSymbolsPerFrame;
numOfFrames = ceil(noOfInfoBitsPerUser / infoBitsPerFrame);
walshCode = generateWalshCode(spreadingFactor);

%% Transmitter
% Generate random bits for each user
bits = randi([0 1], noOfUsers, noOfInfoBitsPerUser);

% Pass the data into the transmitter 
[rxFrames, TXBITS, TXSYM] = generateFrames(bits);
numFrames = length(rxFrames);

%% Channel Setup : Fading & Noise
fadedFrames = zeros(numFrames, numOfQpskSymbolsPerFrame);
for frame = 1:numFrames
    txFrame = rxFrames{frame};
    h = (randn + 1j*randn)/sqrt(2);
    fadedFrames(frame,:) = h * txFrame;
end

%% 4. Receiver loop : Sample-by-Sample Real-Time Processing
SNRdB = 0:2:26;
BER = zeros(size(SNRdB));
SER = zeros(size(SNRdB));
FER = zeros(size(SNRdB));

for snridx = 1:length(SNRdB)
    snr = SNRdB(snridx);
    
    % Storage for this specific SNR iteration
    RXBITS = cell(1, numFrames);
    RXSYM  = cell(1, numFrames);
    
    for frame = 1:numFrames
        fadedSignal = fadedFrames(frame,:);
        rxFrame = awgn(fadedSignal, snr, 'measured');
        
        noisePower = mean(abs(rxFrame - fadedSignal).^2);
        N_samples = length(rxFrame);
        agcOut = zeros(1, N_samples);
        
        % Initialize AGC & Sync variables  
        % P_est = 1;        
        agcGain = 1;        
        corrBuffer = zeros(1, barkerSyncLen);
        corrMagnitude = zeros(1, N_samples);
        
        %% AGC and Sliding Barker Correlation
        for n = 1:N_samples
            % AGC maintain
            if n <= agcSamLen
                P_est = abs(rxFrame(n))^2;     
                agcGain = sqrt(targetPower / P_est);   
            end
            agcOut(n) = rxFrame(n) * agcGain;
            
            % Synchronization sliding window
            corrBuffer(1:end-1) = corrBuffer(2:end);
            corrBuffer(end) = agcOut(n);
            corrMagnitude(n) = abs(sum(corrBuffer .* barkerSync));
        end
        
        %% Frame Detection
        searchStart = agcSamLen + 1;
        searchEnd = agcSamLen + barkerSyncLen + 20; 
        
        [~, localPeakIdx] = max(corrMagnitude(searchStart:searchEnd));
        barkerEndIdx = (searchStart - 1) + localPeakIdx;
        syncStartPoint = barkerEndIdx - barkerSyncLen + 1;
        
        if frame == 1 && snridx == 1
            fprintf('\n--- SNR: %d dB --- \n', snr);
            fprintf('AGC Frozen Gain    = %.4f\n', agcGain);
            fprintf('Sync Start Point   = %d \n', syncStartPoint);
            fprintf('Sync End Point     = %d \n', barkerEndIdx);
        end
        
        %% Channel Estimation
        rxTraining = rxFrame(barkerEndIdx+1 : barkerEndIdx+trainingLen);
        hMMSE = MMSE_Channel_Estimation(rxTraining, trainingSeq, noisePower);
        rxFrameEq = rxFrame / hMMSE;
        
        %% Extract Payload
        rxPayload = rxFrameEq(barkerEndIdx+trainingLen+1 : end-frameGuardLen);
        rxOFDM = reshape(rxPayload, Nfft+cpLength, noOfOfdmSymbolsPerFrame);
        rxNoCP = rxOFDM(cpLength+1:end, :);
        rxFFT = fft(rxNoCP, Nfft);
        
        rxData = rxFFT(guardLeft+1 : guardLeft+usedSubcarriers, :); 
        rxData_flat = rxData(:); 

        % usedSubcarriers -> total qpsk symbols have info
        rxUsers = zeros(noOfUsers, usedSubcarriers);
        
        for user = 1:noOfUsers
            code = walshCode(user,:).';
            idx = 1;
            for k = 1:usedSubcarriers
                chips = rxData_flat(idx:idx+spreadingFactor-1);
                rxUsers(user,k) = (code' * chips) / spreadingFactor; 
                idx = idx + spreadingFactor;
            end
        end
        userSymbols = rxUsers;
        
        % Decision : QPSK constellation
        rxSym = ((real(userSymbols)>0)*2-1 + 1j*((imag(userSymbols)>0)*2-1))/sqrt(2);
        RXSYM{frame} = rxSym;
        
        %% QPSK Demodulation
        codedBitsRx = zeros(noOfUsers, 2*size(userSymbols,2));
        for user = 1:noOfUsers
            sym = userSymbols(user,:);
            bitsHat = zeros(1, 2*length(sym));
            for k = 1:length(sym)
                bitsHat(2*k-1) = real(sym(k)) < 0;
                bitsHat(2*k)   = imag(sym(k)) < 0;
            end
            codedBitsRx(user,:) = bitsHat;
        end
        
        %% Viterbi Decoder
        decodedBits = zeros(noOfUsers, infoBitsPerFrame); 
        for user = 1:noOfUsers
            bitsHat = codedBitsRx(user,:);
            decoded = vitdec(bitsHat, trellis, 20, 'trunc', 'hard');
            decodedBits(user,:) = decoded(1:infoBitsPerFrame);
        end
        RXBITS{frame} = decodedBits;
        fprintf('Frame closed : %d\n', frame);
    end
    
    %% 5. Error Calculation : BER, SER, FER
    
    % received bits for BER calculation
    receivedBits = [];
    for frame = 1:numFrames
        receivedBits = [receivedBits RXBITS{frame}];
    end
    receivedBits = receivedBits(:, 1:noOfInfoBitsPerUser);
    
    % Calculate Overall BER
    BER(snridx) = sum(sum(bits ~= receivedBits)) / (noOfUsers * noOfInfoBitsPerUser);
    
    % Calculate FER & SER Frame-by-Frame
    frameErrors = 0;
    symbolErrors = 0;
    totalSymbols = 0;
    
    for frame = 1:numFrames
        % FER Check
        if any(RXBITS{frame}(:) ~= TXBITS{frame}(:))
            frameErrors = frameErrors + 1;
        end
        
        % SER Check
        symbolErrors = symbolErrors + sum(sum(TXSYM{frame} ~= RXSYM{frame}));
        totalSymbols = totalSymbols + numel(TXSYM{frame});
    end
    
    FER(snridx) = frameErrors / numFrames;
    SER(snridx) = symbolErrors / totalSymbols;
end
   
%% Plots
figure;
semilogy(SNRdB, BER, '-d', 'LineWidth', 2);
hold on; grid on;
semilogy(SNRdB, SER, '-o', 'LineWidth', 2);
semilogy(SNRdB, FER, '-s', 'LineWidth', 2);
xlabel('SNR (dB)');
ylabel('Error Rate');
title('MC-CDMA System Performance');
xlim([0 30]);   
ylim([1e-6 1]);
legend('Bit Error Rate (BER)', 'Symbol Error Rate (SER)', 'Frame Error Rate (FER)');

%% Generate Walsh Code
function H = generateWalshCode(M)
    H = 1;
    while size(H,1) < M
        H = [H H;
             H -H];
    end
end

%% MMSE estimation of Channel
function h_MMSE = MMSE_Channel_Estimation(rxPilot, txPilot, noiseVar)
    H_LS = rxPilot ./ txPilot;
    Np = length(H_LS);
    Rhh = eye(Np);
    W = Rhh / (Rhh + noiseVar*eye(Np));
    H_MMSE = (W * H_LS.').';
    h_MMSE = mean(H_MMSE);
end
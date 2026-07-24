    clc;
clear;
close all;
%% System Parameters
noOfUsers = 4;
noOfInfoBitsPerUser = 20000;
M = 256; % Decide QPSK or QAM -M Modulation

Nfft = 256;
cpLength = 16;
guardLeft  = 8;
guardRight = 8;
usedSubcarriers = Nfft - guardLeft - guardRight;
spreadingFactor = 4;
noOfOfdmSymbolsPerFrame = 4;
bitspersymbol = log2(M);

%% Code Rate Configuration
targetCodeRate = 1/4;

%% CHANNEL CODER 
if (1/targetCodeRate == 2)
    GeneratorPolynomials = [7 5];
elseif (1/targetCodeRate == 3)
    GeneratorPolynomials = [7 5 3];
elseif (1/targetCodeRate == 4) 
    GeneratorPolynomials = [7 5 3 1];
else 
    error('Invalid code rate! Please set targetCodeRate strictly to 1/2, 1/3, or 1/4.');
end
ConstraintLength = 3;
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

% Total no of symbols per frame
numOfTotalSymbolsPerFrame = agcSamLen+trainingLen+barkerSyncLen+ ...
                            (Nfft+cpLength)*noOfOfdmSymbolsPerFrame+frameGuardLen; 
% AGC Settings
targetPower = 1;      
tolerance   = 1e-3;   
maxIter     = 20;

% Derived Capacity Variables
symbolsPerOfdmSymbol = (usedSubcarriers/spreadingFactor);
infoBitsPerOfdmSymbols = (usedSubcarriers/spreadingFactor)*bitspersymbol*codeRate;
infoBitsPerFrame = infoBitsPerOfdmSymbols * noOfOfdmSymbolsPerFrame;
numOfFrames = ceil(noOfInfoBitsPerUser / infoBitsPerFrame);

%% GENERATE WALSH CODE - defined by spreading factor
walshCode = generateWalshCode(spreadingFactor);

%% Transmitter
% Generate random bits for each user
bits = randi([0 1], noOfUsers, noOfInfoBitsPerUser);

% Pass the data into the transmitter 
[rxFrames, TXBITS, TXSYM] = generateFrames(bits, targetCodeRate, M);
numFrames = length(rxFrames);

%% Channel Setup : Fading & Noise
fadedFrames = zeros(numFrames, numOfTotalSymbolsPerFrame);
for frame = 1:numFrames
    txFrame = rxFrames{frame};
    h = (randn + 1j*randn)/sqrt(2);
    fadedFrames(frame,:) = h * txFrame;
end
startSNRdB = -10;
endSNRdB = 40;

%% Receiver loop : Sample-by-Sample Real-Time Processing
SNRdB = startSNRdB:2:endSNRdB;
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
        h_Lse = LSE_Channel_Estimation(rxTraining, trainingSeq ); 
        h_Lse = mean(h_Lse);
        rxFrameEq = rxFrame / h_Lse;
     
        %% Extract Payload (with sync-failure guard)
        expectedPayloadLen = (Nfft+cpLength)*noOfOfdmSymbolsPerFrame;
        rxPayload = rxFrameEq(barkerEndIdx+trainingLen+1 : end-frameGuardLen);
        
        if length(rxPayload) ~= expectedPayloadLen
            RXBITS{frame} = double(rand(noOfUsers, infoBitsPerFrame) > 0.5);
            RXSYM{frame} = zeros(noOfUsers, usedSubcarriers);
            fprintf('Frame %d with SNR(dB) = %d : SYNC FAILURE (got %d samples, expected %d) - marked as lost frame\n', ...
                frame, snr, length(rxPayload), expectedPayloadLen);
            continue
        end
        
        rxOFDM = reshape(rxPayload, Nfft+cpLength, noOfOfdmSymbolsPerFrame);
        rxNoCP = rxOFDM(cpLength+1:end, :);
        rxFFT = fft(rxNoCP, Nfft);
        
        % Reverse the transmitter's OFDM power normalization
        % so the M-QAM constellation amplitudes return to ideal 3GPP values
        scaleFactor = Nfft / sqrt(usedSubcarriers * noOfUsers);
        rxFFT = rxFFT / scaleFactor;
        
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
        
        %% Decision : QPSK constellation
        %% 3GPP M-QAM Demodulation
        codedBitsRx = zeros(noOfUsers, bitspersymbol * size(userSymbols,2));
        rxSym = zeros(size(userSymbols));
        
        for user = 1:noOfUsers
            % 1. Extract bits using strict 3GPP decision boundaries
            recoveredBits = symbolDemapper(userSymbols(user,:), M);
            codedBitsRx(user,:) = recoveredBits;
            
            % 2. Remap bits back to ideal constellation points for SER calculation
            rxSym(user,:) = symbolMapper(recoveredBits, M);
        end
        RXSYM{frame} = rxSym;
        
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
title(sprintf('MC-CDMA System Performance (Code Rate = 1/%d, QAM-%d)', ...
    length(GeneratorPolynomials), M));
xlim([startSNRdB endSNRdB]);   
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
function H_LSE = LSE_Channel_Estimation(rxPilot, txPilot)
    
    H_LSE = rxPilot ./ txPilot;
    
   
end


%% SYMBOL DEMAPPER FUNCTION (Strict 3GPP Hard Decisions)
function bits = symbolDemapper(symbols, M)
    I = real(symbols);
    Q = imag(symbols);
    
    switch M
        case 4 % QPSK
            b_I = I < 0;
            b_Q = Q < 0;
            tempBits = [b_I(:) b_Q(:)];
            
        case 16 % 16-QAM
            I = I * sqrt(10); Q = Q * sqrt(10);
            b_I1 = I < 0;           b_Q1 = Q < 0;
            b_I2 = abs(I) > 2;      b_Q2 = abs(Q) > 2;
            tempBits = [b_I1(:) b_Q1(:) b_I2(:) b_Q2(:)];
            
        case 64 % 64-QAM
            I = I * sqrt(42); Q = Q * sqrt(42);
            b_I1 = I < 0;               b_Q1 = Q < 0;
            b_I2 = abs(I) > 4;          b_Q2 = abs(Q) > 4;
            b_I3 = abs(abs(I)-4) > 2;   b_Q3 = abs(abs(Q)-4) > 2;
            tempBits = [b_I1(:) b_Q1(:) b_I2(:) b_Q2(:) b_I3(:) b_Q3(:)];
            
        case 256 % 256-QAM
            I = I * sqrt(170); Q = Q * sqrt(170);
            b_I1 = I < 0;                       b_Q1 = Q < 0;
            b_I2 = abs(I) > 8;                  b_Q2 = abs(Q) > 8;
            b_I3 = abs(abs(I)-8) > 4;           b_Q3 = abs(abs(Q)-8) > 4;
            b_I4 = abs(abs(abs(I)-8)-4) > 2;    b_Q4 = abs(abs(abs(Q)-8)-4) > 2;
            tempBits = [b_I1(:) b_Q1(:) b_I2(:) b_Q2(:) b_I3(:) b_Q3(:) b_I4(:) b_Q4(:)];
            
        otherwise
            error('Unsupported modulation order for 3GPP demapping.');
    end
    
    % Flatten into a 1D bitstream
    bits = tempBits.';
    bits = bits(:).';
end

%% SYMBOL MAPPER FUNCTION (Receiver SER calculation)
function symbols = symbolMapper(bits, M)
    switch M
        case 4 % QPSK
            tempBits = reshape(bits, 2, []).';
            I = (1 - 2*tempBits(:,1)) / sqrt(2);
            Q = (1 - 2*tempBits(:,2)) / sqrt(2);
            symbols = (I + 1j*Q).';
            
        case 16 % 16-QAM
            tempBits = reshape(bits, 4, []).';
            I = (1 - 2*tempBits(:,1)) .* (2 - (1 - 2*tempBits(:,3))) / sqrt(10);
            Q = (1 - 2*tempBits(:,2)) .* (2 - (1 - 2*tempBits(:,4))) / sqrt(10);
            symbols = (I + 1j*Q).';
            
        case 64 % 64-QAM
            tempBits = reshape(bits, 6, []).';
            I = (1 - 2*tempBits(:,1)) .* (4 - (1 - 2*tempBits(:,3)) .* (2 - (1 - 2*tempBits(:,5)))) / sqrt(42);
            Q = (1 - 2*tempBits(:,2)) .* (4 - (1 - 2*tempBits(:,4)) .* (2 - (1 - 2*tempBits(:,6)))) / sqrt(42);
            symbols = (I + 1j*Q).';
            
        case 256 % 256-QAM
            tempBits = reshape(bits, 8, []).';
            I = (1 - 2*tempBits(:,1)) .* (8 - (1 - 2*tempBits(:,3)) .* (4 - (1 - 2*tempBits(:,5)) .* (2 - (1 - 2*tempBits(:,7))))) / sqrt(170);
            Q = (1 - 2*tempBits(:,2)) .* (8 - (1 - 2*tempBits(:,4)) .* (4 - (1 - 2*tempBits(:,6)) .* (2 - (1 - 2*tempBits(:,8))))) / sqrt(170);
            symbols = (I + 1j*Q).';
    end
end
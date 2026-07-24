clc;
clear;
close all;

%% PARAMETERS
M = 16 ;     % For QAM -M : change this to 4, 16, 64, or 256
noOfUsers  = 4;
noOfInfoBitsPerUser = 200000;
Nfft = 256;
cpLength = 16;
guardLeft  = 8;
guardRight = 8;
usedSubcarriers = Nfft - guardLeft - guardRight;
spreadingFactor = 4;
noOfOfdmSymbolsPerFrame = 4;
bitspersymbol = log2(M);

%% Code Rate Configuration
targetCodeRate = 1/2;   % For Code Rate : change this to 1/2, 1/3, or 1/4

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
trellis = poly2trellis(ConstraintLength,GeneratorPolynomials); 

%% FRAME CAPACITY
symbolsPerOfdmSymbol = (usedSubcarriers/spreadingFactor);   % per user
infoBitsPerOfdmSymbols = (usedSubcarriers/spreadingFactor)*bitspersymbol*codeRate ;   % (232/4)*2*1/2 = 58 info bits/ofdm symbol for a user
infoBitsPerFrame = infoBitsPerOfdmSymbols*noOfOfdmSymbolsPerFrame;    % 58*4 = 232 info bits/frame for a user 
numOfFrames = ceil(noOfInfoBitsPerUser/infoBitsPerFrame);

%% FRAME PREAMBLE (AGC + BARKER SYNC + TRAINING)
agcField = repmat([1 -1],1,50);
% BARKER 11 SEQUENCE , Used for frame synchronization
barker11 = [ 1  1  1 -1 -1 -1 1 -1 -1  1 -1 ];
% Repeat Barker 4 times
barkerSync = repmat(barker11,1,4);     % Length = 44
trainingSeq = repmat([1 -1],1,50);
frameGuard=zeros(1,30);

%% USER DATA
randomBits = randi([0 1],noOfUsers,noOfInfoBitsPerUser);

%% WALSH CODE
walshCode = generateWalshCode(spreadingFactor);

%% TX FRAME LOOP
infoBitsPerFrameContainer = zeros(noOfUsers, infoBitsPerFrame);
for frame = 1:numOfFrames
    fprintf('TX Frame = %d\n',frame);
    
    %% FRAME DATA EXTRACTION
    startBit = (frame-1)*infoBitsPerFrame + 1;
    endBit   = min(frame*infoBitsPerFrame,noOfInfoBitsPerUser);
    infoBitsPerFrameContainer = randomBits(:,startBit:endBit);
    
    %% LAST FRAME PADDING
    if size(infoBitsPerFrameContainer,2) < infoBitsPerFrame
        padLen = infoBitsPerFrame - size(infoBitsPerFrameContainer,2);
        infoBitsPerFrameContainer = [infoBitsPerFrameContainer zeros(noOfUsers,padLen)];
    end
   
    %% CONVOLUTIONAL ENCODING
    codedBits = zeros(noOfUsers,(1/codeRate)*infoBitsPerFrame);
    for user = 1:noOfUsers
        codedBits(user,:) = convenc(infoBitsPerFrameContainer(user,:),trellis);
    end
   
    %% M-QAM SYMBOL MAPPING 
    totalSymbolsPerFrame = symbolsPerOfdmSymbol * noOfOfdmSymbolsPerFrame;
    mappedSymbols = zeros(noOfUsers, totalSymbolsPerFrame);
    for user = 1:noOfUsers
        % Dynamically maps using the 3GPP standard based on the M variable
        mappedSymbols(user,:) = symbolMapper(codedBits(user,:), M);
    end
    
    %% STORE OFDM SYMBOLS
    ofdmSymbolsPerFrame = zeros(Nfft+cpLength,noOfOfdmSymbolsPerFrame);   
    ofdmSymbolsPerFrameUnspread = zeros(Nfft+cpLength,noOfOfdmSymbolsPerFrame);  
    
    % Generalized variable name
    symbolsPerOFDMBeforeSpreading = (usedSubcarriers/spreadingFactor);
    
    %% OFDM Symbols per frame loop
    OfdmFramePlot = zeros(Nfft+cpLength,1);
    OfdmFrameUnspreadPlot = zeros(Nfft+cpLength,1);
    for ofdm = 1:noOfOfdmSymbolsPerFrame
        startSym = (ofdm-1)*symbolsPerOFDMBeforeSpreading + 1;
        endSym = ofdm*symbolsPerOFDMBeforeSpreading;
        
        % Extract from the newly generalized mappedSymbols array
        currentSymbols = mappedSymbols(:,startSym:endSym);
        
        %% SPREADING
        spreadSignal = zeros(noOfUsers,symbolsPerOFDMBeforeSpreading*spreadingFactor);
        for user = 1:noOfUsers
            code = walshCode(user,:);
            userSpread = [];
            for k = 1:size(currentSymbols,2)
                chips = currentSymbols(user,k).*code;
                userSpread = [userSpread chips];
            end
            spreadSignal(user,:) = userSpread;
        end
        
        %% MULTIUSER COMBINATION
        txCombined = sum(spreadSignal,1); % sum of user symbols column wise
        
        %% OFDM 
        % 1. IFFT of OFDM symbols for spread signal
        ofdmInput = zeros(Nfft,1);
        ofdmInput(guardLeft+1 :guardLeft+usedSubcarriers) = txCombined(:);
        ofdmTime = ifft(ofdmInput,Nfft);
        
        % 2. Unspread Signal (Normal OFDMA for comparison)
        % Taking only the 58 symbols for a true narrowband signal
        txUnspread = currentSymbols(1, :);  % for taking one user unspread data 
        
        centerIdx = Nfft/2;
        halfLen = length(txUnspread)/2;
        
        ofdmInputUnspread = zeros(Nfft,1);
        ofdmInputUnspread(centerIdx - halfLen + 1 : centerIdx + halfLen) = txUnspread(:);
        ofdmTimeUnspread = ifft(ofdmInputUnspread, Nfft);
        
        %% POWER NORMALIZATION
        % MATLAB's ifft() divides by Nfft, which shrinks the signal. 
        % We multiply by Nfft to restore it, then divide by the total variance 
        % (active subcarriers * number of users) to normalize average power to 1.
        scaleFactor = Nfft / sqrt(usedSubcarriers * noOfUsers);  
        ofdmTime = ofdmTime * scaleFactor;
        
        % Normalize the unspread signal identically for fair PSD comparison
        % (Assuming 1 user active in the unspread comparison)
        scaleFactorUnspread = Nfft / sqrt(usedSubcarriers * 1); 
        ofdmTimeUnspread = ofdmTimeUnspread * scaleFactorUnspread;

        %% CYCLIC PREFIX
        % for Spread
        cp = ofdmTime(end-cpLength+1:end);
        cpAdded = [cp ; ofdmTime];
        OfdmFrame(:,ofdm) = cpAdded;
        
        % for unspread
        cpU = ofdmTimeUnspread(end-cpLength+1:end);
        cpAddedU = [cpU ; ofdmTimeUnspread];
        OfdmFrameUnspread(:,ofdm) = cpAddedU;
    end
   
    %% SERIALIZE OFDM PAYLOAD
    combinedOfdmSymbols = OfdmFrame(:).';
    combinedOfdmSymbolsUnspread = OfdmFrameUnspread(:).';


    %% COMPLETE FRAME
    txFrame = [agcField barkerSync trainingSeq combinedOfdmSymbols frameGuard];
    
    % Store the first frame for plotting
    if frame == 1
        txBaseband = combinedOfdmSymbols(1:Nfft+cpLength);
        txBasebandUnspread = combinedOfdmSymbolsUnspread(1:Nfft+cpLength); 
    end
end
disp('Transmitter Complete');

%% Plotting Specifications
Fs = 40e6;      % Sampling frequency (40 MHz)
N = length(txBaseband);

%% 1. FREQUENCY SPECTRUM (FFT)
% Using fftshift to center the frequency at 0 Hz
TX_FFT_spread   = fftshift(fft(txBaseband,N));
TX_FFT_unspread = fftshift(fft(txBasebandUnspread,N));

% Create the true frequency axis in Hz
freqAxis = (-N/2:N/2-1)*(Fs/N);

figure;
plot(freqAxis/1e6, 20*log10(abs(TX_FFT_unspread)+1e-15), 'b', 'LineWidth', 1.5);
hold on;
plot(freqAxis/1e6, 20*log10(abs(TX_FFT_spread)+1e-15), 'r', 'LineWidth', 1.5);
grid on;
xlabel('Frequency (MHz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum: Before vs After Spreading');
legend('Before Spreading (58 Narrowband Subcarriers)', 'After Spreading (232 Wideband Subcarriers)', 'Location', 'best');
xlim([-Fs/2/1e6 Fs/2/1e6]);

%% 2. POWER SPECTRAL DENSITY (PSD)
% Using pwelch to estimate PSD centered at 0 Hz
[pxx_spread, f_psd]   = pwelch(txBaseband, hamming(256), [], [], Fs, 'centered');
[pxx_unspread, ~]     = pwelch(txBasebandUnspread, hamming(256), [], [], Fs, 'centered');

figure;
plot(f_psd/1e6, 10*log10(pxx_unspread+1e-20), 'b', 'LineWidth', 1.5);
hold on;
plot(f_psd/1e6, 10*log10(pxx_spread+1e-20), 'r', 'LineWidth', 1.5);
grid on; 
xlabel('Frequency (MHz)');
ylabel('Power/Frequency (dB/Hz)');
title('PSD of Transmitted Baseband Signal: Before vs After Spreading');
legend('Before Spreading (58 Narrowband Subcarriers)', 'After Spreading (232 Wideband Subcarriers)', 'Location', 'best');
xlim([-Fs/2/1e6 Fs/2/1e6]);


figure;
% Flatten the mappedSymbols matrix to plot the constellation of all users
scatter(real(mappedSymbols(:)), imag(mappedSymbols(:)), 50, 'x', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
grid on;
axis square; % Keeps the I and Q axes proportionally equal

% Title based on your M parameter
title(sprintf('Transmitted Constellation Diagram (QAM-%d)', M));
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');

xline(0, 'k--', 'LineWidth', 1);
yline(0, 'k--', 'LineWidth', 1);
maxAmp = max(abs(mappedSymbols(:)));
xlim([-(maxAmp + 0.3) (maxAmp + 0.3)]);
ylim([-(maxAmp + 0.3) (maxAmp + 0.3)]);

%% WALSH CODE FUNCTION
function H = generateWalshCode(sf)
    H = 1;
    while size(H,1) < sf
        H = [H H;
             H -H];
    end
end


%% SYMBOL MAPPER FUNCTION (3GPP TS 38.211 / 36.211 Standard)
function symbols = symbolMapper(bits, M)
    % bits : 1D array of binary data (row vector)
    % M    : Modulation order (4, 16, 64, 256)
    
    switch M
        case 4 % QPSK
            tempBits = reshape(bits, 2, []).';
            b_I = tempBits(:,1);
            b_Q = tempBits(:,2);
            
            % 3GPP QPSK Equation
            I = (1 - 2*b_I) / sqrt(2);
            Q = (1 - 2*b_Q) / sqrt(2);
            symbols = (I + 1j*Q).';
            
        case 16 % 16-QAM
            tempBits = reshape(bits, 4, []).';
            b_I1 = tempBits(:,1); b_Q1 = tempBits(:,2);
            b_I2 = tempBits(:,3); b_Q2 = tempBits(:,4);
            
            % 3GPP 16-QAM Equation
            I = (1 - 2*b_I1) .* (2 - (1 - 2*b_I2)) / sqrt(10);
            Q = (1 - 2*b_Q1) .* (2 - (1 - 2*b_Q2)) / sqrt(10);
            symbols = (I + 1j*Q).';
            
        case 64 % 64-QAM
            tempBits = reshape(bits, 6, []).';
            b_I1 = tempBits(:,1); b_Q1 = tempBits(:,2);
            b_I2 = tempBits(:,3); b_Q2 = tempBits(:,4);
            b_I3 = tempBits(:,5); b_Q3 = tempBits(:,6);
            
            % 3GPP 64-QAM Equation
            I = (1 - 2*b_I1) .* (4 - (1 - 2*b_I2) .* (2 - (1 - 2*b_I3))) / sqrt(42);
            Q = (1 - 2*b_Q1) .* (4 - (1 - 2*b_Q2) .* (2 - (1 - 2*b_Q3))) / sqrt(42);
            symbols = (I + 1j*Q).';
            
        case 256 % 256-QAM
            tempBits = reshape(bits, 8, []).';
            b_I1 = tempBits(:,1); b_Q1 = tempBits(:,2);
            b_I2 = tempBits(:,3); b_Q2 = tempBits(:,4);
            b_I3 = tempBits(:,5); b_Q3 = tempBits(:,6);
            b_I4 = tempBits(:,7); b_Q4 = tempBits(:,8);
            
            % 3GPP 256-QAM Equation
            I = (1 - 2*b_I1) .* (8 - (1 - 2*b_I2) .* (4 - (1 - 2*b_I3) .* (2 - (1 - 2*b_I4)))) / sqrt(170);
            Q = (1 - 2*b_Q1) .* (8 - (1 - 2*b_Q2) .* (4 - (1 - 2*b_Q3) .* (2 - (1 - 2*b_Q4)))) / sqrt(170);
            symbols = (I + 1j*Q).';
            
        otherwise
            error('Unsupported modulation order. 3GPP mapping supports M = 4, 16, 64, or 256.');
    end
end
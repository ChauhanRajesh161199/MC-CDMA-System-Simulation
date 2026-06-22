clc;
clear;
close all;

%% PARAMETERS
noOfUsers  = 4;
noOfInfoBitsPerUser = 10000;
Nfft = 256;
cpLength = 16;
guardLeft  = 8;
guardRight = 8;
usedSubcarriers = Nfft - guardLeft - guardRight;
spreadingFactor = 4;
noOfOfdmSymbolsPerFrame = 4;
bitspersymbol = 2;  % QPSK Modulation

%% CHANNEL CODER
ConstraintLength = 3;  
GeneratorPolynomials = [7 5];
codeRate = 1/length(GeneratorPolynomials);
trellis = poly2trellis(ConstraintLength,GeneratorPolynomials); 

%% FRAME CAPACITY
infoBitsPerOfdmSymbols = (usedSubcarriers/spreadingFactor)*bitspersymbol*codeRate ;   % (232/4)*2*1/2 = 58 info bits/ofdm symbol for a user
infoBitsPerFrame = infoBitsPerOfdmSymbols*noOfOfdmSymbolsPerFrame;    % 58*4 = 232 info bits/frame for a user 
qpskSymbolsPerOfdmSymbol = (usedSubcarriers/spreadingFactor);   % per user
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
   
    %% QPSK 
    totalQpskSymbolsPerFrame = qpskSymbolsPerOfdmSymbol*noOfOfdmSymbolsPerFrame;
    qpskSymbols = zeros(noOfUsers,totalQpskSymbolsPerFrame);
    for user = 1:noOfUsers
        tempBits = reshape(codedBits(user,:),2,[]).';
        qpskSymbols(user,:) = ...
            ((1-2*tempBits(:,1)) + ...
            1j*(1-2*tempBits(:,2)))/sqrt(2);
    end
    
    %% STORE OFDM SYMBOLS
    ofdmSymbolsPerFrame = zeros(Nfft+cpLength,noOfOfdmSymbolsPerFrame);   % its look like serial to parallel
    ofdmSymbolsPerFrameUnspread = zeros(Nfft+cpLength,noOfOfdmSymbolsPerFrame);  
    qpskSymbolsPerOFDMBeforeSpreading = (usedSubcarriers/spreadingFactor);
    %% OFDM Sumbols per frame loop

    OfdmFramePlot = zeros(Nfft+cpLength,1);
    OfdmFrameUnspreadPlot = zeros(Nfft+cpLength,1);
    for ofdm = 1:noOfOfdmSymbolsPerFrame
        startSym = (ofdm-1)*qpskSymbolsPerOFDMBeforeSpreading + 1;
        endSym = ofdm*qpskSymbolsPerOFDMBeforeSpreading;
        currentSymbols = qpskSymbols(:,startSym:endSym);
        
        %% SPREADING
        spreadSignal = zeros(noOfUsers,qpskSymbolsPerOFDMBeforeSpreading*spreadingFactor);
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
        txUnspread = currentSymbols(1, :);  % for taking one user unspread data (qpsk symbols)
        
        centerIdx = Nfft/2;
        halfLen = length(txUnspread)/2;
        
        ofdmInputUnspread = zeros(Nfft,1);
        ofdmInputUnspread(centerIdx - halfLen + 1 : centerIdx + halfLen) = txUnspread(:);
        ofdmTimeUnspread = ifft(ofdmInputUnspread, Nfft);
        
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

%% WALSH CODE FUNCTION
function H = generateWalshCode(sf)
    H = 1;
    while size(H,1) < sf
        H = [H H;
             H -H];
    end
end
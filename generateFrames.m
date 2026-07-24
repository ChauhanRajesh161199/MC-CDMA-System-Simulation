function [frames, TXBITS, TXSYM] = generateFrames(bits, targetCodeRate, M)
    %% PARAMETERS
    % Dynamically determine the number of users and total bits from the input array
    [noOfUsers, noOfInfoBitsPerUser] = size(bits);
    Nfft = 256;
    cpLength = 16;
    guardLeft  = 8;
    guardRight = 8;
    usedSubcarriers = Nfft - guardLeft - guardRight;
    spreadingFactor = 4;
    noOfOfdmSymbolsPerFrame = 4;
    
    % Calculate bits per symbol based on Modulation Order (M)
    bitspersymbol = log2(M);


    %% FRAME PREAMBLE
    agcField = repmat([1 -1],1,50);
    barker11 = [ 1 1 1 -1 -1 -1 1 -1 -1 1 -1 ];
    barkerSync = repmat(barker11,1,4);
    trainingSeq = repmat([1 -1],1,50);
    frameGuard = zeros(1,30);

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
    symbolsPerOfdmSymbol = (usedSubcarriers/spreadingFactor);
    infoBitsPerOfdmSymbols = (usedSubcarriers/spreadingFactor)*bitspersymbol*codeRate;
    infoBitsPerFrame = infoBitsPerOfdmSymbols*noOfOfdmSymbolsPerFrame;
    numOfFrames = ceil(noOfInfoBitsPerUser/infoBitsPerFrame);
    
    %% WALSH CODE
    walshCode = generateWalshCode(spreadingFactor);
    
    %% STORAGE DATA
    frames = cell(1,numOfFrames);
    TXBITS = cell(1,numOfFrames);
    TXSYM  = cell(1,numOfFrames);
    
    for frame = 1:numOfFrames
        startBit = (frame-1)*infoBitsPerFrame + 1;
        endBit   = min(frame*infoBitsPerFrame,noOfInfoBitsPerUser);
        infoBitsPerFrameContainer = bits(:,startBit:endBit);
        
        if size(infoBitsPerFrameContainer,2) < infoBitsPerFrame
            padLen = infoBitsPerFrame - size(infoBitsPerFrameContainer,2);
            infoBitsPerFrameContainer = [infoBitsPerFrameContainer zeros(noOfUsers,padLen)];
        end
        
        % Save true bits into cell array for FER calculation at receiver
        TXBITS{frame} = infoBitsPerFrameContainer;
        
        %% CONVOLUTIONAL ENCODING
        codedBits = zeros(noOfUsers, (1/codeRate)*infoBitsPerFrame);
        for user = 1:noOfUsers
            codedBits(user,:) = convenc(infoBitsPerFrameContainer(user,:),trellis);
        end

        %% MODULATION 
        totalSymbolsPerFrame = symbolsPerOfdmSymbol*noOfOfdmSymbolsPerFrame;
        mappedSymbols = zeros(noOfUsers, totalSymbolsPerFrame);
        
        for user = 1:noOfUsers
            % Pass the user's coded bits directly into the 3GPP mapping function
            mappedSymbols(user,:) = symbolMapper(codedBits(user,:), M);
        end
        
        % Save true mapped symbols into cell array for SER calculation at receiver
        TXSYM{frame} = mappedSymbols;

        %% OFDM
        OfdmFrame = zeros(Nfft+cpLength, noOfOfdmSymbolsPerFrame);
        symbolsPerOFDMBeforeSpreading = usedSubcarriers/spreadingFactor;
        
        for ofdm = 1:noOfOfdmSymbolsPerFrame
            startSym = (ofdm-1)*symbolsPerOFDMBeforeSpreading + 1;
            endSym = ofdm*symbolsPerOFDMBeforeSpreading;
            % Fetch from the newly named mappedSymbols array
            currentSymbols = mappedSymbols(:,startSym:endSym);
            
            %% SPREADING
            spreadSignal = zeros(noOfUsers, symbolsPerOFDMBeforeSpreading*spreadingFactor);
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
            txCombined = sum(spreadSignal,1);
            
            %% SPREAD OFDM
            ofdmInput = zeros(Nfft,1);
            ofdmInput(guardLeft+1:guardLeft+usedSubcarriers) = txCombined(:);
            ofdmTime = ifft(ofdmInput,Nfft);

            %% CP ADDITION (with power normalization)
            scaleFactor = Nfft / sqrt(usedSubcarriers * noOfUsers);  % normalizes avg power to 1
            ofdmTime = ofdmTime * scaleFactor;
            cp = ofdmTime(end-cpLength+1:end);
            OfdmFrame(:,ofdm) = [cp; ofdmTime];
           
        end
        
        %% SERIALIZE
        combinedOfdmSymbols = OfdmFrame(:).';
        
        txFrame = [agcField barkerSync trainingSeq combinedOfdmSymbols frameGuard];
        frames{frame} = txFrame;
        
    end
    disp('Transmitter Complete');
end
%% WALSH FUNCTION
function H = generateWalshCode(sf)
    H = 1;
    while size(H,1) < sf
        H = [H H;
             H -H];
    end
end


%% SYMBOL MAPPER FUNCTION (3GPP Standard)
function symbols = symbolMapper(bits, M)
    switch M
        case 4 % QPSK
            tempBits = reshape(bits, 2, []).';
            b_I = tempBits(:,1);
            b_Q = tempBits(:,2);
            I = (1 - 2*b_I) / sqrt(2);
            Q = (1 - 2*b_Q) / sqrt(2);
            symbols = (I + 1j*Q).';
            
        case 16 % 16-QAM
            tempBits = reshape(bits, 4, []).';
            b_I1 = tempBits(:,1); b_Q1 = tempBits(:,2);
            b_I2 = tempBits(:,3); b_Q2 = tempBits(:,4);
            I = (1 - 2*b_I1) .* (2 - (1 - 2*b_I2)) / sqrt(10);
            Q = (1 - 2*b_Q1) .* (2 - (1 - 2*b_Q2)) / sqrt(10);
            symbols = (I + 1j*Q).';
            
        case 64 % 64-QAM
            tempBits = reshape(bits, 6, []).';
            b_I1 = tempBits(:,1); b_Q1 = tempBits(:,2);
            b_I2 = tempBits(:,3); b_Q2 = tempBits(:,4);
            b_I3 = tempBits(:,5); b_Q3 = tempBits(:,6);
            I = (1 - 2*b_I1) .* (4 - (1 - 2*b_I2) .* (2 - (1 - 2*b_I3))) / sqrt(42);
            Q = (1 - 2*b_Q1) .* (4 - (1 - 2*b_Q2) .* (2 - (1 - 2*b_Q3))) / sqrt(42);
            symbols = (I + 1j*Q).';
            
        case 256 % 256-QAM
            tempBits = reshape(bits, 8, []).';
            b_I1 = tempBits(:,1); b_Q1 = tempBits(:,2);
            b_I2 = tempBits(:,3); b_Q2 = tempBits(:,4);
            b_I3 = tempBits(:,5); b_Q3 = tempBits(:,6);
            b_I4 = tempBits(:,7); b_Q4 = tempBits(:,8);
            I = (1 - 2*b_I1) .* (8 - (1 - 2*b_I2) .* (4 - (1 - 2*b_I3) .* (2 - (1 - 2*b_I4)))) / sqrt(170);
            Q = (1 - 2*b_Q1) .* (8 - (1 - 2*b_Q2) .* (4 - (1 - 2*b_Q3) .* (2 - (1 - 2*b_Q4)))) / sqrt(170);
            symbols = (I + 1j*Q).';
            
        otherwise
            error('Unsupported modulation order. 3GPP mapping supports M = 4, 16, 64, or 256.');
    end
end
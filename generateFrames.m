function [frames, TXBITS, TXSYM] = generateFrames(bits, targetCodeRate)
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
    bitspersymbol = 2;
    
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
    infoBitsPerOfdmSymbols = (usedSubcarriers/spreadingFactor)*bitspersymbol*codeRate;
    infoBitsPerFrame = infoBitsPerOfdmSymbols*noOfOfdmSymbolsPerFrame;
    qpskSymbolsPerOfdmSymbol = (usedSubcarriers/spreadingFactor);
    numOfFrames = ceil(noOfInfoBitsPerUser/infoBitsPerFrame);
    
    %% WALSH CODE
    walshCode = generateWalshCode(spreadingFactor);
    
    %% STORAGE (Outputs)
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
        
        %% QPSK
        totalQpskSymbolsPerFrame = qpskSymbolsPerOfdmSymbol*noOfOfdmSymbolsPerFrame;
        qpskSymbols = zeros(noOfUsers, totalQpskSymbolsPerFrame);
        for user = 1:noOfUsers
            tempBits = reshape(codedBits(user,:),2,[]).';
            qpskSymbols(user,:) = ((1-2*tempBits(:,1)) + 1j*(1-2*tempBits(:,2)))/sqrt(2);
        end
        
        % Save true QPSK symbols into cell array for SER calculation at receiver
        % This stores the exact 4x240 double matrix you requested
        TXSYM{frame} = qpskSymbols;
        
        %% OFDM
        OfdmFrame = zeros(Nfft+cpLength, noOfOfdmSymbolsPerFrame);
        qpskSymbolsPerOFDMBeforeSpreading = usedSubcarriers/spreadingFactor;
        
        for ofdm = 1:noOfOfdmSymbolsPerFrame
            startSym = (ofdm-1)*qpskSymbolsPerOFDMBeforeSpreading + 1;
            endSym = ofdm*qpskSymbolsPerOFDMBeforeSpreading;
            currentSymbols = qpskSymbols(:,startSym:endSym);
            
            %% SPREADING
            spreadSignal = zeros(noOfUsers, qpskSymbolsPerOFDMBeforeSpreading*spreadingFactor);
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
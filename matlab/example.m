clear all
close all
clc

%% settings
path = 'C:\Users\vifle001\sciebo\Forschung\Datasets\PPG_DFG_BP_Studie'; % path to data
numKernels = 3;
kernelTypes = 'RayleighGaussian';
method = 'generic';
normOut = false;
parameterList = {'b_a','T1'};

%% include directories
workingDir = pwd;
addpath('helpFunctions')
addpath(genpath('Parameters'));
addpath(path);
cd(path)
javaaddpath 'unisens\unisens\org\lib\org.unisens.jar'
javaaddpath 'unisens\unisens\org\lib\org.unisens.ri.jar'
addpath(genpath('unisens\'))
addpath('cbppgprocessing/examples')
originalDataDirectory='data\';%directory where data folders are currently stored (should be relativ from this script) '..\..\measurements\'
currentPatient='subject015';%enter here a subject ID
currentUnisens= [abspath(originalDataDirectory) '\' currentPatient '_unisens'];%current patient + path (in the original folder additional '\' currentPatient 
cd(workingDir)

%% load ppg data
ppgSignal=unisens_get_data(currentUnisens,'PPG_finger.bin','all');%get PPG (whole signal)
ppgSignalSampleRate=unisens_get_samplerate(currentUnisens,'PPG_finger.bin');%get current sample rate using th eunisens function
ppgSignalTime=(0:1:size(ppgSignal,1)-1)/ppgSignalSampleRate;%create a time axis (in seconds)
markers = unisens_get_data(currentUnisens,['Markers' '.csv'],'all');%get responses
markers.times=markers.samplestamp/unisens_get_samplerate(currentUnisens,['Markers' '.csv']);%convert marker times' into seconds
markerNumber=2;%specify the number where to read
timeBefore=5;%time before marker
timeAfter=5;%time after marker
startIndex=find(ppgSignalTime>markers.times(2)-timeBefore,1,'first');%find first index
endIndex=find(ppgSignalTime<=markers.times(2)+timeAfter,1,'last');%find last index
ppgExcerpt=ppgSignal(startIndex:endIndex);%get PPG excerpt

% filter ppg
ppgExcerpt = movmean(ppgExcerpt,50);
% plot(ppgExcerpt)
% hold on
% plot(ppgExcerpt2)

[ppgDetections] = pqe_beatDetection_lazaro( ppgExcerpt', ppgSignalSampleRate, 0 );%find peaks (maximum slopes)

%% segment into beats
% can is use old version of create single beats?
[singleBeatsProcessed,singleBeats,importantPoints] = createSingleBeats(ppgExcerpt,ppgSignalSampleRate,ppgDetections);

%% decompose beats & calculate parameters
decompositionResults = struct;
parameters = cell(numel(singleBeatsProcessed),numel(parameterList));
for currentBeat = 1:numel(singleBeatsProcessed)
    if (~isnan(singleBeatsProcessed{currentBeat}))
        [signal_mod,y,opt_val_sort] = decompositionAlgorithm(singleBeatsProcessed{currentBeat},ppgSignalSampleRate,...
            'numKernels',numKernels,'kernelTypes',kernelTypes,'method',method,'normalizeOutput',normOut);
        [parameters{currentBeat,1:numel(parameterList)}] = calculateParameter(signal_mod,singleBeatsProcessed{currentBeat},y,opt_val_sort,[kernelTypes,numKernels],ppgSignalSampleRate,parameterList);
        figure('name',num2str(currentBeat))
        %plot(normalize(deriv2(signal_mod)))
        hold on
        plot(normalize(singleBeatsProcessed{currentBeat}))
        plot(normalize(signal_mod))
    else
        signal_mod = NaN;
        y = NaN;
        opt_val_sort = NaN;
    end
    decompositionResults(currentBeat).sig = signal_mod;
    decompositionResults(currentBeat).kernels = y;
    decompositionResults(currentBeat).optVals = opt_val_sort;
end

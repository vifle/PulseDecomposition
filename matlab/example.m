clear all
close all
clc

%% settings
sampleRate = 2000;
numKernels = 3;
kernelTypes = 'RayleighGaussian';
method = 'generic';
normOut = false;
parameterList = {'b_a','T1'};

%% include directories
addpath('helpFunctions')
addpath(genpath('Parameters'));
addpath(path);
dataDirectory = '..\data\';%directory where data folders are currently stored
currentPatient = 'example';%enter here a subject ID

%% load ppg data
data = readmatrix([dataDirectory,currentPatient,'.txt']);
ppgSignal = data(:,2);
ppgSignalTime = data(:,1);

% find beats
[ppgDetections] = pqe_beatDetection_lazaro( ppgSignal', sampleRate, 0 );%find peaks (maximum slopes)

%% segment into beats
% can is use old version of create single beats?
[singleBeatsProcessed,singleBeats,importantPoints] = createSingleBeats(ppgSignal,sampleRate,ppgDetections);

%% decompose beats & calculate parameters
decompositionResults = struct;
parameters = cell(numel(singleBeatsProcessed),numel(parameterList));
for currentBeat = 1:numel(singleBeatsProcessed)
    if (~isnan(singleBeatsProcessed{currentBeat}))
        [signal_mod,y,opt_val_sort] = decompositionAlgorithm(singleBeatsProcessed{currentBeat},sampleRate,...
            'numKernels',numKernels,'kernelTypes',kernelTypes,'method',method,'normalizeOutput',normOut);
        [parameters{currentBeat,1:numel(parameterList)}] = calculateParameter(signal_mod,singleBeatsProcessed{currentBeat},y,opt_val_sort,[kernelTypes,numKernels],sampleRate,parameterList);
    else
        signal_mod = NaN;
        y = NaN;
        opt_val_sort = NaN;
    end
    decompositionResults(currentBeat).sig = signal_mod;
    decompositionResults(currentBeat).kernels = y;
    decompositionResults(currentBeat).optVals = opt_val_sort;
end

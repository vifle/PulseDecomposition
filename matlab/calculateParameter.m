function[varargout] = calculateParameter(PPGmod,PPGbeat,y,opt_params,algorithmName,freq,parameterList)
% inputs:
% PPGmod            ...     PPG beat modeled by kernels
% PPGbeat           ...     beat of PPG signal that is to be decomposed
% y                 ...     shapes of kernels based on optimized parameters
% opt_params        ...     optimized parameters of the kernels
% algorithmName     ...     algorithm that was used for the decomposition
% freq              ...     sampling frequency of input signal
% parameterList     ...     array of parameters (cell array of strings)
%
% outputs:
% varargout         ...     array of calculated parameters (same order as
%                           specified input)

% do parameter check once for all inputs

% check if parameterList has right size
if(size(parameterList,2) > size(parameterList,1))
    parameterList = parameterList';
end


% iterate through array and call all functions needed
varargout = cell(1,size(parameterList,1));
for actualParameter = 1:size(parameterList,1)
    % check if parameter functions exist
    if(exist(['calculate_' parameterList{actualParameter,1}],'file')==2)
        varargout{1,actualParameter} = feval(['calculate_' parameterList{actualParameter,1}],...
            PPGmod, PPGbeat, y, opt_params, algorithmName, freq);
    else
        warning(['There is no function ''calculate_',parameterList{actualParameter,1},'''. Output set to NaN.'])
        varargout{1,actualParameter} = NaN;
    end
end

end
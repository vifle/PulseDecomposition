function [a] = calculate_a(PPGmod,PPGbeat,y,opt_params,algorithmName,freq)
% input:
% PPGmod            ...     PPG beat modeled by kernels
% PPGbeat           ...     beat of PPG signal that is to be decomposed
% y                 ...     shapes of kernels based on optimized parameters
% opt_params        ...     optimized parameters of the kernels
% algorithmName     ...     algorithm that was used for the decomposition
% freq              ...     sampling frequency of input signal
%
% outputs:
% a                 ...     amplitude of a wave of the second derivative of 
%                           a PPG beat

%% exceptions
if(any(isnan(PPGmod)))
    a = NaN;
    return
end

%% calculate b over a
second_deriv = deriv2(PPGmod);
a = max(second_deriv); % find a
b = min(second_deriv); % find b

%% verification
% b should come after a
t = 0:1/freq:(length(second_deriv)-1);
t_a = t(second_deriv==a);
t_a = t_a(1);
t_b = t(second_deriv==b);
t_b = t_b(1);
if(t_a > t_b)
    a = NaN;
end

end
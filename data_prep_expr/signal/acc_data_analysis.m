load acc_data
Fs = 10000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = size(acc_data, 1);             % Length of signal
t = (0:L-1)*T;        % Time vector
S = acc_data; % Signal

plot(t,acc_data)
[s, w, tt, ff, ps ] = spectrogram(S);

spectrogram(S, 'yaxis')


% X = S;
% Y = fft(X);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
% 
% figure;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')

% Lab 8
%% Task 1 - Lowpass FIR/IIR comparison
fs=8000;
t=0:1/fs:1;
signal=sin(2*pi*500*t)+sin(2*pi*2000*t);
noise=0.5*randn(size(t));
signal=signal+noise;
filtered_FIR=filter(Num,1,signal);
filtered_IIR=filtfilt(SOS,G,signal);
figure;
subplot(311);plot(t(1:100),signal(1:100));grid on;
title('Original signal');xlabel('Time (s)');ylabel('Amplitude');

subplot(312);plot(t(1:100),filtered_FIR(1:100));grid on;
title('Filtered signal (FIR)');xlabel('Time (s)');ylabel('Amplitude');

subplot(313);plot(t(1:100),filtered_IIR(1:100));grid on;
title('Filtered signal (IIR)');xlabel('Time (s)');ylabel('Amplitude');

sgtitle('Comparison of FIR and IIR filtering');

%Compute FFT
N=length(signal);
f=linspace(0,fs/2,N/2);
X_original=abs(fft(signal));
X_FIR=abs(fft(filtered_FIR));
X_IIR=abs(fft(filtered_IIR));

%Normalize and keep only positive frequencies
X_original=X_original(1:N/2)/max(X_original);
X_FIR=X_FIR(1:N/2)/max(X_FIR);
X_IIR=X_IIR(1:N/2)/max(X_IIR);

figure;
subplot(311);plot(f,X_original);grid on;
title('Frequency response of original signal');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(312);plot(f,X_FIR);grid on;
title('Frequency response of filtered signal (FIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(313);plot(f,X_IIR);grid on;
title('Frequency response of filtered signal (IIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

sgtitle('Comparison of frequency responses');

%% Task 2 - Highpass FIR/IIR comparison
fs=8000;
t=0:1/fs:1;
signal=sin(2*pi*500*t)+sin(2*pi*2500*t);
noise=0.5*randn(size(t));
signal=signal+noise;
filtered_FIR=filter(Num1,1,signal);
filtered_IIR=filtfilt(SOS1,G1,signal);
figure;
subplot(311);plot(t(1:100),signal(1:100));grid on;
title('Original signal');xlabel('Time (s)');ylabel('Amplitude');

subplot(312);plot(t(1:100),filtered_FIR(1:100));grid on;
title('Filtered signal (FIR)');xlabel('Time (s)');ylabel('Amplitude');

subplot(313);plot(t(1:100),filtered_IIR(1:100));grid on;
title('Filtered signal (IIR)');xlabel('Time (s)');ylabel('Amplitude');

sgtitle('Comparison of FIR and IIR filtering');

%Compute FFT
N=length(signal);
f=linspace(0,fs/2,N/2);
X_original=abs(fft(signal));
X_FIR=abs(fft(filtered_FIR));
X_IIR=abs(fft(filtered_IIR));

%Normalize and keep only positive frequencies
X_original=X_original(1:N/2)/max(X_original);
X_FIR=X_FIR(1:N/2)/max(X_FIR);
X_IIR=X_IIR(1:N/2)/max(X_IIR);

figure;
subplot(311);plot(f,X_original);grid on;
title('Frequency response of original signal');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(312);plot(f,X_FIR);grid on;
title('Frequency response of filtered signal (FIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(313);plot(f,X_IIR);grid on;
title('Frequency response of filtered signal (IIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

sgtitle('Comparison of frequency responses');

%% Task 3 - Bandpass FIR/IIR comparison
fs=8000;
t=0:1/fs:1;
signal = sin(2*pi*100*t) + sin(2*pi*800*t) + sin(2*pi*370*t);
noise=0.5*randn(size(t));
signal=signal+noise;
filtered_FIR=filter(Num2,1,signal);
filtered_IIR=filtfilt(SOS2,G2,signal);
figure;
subplot(311);plot(t(1:100),signal(1:100));grid on;
title('Original signal');xlabel('Time (s)');ylabel('Amplitude');

subplot(312);plot(t(1:100),filtered_FIR(1:100));grid on;
title('Filtered signal (FIR)');xlabel('Time (s)');ylabel('Amplitude');

subplot(313);plot(t(1:100),filtered_IIR(1:100));grid on;
title('Filtered signal (IIR)');xlabel('Time (s)');ylabel('Amplitude');

sgtitle('Comparison of FIR and IIR filtering');

%Compute FFT
N=length(signal);
f=linspace(0,fs/2,N/2);
X_original=abs(fft(signal));
X_FIR=abs(fft(filtered_FIR));
X_IIR=abs(fft(filtered_IIR));

%Normalize and keep only positive frequencies
X_original=X_original(1:N/2)/max(X_original);
X_FIR=X_FIR(1:N/2)/max(X_FIR);
X_IIR=X_IIR(1:N/2)/max(X_IIR);

figure;
subplot(311);plot(f,X_original);grid on;
title('Frequency response of original signal');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(312);plot(f,X_FIR);grid on;
title('Frequency response of filtered signal (FIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(313);plot(f,X_IIR);grid on;
title('Frequency response of filtered signal (IIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

sgtitle('Comparison of frequency responses');


% Lab 9
%% Task 1 
a=audioinfo('omni_drum.mp3');
disp(a);
whos a;

a.Title='My Song';
a.Rating=[10, 5];
mystructure=a;
mystructure.b=5;

sampleRate=a.TotalSamples/a.Duration;
disp(['Calculated Sample Rate:', num2str(sampleRate)]);

%% Task 2 - creating sinewave
fs=3000;
t=0:1/fs:2;
f=1000;
A=.5;
w=15*pi/180;
y=A*sin(2*pi*f*t+w);
sound(y,fs,16);
plot(t,y);xlabel('Time (s)');ylabel('Amplitude');
title('Sinewave with a frequency of 1000 Hz, amplitude of 0.2 and phase of 15 degree');

%% Task 3 - Phase cancellation
fs=44100;
f1=440;
f2=440;
A1=.3;
A2=.3;
t=0:1/fs:5;
w1=0*pi/180;
w2=180*pi/180;
y1=A1*sin(2*pi*f1*t+w1);
y2=A2*sin(2*pi*f2*t+w2);
y=(y1+y2)/2;
sound(y,fs,16);
plot(t,y);title('Phase cancellation');xlabel('Time (s)');ylabel('Amplitude');

% Change one frequency to 880 Hz
f2=880;
y2=A2*sin(2*pi*f2*t+w2);
y=(y1+y2)/2;
sound(y,fs,16);
figure;
subplot(221);plot(t,y);
title('Changed frequency to 880 Hz');xlabel('Time (s)');ylabel('Amplitude');

% Change one frequency to 441 Hz
f2=441;
y2=A2*sin(2*pi*f2*t+w2);
y=(y1+y2)/2;
sound(y,fs,16);
subplot(222);plot(t,y);
title('Changed frequency to 441 Hz');xlabel('Time (s)');ylabel('Amplitude');

% Phase changed to 179 degrees
w2=179*pi/180;
y2=A2*sin(2*pi*f2*t+w2);
sound(y,fs,16);
subplot(223);plot(t,y);
title('Phase changed to 179 degrees');xlabel('Time(s)');ylabel('Amplitude');

% Phase changed to 181 degrees
w2=181*pi/180;
y2=A2*sin(2*pi*f2*t+w2);
sound(y,fs,16);
subplot(224);plot(t,y);
title('Phase changed to 181 degrees');xlabel('Time(s)');ylabel('Amplitude');

%% Task 4 - Binaural beats/Cocktail party effect
fs=44100;
t=0:1/fs:5;
f1=300;
f2=310;
w=0*pi/180;
A=.5;
y1=A*sin(2*pi*f1*t+w);
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

% Change frequency difference to 2 Hz
f2=f1+2;
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

% Play 1000 Hz and 1010 Hz
f1=1000;
f2=1010;
y1=A*sin(2*pi*f1*t+w);
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

% Create cocktail party effect
f1=300;
f2=330;
y1=A*sin(2*pi*f1*t+w);
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

%% Task 5 - Complex tones via additive synthesis
fs=44100;
t=0:1/fs:5;
f1=400; f2=2*f1; f3=3*f1; f4=4*f1;
A1=.3; A2=A1/2; A3=A1/3; A4=A1/4;
w=0;
y1=A1*sin(2*pi*f1*t+w);
y2=A2*sin(2*pi*f2*t+w);
y3=A3*sin(2*pi*f3*t+w);
y4=A4*sin(2*pi*f4*t+w);
y=(y1+y2+y3+y4)/4;
sound(y,fs,16);

%% Task 6 - Melodies
fs=44100;
t=0:1/fs:5;
A1=.3; A2=A1/2; A3=A1/3; A4=A1/4;
f1=440; f2=f1*2; f3=f1*3; f4=f1*4;
w=0;
y1=A1*sin(2*pi*f1*t+w);
y2=A2*sin(2*pi*f2*t+w);
y3=A3*sin(2*pi*f3*t+w);
y4=A4*sin(2*pi*f4*t+w);
y=[y1 y2 y3 y4];
soundsc(y,fs);

x=y(1:2:end);
soundsc(x,fs);

%% Task 6 - Melodies 2nd way
fs=44100;
t=0:1/fs:5;
notes=[440, 494, 523, 587];
melody=[];
for f=notes
    melody=[melody, sin(2*pi*f*t)];
end
soundsc(melody,fs);
x=melody(1:2:end);
soundsc(x,fs);

%% Task 7 - Modulation
fs=44100;
t=0:1/fs:5;
f=440;
y=sin(2*pi*f*t);
% frequency modulation
fm=modulate(y,20,fs,'fm');
soundsc(fm,fs);
% amplitude modulation
am=modulate(y,.5,fs,'am');
soundsc(am,fs);
% phase modulation
pm=modulate(y,20,fs,'pm');
soundsc(pm,fs);

%% Task 8 - FFT
fs=44100;
t=0:1/fs:5;
f=440;
y=sin(2*pi*f*t);
N=fs;
Y=fft(y,N)/N;
magTransform=abs(Y);
faxis=linspace(-fs/2,fs/2,N);
plot(faxis, fftshift(magTransform));
xlabel('Frequency (Hz)');
axis([0 2000 0 max(magTransform)]);

%% Task 9 - Spectrogram
fs=44100;
t=0:1/fs:5;
f=440;
y=sin(2*pi*f*t);
win=128;
hop=win/2;
nfft=win;
spectrogram(y,win,hop,nfft,fs,'yaxis');

% quadratic chirp
c=chirp(t,0,1,440,'q');
spectrogram(c,win,hop,nfft,fs,'yaxis');

%% Task 10 - Shephard Tones
fs=16000;
d=1;
t=0:1/fs:d-1/fs;
fmin=300;
fmax=3400;
n=12;
l=mod(([0:n-1]/n)'*ones(1,fs*d)+ones(n,1)*(t/(d*n)),1);
f=fmin*(fmax/fmin).^l;
p=2*pi*cumsum(f,2)/fs;
p=diag((2*pi*floor(p(:,end)/(2*pi)))./p(:,end))*p;
s=sin(p);
a=0.5-0.5*cos(2*pi*l);
w=sum(s.*a)/n;
w=repmat(w,1,3);
specgram(w,2048,fs,2048,1800);
ylim([0 4000]);
soundsc(w,fs);
audiowrite('shephard.wav',w,fs);

%% Task 11 - Amplitude modulation and Ring modulation
load handel;
index=1:length(y);
Fc=5;
A=.5;
w=0;
trem=(w*pi/180+A*sin(2*pi*index*(Fc/Fs)))';
y=y.*trem;
soundsc(y,Fs);

Fc=1000;
trem=(w*pi/180+A*sin(2*pi*index*(Fc/Fs)))';
y=y.*trem;
soundsc(y,Fs);

%% Task 12 - Filter design
fs=44100;
t=0:1/fs:5;
y=sin(2*pi*400*t)+sin(2*pi*2500*t);
noise=0.5*randn(size(t));
y=y+noise;
% Lowpass filter cutoff 442 order 10
cutoff=442/(fs/2);
order=10;
d=designfilt('lowpassfir','CutoffFrequency',cutoff,'FilterOrder',order);
o=filter(d,y);
soundsc(o,fs);

figure;
subplot(221);plot(y(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Original signal (first 800 samples)');

subplot(222);plot(o(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Filtered signal (Lowpass, cutoff 442 Hz, order 10)');

% Change cutoff and filter order
cutoff=600/(fs/2);
order=20;
d=designfilt('lowpassfir','CutoffFrequency',cutoff,'FilterOrder',order);
o2=filter(d,y);
soundsc(o2,fs);

subplot(223);plot(o2(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Filtered signal (Lowpass, cutoff 600 Hz, order 20)');

% Highpass filter cutoff 442 order 10
cutoff=442/(fs/2);
order=10;
d_high=designfilt('highpassfir','CutoffFrequency',cutoff,'FilterOrder',order);
o_high=filter(d_high,y);
soundsc(o_high,fs);

subplot(224);plot(o_high(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Filtered signal (Highpass, cutoff 442 Hz, order 10)');

%% Convolution Reverb
[ir, fs_ir]=audioread('omni_drum.mp3');
[x, fs_x]=audioread('singing.wav');

if fs_ir~=fs_x
    ir=resample(ir, fs_x, fs_ir);
end

fs=fs_x;
y=conv(x, ir);
soundsc(y, fs);

figure;
subplot(211);plot(y, 'b');hold on;plot(x, 'k');
title('Impulse response of reverberated signal');
xlabel('Time (samples)');ylabel('Amplitude');legend('Reverberated', 'Original');

subplot(212);plot(ir);
title('Impulse response'); xlabel('Sample index');ylabel('Amplitude');


% Lab 10
%% Task 1
f=45;
fs=100;
duration=2;
t=0:1/fs:duration;
x=sin(2*pi*f*t);
echo_amplitude=.5;
echo_delay=.2;
delay_samples=round(echo_delay*fs);
echo=[zeros(1,delay_samples), x*echo_amplitude];
x_original=[x,zeros(1,delay_samples)];
y=x_original+echo;

sound(y,8000);
pause(duration+echo_delay+0.5);

figure;
subplot(211);plot((0:length(y)-1)/fs,y);
title('Signal with echo');xlabel('Time (s)');ylabel('Amplitude');

Y=abs(fft(y));
faxis=linspace(0,fs,length(Y));
subplot(212);plot(faxis,Y);
title('Signal with echo (frequency domain)');xlabel('Frequency (Hz)');ylabel('Magnitude');

% Compute complex cepstrum
[xhat, nd]=cceps(y);

figure;
plot((0:length(xhat)-1)/fs,xhat);
title('Complex cepstrum of Signal with echo');xlabel('Quefrency (s)');ylabel('Amplitude');

% Remove echo
cepstrum_cleaned=xhat;
cepstrum_cleaned(delay_samples)=0;

y_clean=icceps(cepstrum_cleaned, nd);

sound(real(y_clean), 8000);
pause(duration+echo_delay+0.5);

figure;
subplot(211);plot((0:length(y_clean)-1)/fs,real(y_clean));
title('Signal after echo removal');xlabel('Time (s)');ylabel('Amplitude');

Y_clean=abs(fft(real(y_clean)));
faxis=(0,fs,length(Y_clean));
subplot(212);plot(faxis,Y_clean);
title('Signal after echo removal in frequency domain');xlabel('Frequency (Hz)');ylabel('Magnitude');


%% 1.2
[x2 Fs2]=audioread('singing.wav');
x2=x2(:,1); % use only one channel if stereo
echo_amplitude=0.4;
echo_delay=0.3;
delay_samples=round(echo_delay*Fs2);
echo=[zeros(1,delay_samples), x2*echo_amplitude];
x2_original=[x2, zeros(1,delay_samples)];
y2=x2_original+echo;

sound(y2,8000);
pause(echo_delay+duration+0.5);

figure;
subplot(211);plot((0:length(y2)-1)/Fs2, y2);
title('Signal with echo');xlabel('Time (s)');ylabel('Amplitude');

Y2=abs(fft(y2));
faxis2=linspace(0,Fs2,length(Y2));
subplot(212);plot(faxis2,Y2);
title('Signal with echo in frequency domain');xlabel('Frequency (Hz)');ylabel('Magnitude');

[xhat2 nd2]=cceps(y2);

figure;
plot((0:length(xhat2)-1)/Fs2,xhat2);
title('Complex cepstrum of signal with echo');xlabel('Quefrency (s)');ylabel('Amplitude');

cepstrum_cleaned2=xhat2;
cepstrum_cleaned2(delay_samples)=0;

y_clean2=icceps(cepstrum_cleaned2, nd2);
sound(real(y_clean2),8000);
pause(duration+echo_delay+0.5);

figure;
subplot(211);plot((0:length(y_clean2)-1)/Fs2,real(y_clean2));
title('Signal after echo removal');xlabel('Time (s)');ylabel('Amplitude');

Y_clean2=abs(fft(real(y_clean2)));
faxis2=linspace(0, Fs2, length(Y_clean2));
subplot(212);plot(faxis2,Y_clean2);
title('Signal after echo removal in frequency domain');xlabel('Frequency (Hz)');ylabel('Magnitude');

%% task 1.2 
clc;
% --- Load another audio signal ---
[x2, Fs2] = audioread('singing.wav');  % load signal
x2 = x2(:,1);  % use only one channel if stereo

% Play original with echo
sound(x2, Fs2);
pause(length(x2)/Fs2 + 0.5);

% --- Plot BEFORE Echo Removal ---
figure;
subplot(2,1,1);
plot((0:length(x2)-1)/Fs2, x2);
title('Echoed Signal (Time Domain)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
Y2_fft = abs(fft(x2));
N = length(Y2_fft);
f_axis2 = linspace(0, Fs2/2, floor(N/2));  % frequency axis up to Fs/2
plot(f_axis2, Y2_fft(1:floor(N/2)));
title('Echoed Signal (Frequency Domain)');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

% --- Complex cepstrum analysis ---
[xhat2, nd2] = cceps(x2);

% Zero out echo region in cepstrum (around 0.3s)
cepstrum_cleaned2 = xhat2;
region2 = round(0.3 * Fs2);
cepstrum_cleaned2(region2-2:region2+2) = 0;

% Inverse complex cepstrum
y2_clean = icceps(cepstrum_cleaned2, nd2);

% Play denoised signal
sound(real(y2_clean), Fs2);
pause(length(y2_clean)/Fs2 + 0.5);

% --- Plot AFTER Echo Removal ---
figure;
subplot(2,1,1);
plot((0:length(y2_clean)-1)/Fs2, real(y2_clean));
title('Filtered Audio After Echo Removal (Time Domain)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
Y2_clean_fft = abs(fft(real(y2_clean)));
N2 = length(Y2_clean_fft);
f_axis_clean = linspace(0, Fs2/2, floor(N2/2));
plot(f_axis_clean, Y2_clean_fft(1:floor(N2/2)));
title('Filtered Audio After Echo Removal (Frequency Domain)');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

%% Task 2
load mtlb;

sound(mtlb, Fs);
pause(length(mtlb)/Fs+0.5);

segmentlen=100;
noverlap=90;
NFFT=128;

spectrogram(mtlb, segmentlen, noverlap, NFFT, Fs, 'yaxis');

dt=1/Fs;
I0=round(0.1/dt);
Iend=round(0.25/dt);
x=mtlb(I0:Iend);

sound(x, Fs);
pause(length(x)/Fs + 0.5);

c=cceps(x);

t=0:dt:(length(x)-1)*dt;

trng=t(t>=2e-3 & t<=10e-3);
crng=c(t>=2e-3 & t<=10e-3);

[~,I]=max(crng);

fprintf('Complex cepstrum F0 estimate is %3.2f Hz.\n',1/trng(I));

plot(trng*1e3, crng)
xlabel('ms')
hold on
plot(trng(I)*1e3,crng(I),'ro','MarkerFaceColor','r')
hold off

[b0, a0]=butter(2, 325/(Fs/2));
xin=abs(x);
xin=filter(b0,a0,xin);
xin=xin-mean(xin);
x2=zeros(length(xin),1);
x2(1:length(x)-1)=xin(2:length(x));
zc=length(find((x2>0 & xin<0) | (x2<0 & xin>0)));
F0=0.5*Fs*zc/length(x);
fprintf('Zero crossing F0 estimate is %3.2f Hz.\n',F0);

%% Task 3
I=imread('AT3_1m4_01.tif');
imshow(I);

I=im2double(I);
I=log(1+I);

N=2*size(I,1)+1;
M=2*size(I,2)+1;

sigma=10;

[X,Y]=meshgrid(1:N, 1:M);
centerX=ceil(N/2);
centerY=ceil(M/2);
gaussianNumerator=(X-centerX).^2+(Y-centerY).^2;
H=exp(-gaussianNumerator./(2*sigma.^2));
H=1-H;

figure;
imshow(H, 'InitialMagnification', 25);

H=fftshift(H);
If=fft2(I, M, N);
Iout=real(ifft2(H.*If));
Iout=Iout(1:size(I,1),1:size(I,2));

Ihmf=exp(Iout)-1;

imshowpair(I, Ihmf, 'montage');
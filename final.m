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



Lecture 7
1. Analog vs. Digital Filters
The lecture starts by comparing a six-pole Chebyshev analog filter with a 129-point windowed-sinc digital filter, both designed for a low-pass cutoff at 1 kHz.


Performance Comparison:


Frequency Response & Passband: The analog filter shows a 6% ripple in its passband, which is a variation in the signal's amplitude. In contrast, the digital filter's passband is almost perfectly flat (within 0.02%). The flatness of digital filters is limited by round-off error, making them significantly flatter than analog filters, which are limited by the precision of their physical components like resistors and capacitors.




Roll-off and Stopband: When looking at the frequency response on a logarithmic scale, the digital filter demonstrates a much sharper "roll-off" (the transition from passband to stopband) and superior "stopband attenuation" (blocking of unwanted frequencies). Improving these aspects in an analog filter is difficult, whereas it can be achieved with simple modifications in a digital filter.




Step Response & Phase: The digital filter has a symmetrical step response, which indicates it has a linear phase. The analog filter's step response is not symmetrical, meaning it has a nonlinear phase.


Inherent Advantages of Analog Filters:
Despite the performance advantages of digital filters, analog filters are necessary in certain situations due to two main advantages:


Speed: Analog circuits are significantly faster. An operational amplifier (op-amp) can operate at speeds 10 to 100 times faster than a digital system processing data with FFT convolution.



Dynamic Range: Analog systems have a much larger dynamic range.


Amplitude: A standard op-amp has a dynamic range of about ten million, whereas a 12-bit Analog-to-Digital Converter (ADC) has a dynamic range of about 14,000.


Frequency: It is easier for analog circuits to handle a wide range of frequencies simultaneously. For a digital system to capture a low frequency like 0.01 Hz while sampling at 200 kHz, it would need to process an enormous number of data points (20 million)




Lecture 8

1. The Mechanics of Human Hearing
The process of hearing involves the outer, middle, and inner ear working together to transmit sound to the brain.


Outer and Middle Ear:

The outer ear, consisting of the visible part (pinna) and the ear canal, directs sound waves to the tympanic membrane (eardrum), causing it to vibrate.


The middle ear acts as an "impedance matching network". Sound waves struggle to pass from air (low impedance) to the liquid-filled inner ear (high impedance), with most energy reflecting away.


To overcome this, the middle ear uses the difference in area between the large eardrum (about 60 mm²) and the small oval window (about 4 mm²) to increase the sound wave's pressure by approximately 15 times, ensuring more sound energy enters the inner ear.

Inner Ear (Cochlea):

The cochlea is a liquid-filled, snail-shaped tube that contains the basilar membrane.


The basilar membrane functions as a frequency spectrum analyzer. Different parts of the membrane resonate at different frequencies.

High-frequency sounds cause vibrations near the oval window, while low-frequency sounds cause vibrations at the far end. This is known as the 

place principle, where specific nerve fibers respond to specific frequencies.

2. How the Brain Interprets Sound
The brain uses multiple schemes to encode and perceive sound.


The Volley Principle: This is another method for encoding audio information, particularly for frequencies below 4 kHz.


A single neuron can fire in response to each cycle of a sound wave up to about 500 Hz.

For higher frequencies, groups of neurons take turns firing, allowing the brain to perceive frequencies up to about 4,000 Hz.

Loudness and Decibels (dB):

Sound intensity is measured on a logarithmic scale called decibels (dB SPL). 0 dB SPL is the weakest sound a human can typically hear, while pain occurs around 140 dB SPL. Normal conversation is about 60 dB SPL.





The human hearing range is about 120 dB, a one-million-fold difference in amplitude. We can distinguish about 120 different levels of loudness.


Our perception of loudness is not linear. It relates to sound power by an exponent of 1/3. This means that to cut the 

perceived loudness by 80%, you must eliminate 99% of the sound power.

3. Directional Hearing
Having two ears provides the ability to identify the direction of a sound. This is achieved in two ways:


Shadowing: For frequencies above 1 kHz, the head casts a "sound shadow," making the signal stronger in the ear nearer to the source.



Time Delay: The ear farther from the sound source hears the sound slightly later because of the greater distance the sound must travel.

Interestingly, when identical monaural sound is played through headphones, the brain cannot find any directional cues and concludes the sound is coming from inside the listener's head.

4. The Qualities of Sound: Loudness, Pitch, and Timbre
A continuous sound is perceived in three parts:


Loudness: The intensity of the sound wave.


Pitch: The fundamental frequency of the sound.


Timbre: The quality or character of the sound, determined by its harmonic content.


The lecture emphasizes that 

human hearing is very insensitive to the phase of a sound wave. Two waveforms with the same frequency components but different phases (and therefore different shapes) will sound identical. This is because as sound reflects off surfaces in an environment, the relative phases of its frequencies become randomized anyway.





Timbre arises from the presence and relative amplitudes of harmonics (multiples of the fundamental frequency). A violin and a piano playing the same note (e.g., A at 220 Hz) have the same pitch, but their different harmonic structures give them a different timbre.



5. Music and the Logarithmic Scale
The standard musical scale is built upon the relationship between fundamental frequencies and their harmonics.


The Piano Keyboard: The note 'A' repeats across the keyboard at frequencies that are multiples of each other (e.g., 27.5 Hz, 55 Hz, 110 Hz, etc.). Because these notes are harmonics of one another, they sound similar.


Octaves: An octave represents a doubling of frequency. On a piano, this occurs over eight white keys. Since octaves are based on a repeated doubling of frequency, they represent a logarithmic scale.


Sampling Rate: The fact that audio information is distributed logarithmically is important and directly impacts the required sampling rate for digital audio signals.


Lecture 9

1. The Trade-off: Sound Quality vs. Data Rate
When designing a digital audio system, one must balance sound quality with the data rate that can be tolerated. This leads to three main categories:




High-Fidelity Music: The primary goal is the best possible sound quality, and a high data rate is acceptable.


Example (CD): Uses a 44.1 kHz sampling rate and 16-bit precision, resulting in a data rate of 706k bits/sec, which is considered better than human hearing.


Telephone Communication: The goal is natural-sounding speech at a low data rate to minimize system cost.


Example: Uses an 8 kHz sampling rate and 8-bit precision (with companding), for a data rate of 64k bits/sec. This is good for speech but poor for music.






Compressed Speech: Reducing the data rate is the most important factor, even if it means the sound is somewhat unnatural. This is used in applications like military communications and voice mail.


Example (Linear Predictive Coding): Can reduce the data rate to as low as 4k bits/sec, but with poor voice quality.


2. High-Fidelity Audio: The Compact Disc (CD)
High-fidelity audio systems are designed to exceed the limits of human hearing to ensure the reproduction is faithful to the original. The CD brought this technology to the masses.


How a CD Works:

Digital data is stored as a series of tiny dark "pits" burned onto a shiny, reflective surface.

An optical sensor reads this data by detecting the reflections as the disc spins.

The raw data is read at a rate of 4.3 million bits per second (Mbits/sec).

Encoding for Reliability:


Eight-to-Fourteen Modulation (EFM): Raw 8-bit chunks of data are converted into 14-bit codes before being stored on the disc. This special encoding ensures the physical pits are not too short or too long, which reduces errors during optical pickup. The system uses a look-up table to reverse the process during playback.




Reed-Solomon Coding: This is a method for error detection and correction. If errors are found, the system can correct them using redundant data, hide them by interpolating between samples, or mute them by setting the sample to zero.


These encoding schemes triple the amount of data stored on the disc compared to the raw audio signal.

Playback and Filtering:

To make the analog filtering process easier, CD players use a 

multirate technique called interpolation.


They increase the sampling rate by a factor of four (from 44.1 kHz to 176.4 kHz) by inserting zero-value samples and then using a digital filter to smooth the signal.

This pushes the unwanted frequencies created by the digital-to-analog conversion much higher, making them easier to remove with a simpler analog filter like a Bessel filter.

Stereo and Surround Sound:

Stereo systems use multiple channels (e.g., left and right) to provide a more realistic, three-dimensional sound experience.

Motion pictures often use four-channel systems like 

Dolby Stereo (or Dolby Surround Pro Logic for home use). These systems include left, right, center, and surround channels to anchor dialogue to the screen and create an immersive environment.





3. Companding: Reducing Data Rate for Speech
Companding is a technique used to reduce the data rate for audio signals by using unequally spaced quantization levels.


Why it Works: Human hearing perceives loudness logarithmically. We can distinguish small changes in quiet sounds, but we don't notice similarly sized changes in loud sounds. Companding matches this characteristic, using small steps for low-amplitude signals and large steps for high-amplitude signals.



Benefit: This allows telephone-quality speech to be represented with just 8 bits per sample instead of the 12 bits that would be required with linear quantization.


Algorithms: The two main companding algorithms are µ-law (used in North America and Japan) and A-law (used in Europe). Both use a logarithmic curve to compress the signal before quantization.


4. Speech Synthesis and Recognition
Speech Synthesis (Creating Speech):

This is based on a model of human speech production, which classifies sounds as either 

voiced (like vowels, produced by vocal cord vibrations) or fricative (like 's' and 'f', produced by air turbulence).



A synthesizer creates speech by selecting an excitation source (a pulse train for voiced sounds, noise for fricatives) and passing it through a digital filter that mimics the vocal tract.



By updating the parameters (pitch, filter coefficients) about 40 times per second, continuous speech can be generated. This was used in the classic "Speak & Spell" toy. This method, also the basis for 


Linear Predictive Coding (LPC), produces robotic-sounding speech but requires a very low data rate.


Speech Recognition (Understanding Speech):

Recognition algorithms typically work by extracting parameters from a speech signal and comparing them to stored templates of sounds to identify words.


The Main Challenge: This method doesn't work very well because human understanding relies heavily on context. We use our knowledge of the world and the sentence structure to understand words that might sound identical (e.g., "spider ring" vs. "spy during").





Common Limitations:

They often require distinct pauses between words.

Vocabularies are usually limited to a few hundred words.

They must be trained for each individual speaker.

Lecture 10

Of course. Here is a detailed explanation of the lecture on Nonlinear Audio Processing by Associate Professor Naila Allakhverdiyeva.

High-Level Summary
This lecture introduces audio processing techniques that go beyond standard linear filtering. It focuses on two main nonlinear methods used to solve problems where linear approaches fail. The first technique is a 

time-varying Wiener filter for reducing wideband noise in speech when the noise and signal occupy the same frequencies. The second is 


homomorphic signal processing, a powerful method that transforms signals combined nonlinearly (through multiplication or convolution) into a linear problem that can be solved with standard filtering techniques. The lecture details the mechanics, applications, and significant challenges of these advanced methods.

1. Reducing Wideband Noise with a Time-Varying Filter
This technique is designed to reduce noise like magnetic tape hiss, wind, or crowd noise from speech signals. Linear filtering is not effective in these cases because the noise and the voice signal completely overlap in the frequency domain.




The Core Concept: The method works by analyzing very short segments of the audio (e.g., 16 milliseconds).

Within a short segment, the speech signal's energy is concentrated in a few large-amplitude frequencies.


Wideband noise, in contrast, is distributed more uniformly at a low amplitude across the spectrum.

By examining the amplitude of each frequency, a determination can be made:


Large amplitude: Likely mostly signal, so it should be kept.


Small amplitude: Likely mostly noise, so it should be discarded (set to zero).


Mid-size amplitude: Adjusted smoothly between the two extremes.


A Time-Varying Wiener Filter: This technique can be thought of as a Wiener filter whose frequency response is not fixed. Instead, the filter is continuously recalculated for each new segment of the signal, adapting its characteristics based on the spectrum of that specific segment.



Implementation Note: Standard filtering methods like overlap-add are not valid for this nonlinear process because the changing filter response would cause misalignment between processed segments. To overcome this, the signal is broken into overlapping segments, and a smooth window is applied to each segment after processing, ensuring a smooth transition when they are recombined.


2. Homomorphic Signal Processing
Homomorphic processing is a technique used to separate signals that have been combined in a nonlinear way, such as through multiplication or convolution. The goal is to convert the problem into a linear one so that standard filtering can be used.



For Multiplied Signals (e.g., Automatic Gain Control):


Problem: An audio signal a[] is multiplied by a slowly changing gain signal g[] (e.g., in AM radio).

Solution:

Logarithm: Apply a logarithm to the combined signal a[] × g[]. This turns the multiplication into addition: 

log a[] + log g[].



Linear Filter: Use a high-pass filter to remove the low-frequency log g[] component, leaving log a[].



Anti-Logarithm: Apply the exponential function (e 
x
 ) to reverse the logarithm, recovering the original signal a[].

For Convolved Signals (e.g., Echo Removal):


Problem: An audio signal x[] is convolved with an echo y[].

Solution:


Homomorphic Transform: First, a Fourier Transform turns the convolution into multiplication (X[] × Y[]). Then, a logarithm turns that multiplication into addition (

log X[] + log Y[]).


Linear Filter: A linear filter is applied to separate the two added components.


Inverse Transform: The process is reversed with an anti-logarithm and an inverse Fourier transform to recover the original signal x[].

The Cepstrum and "Liftering":

When processing convolved signals, the domains are swapped: linear filtering happens in a frequency-like domain. This has led to playful jargon.


Cepstrum: Defined as the inverse Fourier transform of the log-magnitude of the Fourier transform of a signal. The term is a rearrangement of "spectrum".



Liftering: The act of filtering in the cepstral domain. Similarly, there are "long-pass" and "short-pass" filters instead of low-pass and high-pass.



Challenges of Homomorphic Processing
Despite its power, this technique has significant practical problems:


Complex Logarithm: Audio signals have both positive and negative values, requiring the use of the more advanced complex logarithm.


Aliasing: Applying a logarithm creates sharp corners in a waveform. To capture these without aliasing, the signal may need to be sampled at a much higher rate (e.g., 100 kHz instead of 8 kHz).




Spectral Overlap: The process of taking a logarithm creates many harmonics. Even if the original signals (e.g., two sine waves) did not overlap in frequency, their logged versions might, making complete separation with a linear filter impossible.



The lecture concludes with the key lesson from these techniques: the most effective way to process a signal is in a manner that is consistent with how it was originally formed.




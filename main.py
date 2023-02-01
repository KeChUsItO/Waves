import pandas as pd
import pydub
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import hashlib
from pydub.utils import mediainfo

pd.options.plotting.backend = "plotly"

FILE_NAME = "Bad Bunny - Un Coco (360° Visualizer) _ Un Verano Sin Ti (128 kbps)"


def normal_round(num, n_digits=0):
    if n_digits == 0:
        return int(num + 0.5)
    else:
        digit_value = 10 ** n_digits
        return int(num * digit_value + 0.5) / digit_value


def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    bitrate = int(int(mediainfo(f)['bit_rate']) / 1000)
    if normalized:
        audio = pd.DataFrame(np.float32(y) / 2 ** 15)
        audio.index /= a.frame_rate
        return a.frame_rate, bitrate, audio
    else:
        audio = pd.DataFrame(np.float32(y))
        audio.index /= a.frame_rate
        return a.frame_rate, bitrate, audio


def write(f, sr, x, br=1411, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate=f"{br}k")


def downsample(audio, sr_in, sr_out=44_100):
    if sr_in == sr_out:
        return audio, sr_out
    if sr_in < sr_out:
        return audio, sr_out
    time_len = len(audio) / sr_in
    if (time_len * sr_out) % 1 == 0:
        new_samples = int(time_len * sr_out)
    else:
        new_samples = int(time_len * sr_out) + 1
    diff = len(audio) - new_samples
    print(diff, len(audio))
    step = diff / len(audio)
    print(step * 100)
    return sr_out, audio


sr, br, x = read(f"{FILE_NAME}.mp3")

yf = fft(np.array(x[0]))
xf = fftfreq(len(x[0]), 1 / sr)[:len(x[0]) // 2]

plt.plot(xf, 2.0 / len(x) * np.abs(yf)[0:len(x) // 2])
plt.grid()
plt.show()


def fourier_to_audio(xf, amp, leng, sr):
    song_wave = np.array([0] * leng).astype(np.float32)
    for f_i, f in enumerate(xf):
        print(f / max(xf) * 100, "%")
        wave_lin = np.linspace(0.0, leng * 1 / sr, leng, endpoint=False)
        song_wave += amp[f_i] * np.sin(f * 2.0 * np.pi * wave_lin)
    write("Coco nueva.mp3", sr, song_wave)


hash_waves = hashlib.sha256(b"Waves").hexdigest()
hash_license = hashlib.sha256(b"Ibai Twitch").hexdigest()
hashes = [hash_waves, hash_license]
print(hashes)


def encode(wave, sr, br, hashes, file_name):
    frec = 20_002
    leng = len(wave)
    wave_lin = np.linspace(0.0, leng * 1 / sr, leng, endpoint=False)
    wave[0] += 20 * np.sin(frec * 2.0 * np.pi * wave_lin)
    print(frec)
    for hash in hashes:
        for car in hash:
            if car != ' ':
                frec += (int(car, 16) + 1)
            else:
                frec += 17
            if wave.shape[1] == 2:
                wave[0] += 20 * np.sin(frec * 2.0 * np.pi * wave_lin)
                wave[1] += 20 * np.sin(frec * 2.0 * np.pi * wave_lin)
            else:
                wave += 20 * np.sin(frec * 2.0 * np.pi * wave_lin)
        frec += 18


    wave = wave.astype(np.int16)
    yf = fft(np.array(wave[0]))
    xf = fftfreq(leng, 1 / sr)[:leng // 2]
    plt.plot(xf, 2.0 / leng * np.abs(yf)[0:leng // 2])
    plt.grid()
    plt.show()

    print(wave)

    write(f"{file_name}_encoded.mp3", sr, wave, 320)
    return wave


def decode(file_name, wave=None):
    sr, br, x = read(file_name)
    x = wave

    yf = fft(np.array(x[0]))
    yf = 2.0 / len(x) * np.abs(yf)[:len(x) // 2]
    xf = fftfreq(len(x), 1 / sr)[:len(x) // 2]
    print(xf)
    diff_array = np.absolute(xf - 20_000)
    index_slice = diff_array.argmin()

    yf = yf[index_slice:]
    xf = xf[index_slice:]



    print(np.mean(yf) + np.std(yf) * 3)
    characters = np.where(yf >= np.mean(yf) + 6 * np.std(yf))
    print(len(characters))
    prev = 20_002
    string = ''
    dec_hashes = []
    for char_ind, char in enumerate(np.array(characters).flatten('F')):
        if (normal_round((xf[char] - prev))) != 0:
            if normal_round((xf[char] - prev)) == 17:
                string += ' '
            elif normal_round((xf[char] - prev)) >= 18:
                dec_hashes.append(string)
                string = f'{(normal_round((xf[char] - prev) - 18 - 1)):x}'
            else:
                string += f'{(normal_round((xf[char] - prev) - 1)):x}'
            prev = normal_round(xf[char])
    dec_hashes.append(string)

    print(dec_hashes)


    plt.plot(xf, yf)
    plt.grid()
    plt.show()



enc_wave = encode(x, sr, br, hashes, FILE_NAME)
decode(f"{FILE_NAME}_encoded.mp3", enc_wave)
print(hashes)
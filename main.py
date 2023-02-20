import pandas as pd
import pydub
import numpy as np
from scipy.fft import fft, fftfreq
import hashlib
from pydub.utils import mediainfo
import os
import matplotlib.pyplot as plt
from time import perf_counter
import scipy
from functools import wraps
import time
from numba import njit, vectorize, jit

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.options.plotting.backend = "plotly"

FILE_NAME = "La Jumpa"


def normal_round(num, n_digits=0):
    if n_digits == 0:
        return int(num + 0.5)
    else:
        digit_value = 10 ** n_digits
        return int(num * digit_value + 0.5) / digit_value


def read(f, format='mp3', normalized=False):
    """MP3 to numpy array"""
    if format == 'mp3':
        a = pydub.AudioSegment.from_mp3(f)
    elif format == 'wav':
        a = pydub.AudioSegment.from_wav(f)
    elif format == 'flac':
        a = pydub.AudioSegment.from_file(f, 'flac')
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


def decode(file_name=None, wave=None, sr=None, show_plot=False):
    if file_name:
        sr, br, x = read(file_name, 'flac')
    elif type(wave) != type(None):
        sr = sr
        x = wave
    else:
        return

    yf = fft(np.array(x[:, 0]))
    yf = 2.0 / len(x) * np.abs(yf)[:len(x) // 2]
    xf = fftfreq(len(x), 1 / sr)[:len(x) // 2]
    diff_array = np.absolute(xf - 20_000)
    index_slice = diff_array.argmin()

    yf = yf[index_slice:]
    xf = xf[index_slice:]
    if show_plot:
        plt.plot(xf, yf)
        plt.grid()
        plt.show()
    characters = np.sort(np.argsort(yf)[-64:])
    prev = 20_002
    string = ''
    dec_hashes = []

    for char_ind, char in enumerate(np.array(characters).flatten('F')):
        if (normal_round((xf[char] - prev))) != 0:
            string += f'{(normal_round((xf[char] - prev) - 1)):x}'
            prev = normal_round(xf[char])
    dec_hashes.append(string)
    print(dec_hashes)


def fft_plot(wave, sr):
    print(wave.shape)
    leng = len(wave)
    yf = fft(wave[:, 0])
    xf = fftfreq(leng, 1 / sr)[:leng // 2]
    print(yf.shape, xf.shape)
    plt.plot(xf, 2.0 / leng * np.abs(yf)[:leng // 2])
    plt.grid()
    plt.show()


def write(f, sr, x, br=1411, normalized=False):
    print("WRITING", x.shape)
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="flac")


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


sr, br, x = read(f"{FILE_NAME}.wav", 'wav')
print(x.shape)
yf = fft(np.array(x[0]))
xf = fftfreq(len(x[0]), 1 / sr)[:len(x[0]) // 2]
'''
plt.plot(xf, 2.0 / len(x) * np.abs(yf)[0:len(x) // 2])
plt.grid()
plt.show()
'''


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def fourier_to_audio(xf, amp, leng, sr):
    song_wave = np.array([0] * leng).astype(np.float32)
    for f_i, f in enumerate(xf):
        print(f / max(xf) * 100, "%")
        wave_lin = np.linspace(0.0, leng * 1 / sr, leng, endpoint=False)
        song_wave += amp[f_i] * np.sin(f * 2.0 * np.pi * wave_lin)
    write("Coco nueva.mp3", sr, song_wave)


hash_waves = hashlib.sha256(b"Waves").hexdigest()
hash_license = hashlib.sha256(b"Waves Ibai Twitch La Jumpa").hexdigest()
hashes = [hash_license]
print(hashes)


def hex_to_int(v):
    print(v)
    v = v.view(np.uint8)[::4]
    print(v)
    return np.where(v > 64, v - 55, v - 48)


@vectorize
def aprox_sin_2pi(x):
    x = x - int(x)
    if (x < 0.5):
        return -16 * x ** 2 + 8 * x
    else:
        return 16 * x ** 2 - 24 * x + 8


@njit(fastmath=True)
def encode(wave, frecs, wave_lin):
    frec = 20_002
    for car in frecs:
        frec += car + 1
        wave += aprox_sin_2pi(frec * wave_lin).reshape((-1, 1))
    wave = wave / wave.max() * 32767
    return wave


def get_frecs(hash):
    ret = []
    for h in hash:
        ret.append(int(h, 16))
    return np.array(ret)


def clean_high_frecs(wave):
    if wave.shape[1] == 2:
        xf = fftfreq(len(wave[0]), 1 / sr)
        yf_o = fft(np.array(wave[0]))
        yf_1 = fft(np.array(wave[1]))
        o_indexes = np.where(np.logical_or(xf < -20_000, xf > 20_000))
        yf_o[o_indexes] = 0
        yf_1[o_indexes] = 0
        wave = pd.DataFrame({0: scipy.fft.ifft(yf_o), 1: scipy.fft.ifft(yf_1)})
    else:
        xf = fftfreq(len(wave), 1 / sr)
        yf = fft(np.array(wave))
        o_indexes = np.where(np.logical_or(xf < -20_000, xf > 20_000))
        yf[o_indexes] = 0
        wave = scipy.fft.ifft(yf)
    return wave


wave_x = clean_high_frecs(x)
print("DUR:", (len(wave_x) * 1 / sr))
for i in range(1):
    beg = perf_counter()
    frecs = get_frecs(hash_license)
    lin_space = np.linspace(0.0, len(wave_x) * 1 / sr, len(wave_x), endpoint=False)
    new_wave = encode(np.array(wave_x), frecs, lin_space)
    end = perf_counter()
    print(end - beg)
    print(normal_round((end - beg) / (len(wave_x) * 1 / sr), 3))
beg = perf_counter()
decode(wave=new_wave, sr=sr, show_plot=False)
end = perf_counter()
print(end - beg)
beg = perf_counter()
write(f"{FILE_NAME}_encoded.flac", sr, pd.DataFrame(new_wave))
end = perf_counter()
print(end - beg)
fft_plot(new_wave, sr)


def encode_parts(wave, sr, hashes, file_name, time_part, show_plot=False, do_write=True):
    leng = len(wave)
    wave_lin = np.arange(0.0, leng * 1 / sr, leng)
    amp = 1
    cells = np.floor(leng / (sr * time_part)).astype(np.int64)
    wave_lin_splits = np.zeros(())
    wave_split = np.arange(cells)
    print(cells)
    print(np.array(np.array_split(np.array(wave), leng / (sr * time_part))).shape)
    print(np.array(np.array_split(np.array(wave), leng / (sr * time_part)))[0].shape)
    print(np.array(np.array_split(np.array(wave), leng / (sr * time_part)))[-1].shape)
    print(wave_lin.shape)
    for ii in range(cells):
        wave_lin_splits[ii] = wave_lin[ii * sr:(ii + 1) * sr]
        print(wave_lin_splits.shape)
    wave_split = np.array_split(np.array(wave), leng / (sr * time_part))
    new_wave = None
    for wave_part_i, wave_part in enumerate(wave_split):
        frec = 20_002
        for hash in hashes:
            for car in hash:
                if car != ' ':
                    frec += (int(car, 16) + 1)
                else:
                    frec += 17
                if wave_part.shape[1] == 2:
                    wave_part[:, 0] += amp * np.sin(frec * 2.0 * np.pi * wave_lin_splits[wave_part_i])
                    wave_part[:, 1] += amp * np.sin(frec * 2.0 * np.pi * wave_lin_splits[wave_part_i])
                else:
                    wave_part += amp * np.sin(frec * 2.0 * np.pi * wave_lin_splits[wave_part_i])
            frec += 18

        wave_part = np.array(wave_part) / (32767 + amp * 64) * 32767

        if type(new_wave) == type(None):
            new_wave = wave_part
        else:
            new_wave = np.vstack((new_wave, wave_part))
    '''
    new_wave = pd.DataFrame(new_wave)

    if show_plot:
        yf = fft(np.array(new_wave[0]))
        xf = fftfreq(len(new_wave[0]), 1 / sr)[:len(new_wave[0]) // 2]
        plt.plot(xf, 2.0 / len(x) * np.abs(yf)[0:len(x) // 2])
        plt.grid()
        plt.show()
    if do_write:
        write(f"{file_name}_encoded.flac", sr, wave, 320)
    '''
    return new_wave


# wave_x = clean_high_frecs(x)
# encode_parts(np.array(wave_x), sr, hashes,FILE_NAME, 1, do_write=True)

def encode_2(wave, sr, br, hashes, file_name, show_plot=False, do_write=True):
    frec = 20_002
    leng = len(wave)

    amp = np.array(wave).max() // 50

    if wave.shape[1] == 2:
        xf = fftfreq(len(wave[0]), 1 / sr)

        yf_o = fft(np.array(wave[0]))
        yf_1 = fft(np.array(wave[1]))
        t_beg = perf_counter()
        amp = yf_o.real.max() // 1_500
        diff_array = np.absolute(xf - frec)
        frec_index = diff_array.argmin()
        diff_array = np.absolute(xf + frec)
        frec_index_down = diff_array.argmin()
        o_indexes = np.where(np.logical_or(xf < -20_000, xf > 20_000))
        yf_o[o_indexes] = 0
        yf_1[o_indexes] = 0
        yf_o[frec_index] = amp + 1j * amp
        yf_1[frec_index] = amp + 1j * amp
        yf_o[frec_index_down] = amp + 1j * amp
        yf_1[frec_index_down] = amp + 1j * amp
    else:
        xf = fftfreq(len(wave), 1 / sr)
        diff_array = np.absolute(xf - frec)
        frec_index = diff_array.argmin()
        diff_array = np.absolute(xf + frec)
        frec_index_down = diff_array.argmin()
        yf = fft(np.array(wave))
        amp = yf.real.max() // 1_500
        o_indexes = np.where(np.logical_or(xf < -20_000, xf > 20_000))
        yf[o_indexes] = 0
        yf[frec_index] = amp + 1j * amp
        yf[frec_index_down] = amp + 1j * amp

    for hash in hashes:
        for car in hash:
            if car != ' ':
                frec += (int(car, 16) + 1)
            else:
                frec += 17
            diff_array = np.absolute(xf - frec)
            frec_index = diff_array.argmin()
            diff_array = np.absolute(xf + frec)
            frec_index_down = diff_array.argmin()
            if np.array(wave).shape[1] == 2:
                yf_o[frec_index] = amp + 1j * amp
                yf_1[frec_index] = amp + 1j * amp
                yf_o[frec_index_down] = amp + 1j * amp
                yf_1[frec_index_down] = amp + 1j * amp
            else:
                yf[frec_index] = amp + 1j * amp
                yf[frec_index_down] = amp + 1j * amp
        frec += 18
    if wave.shape[1] != 2:
        wave = scipy.fft.ifft(yf)
    else:
        wave = pd.DataFrame()
        wave[0] = scipy.fft.ifft(yf_o).real
        wave[1] = scipy.fft.ifft(yf_1).real
        # plt.plot(np.array([scipy.fft.ifft(fft(np.array(wave[0]))), scipy.fft.ifft(fft(np.array(wave[1])))]))
        # plt.show()
    wave = wave / wave.max() * 32767
    t_end = perf_counter()
    print('Time:', t_end - t_beg)
    if show_plot:
        plt.plot(wave)
        plt.show()
        try:
            plt.plot(xf, 2.0 / len(xf) * np.abs(yf))
            plt.grid()
            plt.show()
        except:
            plt.plot(xf, 2.0 / len(xf) * np.abs(yf_o))
            plt.grid()
            plt.show()
    if do_write:
        write(f"{file_name}_encoded.flac", sr, wave, 320)

    return wave


def real_time_decode_test(file_name, test, wave=None, show_plot=False):
    sr, br, x = read(file_name, 'flac')
    x = np.array(x)
    for i in range(normal_round(sr * 1), len(x[:, 0]), normal_round(sr * 1)):
        leng = len(x[:i, 0])
        yf = fft(x[:i, 0])
        yf = 2.0 / leng * np.abs(yf)[:leng // 2]
        xf = fftfreq(leng, 1 / sr)[:leng // 2]

        diff_array = np.absolute(xf - 20_000)
        index_slice = diff_array.argmin()

        yf = yf[index_slice:]
        xf = xf[index_slice:]
        if show_plot:
            plt.plot(xf, yf)
            plt.grid()
            plt.show()
        characters = np.sort(np.argsort(yf)[-129:])
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

        if dec_hashes == test:
            return True, normal_round(leng / sr, 2)
    return False, np.nan


'''
enc_wave = encode(x, sr, br, hashes, FILE_NAME)
decode(f"{FILE_NAME}_encoded.wav", enc_wave)
print(hashes)
'''


def test():
    if not os.path.exists('E:\Encoded Songs\Songs_test_results.csv'):
        results = pd.DataFrame([],
                               columns=['Song', 'Original File', 'Encoded File', 'Time To Encode', 'Hash', 'Decoded',
                                        'Time To Decode'])
        results.to_csv('E:\Encoded Songs\Songs_test_results.csv')
    try:
        results = pd.read_csv('E:\Encoded Songs\Songs_test_results.csv', index_col=[0])
    except:
        pass
    prev = "E:\PC antiguo\musica\Bueno"
    hash_waves = hashlib.sha256(b"Waves").hexdigest()
    for art in os.listdir("E:\PC antiguo\musica\Bueno"):
        try:
            os.mkdir(f'E:\Encoded Songs\{art}')
        except OSError as error:
            pass
        for alb in os.listdir(prev + "/" + art):
            try:
                os.mkdir(f'E:\Encoded Songs\{art}\{alb}')
            except OSError as error:
                pass
            if '.' not in prev + "/" + art + "/" + alb:
                for song in os.listdir(prev + "/" + art + "/" + alb):
                    row = []
                    if '.flac' in song and song[:-5] not in results['Song'].values:
                        row.append(song[:-5])
                        row.append(prev + "/" + art + "/" + alb + '/' + song)
                        row.append(f'E:\Encoded Songs\{art}\{alb}\{song[:-5]}_encoded.flac')

                        sr, br, x = read(prev + "/" + art + "/" + alb + '/' + song, 'flac')
                        hash_license = hashlib.sha256(("Ibai Twitch " + song[:-5]).encode('utf-8')).hexdigest()
                        hashes = [hash_waves, hash_license]

                        t_start = perf_counter()
                        enc_wave = encode(x, sr, br, hashes, f'E:\Encoded Songs\{art}\{alb}\{song[:-5]}')
                        t_stop = perf_counter()
                        row.append(normal_round(t_stop - t_start, 2))
                        print(normal_round(t_stop - t_start, 2))
                        row.append(hash_waves + '//' + hash_license)
                        decoded, time_to_decode = real_time_decode_test(
                            f'E:\Encoded Songs\{art}\{alb}\{song[:-5]}_encoded.flac', hashes)
                        row.append(decoded)
                        row.append(time_to_decode)
                        results = pd.concat([results, pd.DataFrame([row],
                                                                   columns=['Song', 'Original File', 'Encoded File',
                                                                            'Time To Encode', 'Hash', 'Decoded',
                                                                            'Time To Decode'])], axis=0).reset_index(
                            drop=True)
                        results.to_csv('E:\Encoded Songs\Songs_test_results.csv')


def get_time_col():
    prev = "E:\PC antiguo\musica\Bueno"
    results = pd.read_csv('E:\Encoded Songs\Songs_test_results.csv', index_col=[0])
    col = []
    for file_i, file in enumerate(results['Original File']):
        print(normal_round(file_i / len(results) * 100, 2), '%')
        sr, br, x = read(file, 'flac')
        col.append(normal_round(len(x) / sr, 2))
    results['Song Duration'] = col
    results.to_csv('E:\Encoded Songs\Songs_test_results.csv')


def csv_analysis():
    results = pd.read_csv('E:\Encoded Songs\Songs_test_results.csv', index_col=[0])
    print("N:", len(results))
    print("Time To Decode: ", str(normal_round(results['Time To Decode'].mean(), 2)) + ' +/- ' + str(
        normal_round(results['Time To Decode'].std(), 2)))
    print("Decoded: ",
          str(normal_round(results['Decoded'].mean(), 2)) + ' +/- ' + str(normal_round(results['Decoded'].std(), 2)))
    times_relation = results['Time To Encode'] / results['Song Duration']
    print("Encoding Time: ",
          str(normal_round(times_relation.mean() / 2, 2)) + ' +/- ' + str(normal_round(times_relation.std() / 2, 2)))
    print("Max Time To Decode: " + str(results['Time To Decode'].max()))
    fig = results['Time To Encode'].plot.hist()
    fig.show()
    fig = results['Time To Decode'].plot.hist()
    fig.show()
    fig = results['Decoded'].plot.hist()
    fig.show()
    print("Decoded: " + str(results['Decoded'].min()))
    print(results.describe())
    print(
        results[['Song', 'Time To Encode', 'Decoded', 'Time To Decode']].sort_values('Time To Decode', ascending=False))


'''
sr, br, x = read(f"E:\Pc antiguo\musica\Bueno\Wisin & Yandel\De Otra Manera/Tarzan.flac", 'flac')
hash_waves = hashlib.sha256(b"Waves").hexdigest()
hash_license = hashlib.sha256(("Ibai Twitch Tarzan").encode('utf-8')).hexdigest()
hashes = [hash_waves, hash_license]
encode(x, sr, br , hashes,'', write = False, show_plot=True)
'''

# real_time_decode_test('E:\Encoded Songs\Wisin & Yandel\De Otra Manera/Tarzan_encoded.flac', hashes, show_plot = True)

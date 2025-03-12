import matplotlib.pyplot as plt
import os.path
from mne.decoding import CSP

from mne.preprocessing import ICA
from mne.io import RawArray
import seaborn as sns

from scipy.signal import stft
import numpy as np

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import set_log_level
from preprocessing import *

def create_CSP_database(epochs, labels, pasta='csp_db_1'):

    ####### Cria a base de Dados do CSP com imagens #######

    csp = CSP(n_components=2, transform_into='csp_space')
    x_csp = csp.fit_transform(epochs, labels)
    #x_csp é do tamanho 225x2 = épocas X componentes

    dpi = 50
    figsize = (224/dpi, 224/dpi)
    
    os.makedirs(os.path.join('databases', pasta, 'hands'), exist_ok=True)
    os.makedirs(os.path.join('databases', pasta, 'feet'), exist_ok=True)

    for i, (epoch, label) in enumerate(zip(x_csp, labels)):
        plt.figure(figsize=figsize, dpi=dpi, layout='tight')
        plt.subplot(2,1,1)
        plt.plot(epoch[0])
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.plot(epoch[1])
        plt.axis('off')
        pasta_classe = "hands" if label == 0 else "feet"
        plt.savefig(os.path.join('databases', pasta, pasta_classe, f'sample{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def create_ICA_database(epochs, labels, info, pasta='ica_db_1', n_components = 64):
    """Aplica ICA nas épocas e salva imagens dos sinais independentes."""
    
    dpi = 50
    figsize = (224 / dpi, 224 / dpi)  # Dimensão da imagem para ML

    os.makedirs(os.path.join('databases', pasta, 'hands'), exist_ok=True)
    os.makedirs(os.path.join('databases', pasta, 'feet'), exist_ok=True)

    import numpy as np

    if np.isnan(epochs).any() or np.isinf(epochs).any():
        print("Os dados contêm NaNs ou valores infinitos!")

    variances = np.var(epochs, axis=(0, 2))  # Variância por canal
    print("Variância mínima:", np.min(variances))
    print("Variância máxima:", np.max(variances))

    for i, (epoch, label) in enumerate(zip(epochs, labels)):
        # Criar objeto ICA
        ica = ICA(n_components=n_components)
        
        # Criar um objeto Raw temporário para rodar a ICA
        raw_epoch = RawArray(epoch, info)
        
        # Ajustar e transformar a época
        ica.fit(raw_epoch)
        sources = ica.get_sources(raw_epoch).get_data()  # shape (n_components, n_amostras)

        #criar figura
        fig = plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(sources, cmap="coolwarm", cbar=False, center=0, xticklabels=False, yticklabels=False)

        # Salvar a imagem
        pasta_classe = "hands" if label == 0 else "feet"
        fig.savefig(os.path.join('databases', pasta, pasta_classe, f'sample{i}.png'), bbox_inches='tight', pad_inches=0)

        plt.close(fig)  # Fecha a figura explicitamente

def create_STFT_database(epochs, labels, pasta="stft_db_1", fs=160, nperseg=16):
    """Aplica STFT nas épocas e salva espectrogramas."""
    
    dpi = 50
    figsize = (224 / dpi, 224 / dpi)  # Tamanho da imagem para CNN

    os.makedirs(os.path.join('databases', pasta, 'hands'), exist_ok=True)
    os.makedirs(os.path.join('databases', pasta, 'feet'), exist_ok=True)

    for i, (epoch, label) in enumerate(zip(epochs, labels)):
        # Média dos canais para um único sinal representativo
        epoch_mean = np.mean(epoch, axis=0)

        # Aplicar STFT
        f, t, Zxx = stft(epoch_mean, fs=fs, nperseg=nperseg)
        spectrogram = np.abs(Zxx)  # Módulo da STFT (espectrograma)

        # Criar Heatmap
        fig = plt.figure(figsize=figsize, dpi=dpi, layout='tight')
        sns.heatmap(spectrogram, cmap="magma", cbar=False, xticklabels=False, yticklabels=False)

        # Salvar a imagem sem bordas
        pasta_classe = "hands" if label == 0 else "feet"
        fig.savefig(os.path.join('databases', pasta, pasta_classe, f'sample{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def open_raw_data(subjects, runs):
    
    set_log_level('ERROR')

    tmin, tmax = -1.0, 4.0

    raw_fnames = eegbci.load_data(subjects, runs)

    # No dataset, 3 dos indivíduos foram coletados com 128 Hz de taxa de 
    # amostragem, enquanto os outros foram com 160 Hz. Estes 3 foram eliminados 

    raws = []
    for f in raw_fnames:
        raw_edf = read_raw_edf(f, preload=True)
        if raw_edf.info['sfreq'] == 160.0:
            raws.append(raw_edf)

    raw = concatenate_raws(raws)
    eegbci.standardize(raw) 
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)

    # Filtro Passa-banda
    # Entre 7 e 30 Hz, segundo a literatura é o ideal
    raw.filter(7.0, 30.0, skip_by_annotation="edge")

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    epochs = Epochs(
        raw,
        event_id=["hands", "feet"],
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1] - 2

    epochs = epochs_train.get_data(copy=False)

    print(f'{len(epochs)} janelas com  {len([i for i in labels if i == 1])} classes "hands"')

    return epochs, labels
    #
    # São 225 janelas, 113 com classe 1 e 112 com classe 0
    # Cada janela é uma matriz de 64 canais X 161 amostras

    #from mne.decoding import CSP

    #csp = CSP(n_components=2, transform_into='csp_space')
    #x_csp = csp.fit_transform(epochs, labels)
    #x_csp é do tamanho 225x62 = épocas X componentes


    # csp_db_1 tem 5 indivíduos (225 imagens)
    # csp_db_2 tem 10 indivíduos (450 imagens)
    # csp_db_3 tem 106 indivíduos (4748 imagens, 2396 classes 1)


    # ica_db_1 tem 5 indivíduos (225 imagens)
    # ica_db_2 tem 106 indivíduos (4748 imagens, 2396 classes 1)

    # stft_db_1 tem 5 indivíduos (225 imagens)
    # stft_db_2 tem 106 indivíduos (4748 imagens, 2396 classes 1)

#!/usr/bin/env python
# coding: utf-8

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import IPython.display as ipd
import random


# # Alloy Sounds - A Demo
# 
# In the next cells, you will find some code, but you actually only need to run:
# 
# *alloy_sounds(audio_1,audio_2,cells,steps,p_array,seed,n_fft=512, attack=0.1)*.
# 
# The function parameters are:
# * *cells* -> The spectrum is equally divided into this number of cells
# * *steps* -> The spectrogram is equally divided into this number of time steps
# * *p_array* -> the probability array, as explained below.
# * *seed* -> Random number seed. Use it to replicate results.
# * *attack* -> Controls the smoothness of the transitions between states in the spectrogram.
# 
# *p_array* is the probability array related to each independent cell (row). It is the probability of the current cell having value 1 in the next time step, given the state of the cell and its neighbors. These previous states are encoded as a three-bit binary number, in which the first position relates to the upwards neighbor, the second position relates to the cell, and the third position relates to the downwards neighbor. Hence, *p_array[3]*, for example, relates to encoding *011*, that is, the upward neighbor is *0*, and the current cell and its downward neighbor are *1*.
# 

# In[2]:


# Functions to create the probabilistic cellular automata
def step(x,p_array):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        situation = 4 * x[ (i-1) % len(y) ] + 2 * x[i] + x[(i+1) % len(y)]
        if random.uniform(0,1) < p_array[situation]:
            y[i] = 1 
    return y

def generate(p_array, size=10, steps=50, seed=2, set_init_zero=False):
    x = np.zeros([steps,size],dtype=np.int8)
    random.seed(seed)
    if set_init_zero == True:
        for i in range(size):
            x[0][i] = 0
    else:
        for i in range(size):
            x[0][i] = random.uniform(0,1) < .5
    for i in range(steps - 1):
        x[i + 1, :] = step(x[i, :], p_array)
    return x

# Perform audio blending
def blend_audio(audio_1,audio_2,cells,steps,p_array,seed,n_fft=512,attack=0.1,set_init_zero=False):
    spectra_1 = librosa.core.stft(audio_1, n_fft=n_fft, hop_length=int(n_fft/4), win_length=int(n_fft/2), window='hann')
    spectra_2 = librosa.core.stft(audio_2, n_fft=n_fft, hop_length=int(n_fft/4), win_length=int(n_fft/2), window='hann')

    cellular_spectra = generate(p_array,cells,steps,seed,set_init_zero)
    cellular_spectra = np.transpose(cellular_spectra)
     
    mask_spectra = np.zeros(spectra_1.shape)
#     print(spectra_1.shape)
    for i in range(mask_spectra.shape[0]):
        for j in range(mask_spectra.shape[1]):
            mask_spectra[i][j] = cellular_spectra [ int(i*cells/mask_spectra.shape[0]) ][ int(j*steps/mask_spectra.shape[1])]
    
    filtB = np.array([attack])
    filtA = np.array([1, -(1-attack)])
    mask_spectra_inv = np.transpose(mask_spectra)
    mask_spectra = sig.lfilter(filtB, filtA, mask_spectra)
    mask_spectra_inv = sig.lfilter(filtB, filtA, mask_spectra_inv)
    
#     plt.figure()
#     plt.imshow(mask_spectra, interpolation='none',cmap=plt.cm.gray)
#     plt.show()
    
    blended_spectra = spectra_2 * mask_spectra + spectra_1 * (1-mask_spectra)
    
    blended_audio = librosa.core.istft(blended_spectra,  hop_length=int(n_fft/4), win_length=int(n_fft/2), window='hann')
    return blended_audio, mask_spectra

# Resizes audio arrays using zero-padding
def resize_audio(audio_1, audio_2):
    diff = audio_1.shape[0] - audio_2.shape[0]
    if diff > 0:
        audio_2 = np.append(audio_2,np.zeros(diff))
    elif diff < 0:
        audio_1 = np.append(audio_1,np.zeros(np.abs(diff)))
    return audio_1, audio_2


# In[3]:


def alloy_sounds(audio_1,audio_2,cells,steps,p_array,seed,n_fft=512, attack=0.1,set_init_zero=False):
    audio_1, audio_2 = resize_audio(audio_1,audio_2)
    blended_audio, a = blend_audio(audio_1,audio_2,cells,steps,p_array,seed,n_fft, attack,set_init_zero)
    return blended_audio, a


# In[4]:


# These are several p_array samples. Use them as demos or devise your own :)
# Uncomment the one you want to use.

# Guide -> 000,  001,  010,  011, 100, 101,  110,  111
#p_array = [0, 1, 1, 0, 1, 1, 1, 0] # Rule 110
#p_array = [.3, .7, .7, .3, .7, .7, .7, .3] # Relaxed Rule 110
#p_array = [0.1, 0, 0, 0, 0, 0, 0, 0] # Spots
#p_array = [0.1, 0, 0.5, 0.9, 0, 0, 0.8, 0.9] # Stripes
p_array = [0.1, 0.1, 0.5, 0.9, 0.1, 0.5, 0.8, 0.9] # Stains


# # Demos

# ## Stained Chords

# In[23]:


# Stained chords
audio1,fs = librosa.core.load("Chord_Clarinet.wav")
audio2,fs = librosa.core.load("Chord_Horns.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [0.1, 0.1, 0.5, 0.9, 0.1, 0.5, 0.8, 0.9] # Stains
blended, mask = alloy_sounds(audio1,audio2,10,120,p_array,2,n_fft=2048, attack=0.01)
ipd.Audio(data=blended, rate=fs)
sf.write('01_stained_chords.wav', blended, fs)


# In[7]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# ## Relaxed Rule 110 Melody Blend

# In[24]:


# Relaxed Rule 110 Melody Blend
audio1,fs = librosa.core.load("Melody_Oboe.wav")
audio2,fs = librosa.core.load("Melody_Violin.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [.3, .7, .7, .3, .7, .7, .7, .3] # Relaxed Rule 110
blended, mask = alloy_sounds(audio1,audio2,20,60,p_array,2,n_fft=2048, attack=0.1)
ipd.Audio(data=blended, rate=fs)
sf.write('02_rr110_melody_blend.wav', blended, fs)


# In[9]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# ## Ostinato Stripes

# In[25]:


# Ostinato Stripes
audio2,fs = librosa.core.load("Ostinato_Clarinet.wav")
audio1,fs = librosa.core.load("Ostinato_Violin_1.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [0.1, 0, 0.5, 0.9, 0, 0, 0.8, 0.9] # Stripes
blended, mask = alloy_sounds(audio1,audio2,20,60,p_array,2,n_fft=2048, attack=0.3)
ipd.Audio(data=blended, rate=fs)
sf.write('03_ostinato_stripes.wav', blended, fs)


# In[11]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# ## Ocean Staining the Violin

# In[26]:


# Ocean Staining the Violin
audio1,fs = librosa.core.load("Melody_Violin.wav")
audio2,fs = librosa.core.load("usar_no_exemplo_4_oceano.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [0.1, 0, 0, 0, 0, 0, 0, 0] # Spots
blended, mask = alloy_sounds(audio1,audio2,20,60,p_array,2,n_fft=2048, attack=0.7)
ipd.Audio(data=blended, rate=fs)
sf.write('04_ocean_staining_the_violin.wav', blended, fs)


# In[13]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# ## City and Train: a Blended Landscape

# In[27]:


# City and Train: a Blended Landscape
audio1,fs = librosa.core.load("usar_no_exemplo_5_cidade.wav")
audio2,fs = librosa.core.load("usar_no_exemplo_5_trem.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [0.1, 0.1, 0.5, 0.9, 0.1, 0.5, 0.8, 0.9] # Stains
blended, mask = alloy_sounds(audio1,audio2,30,480,p_array,2,n_fft=2048, attack=0.5)
ipd.Audio(data=blended, rate=fs)
sf.write('05_blended_landscape.wav', blended, fs)


# In[15]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# ## A Blend of Maracatu and Orchestra

# In[28]:


# A Blend of Maracatu and Orchestral Chords
audio1,fs = librosa.core.load("usar_no_exemplo_6_maracatu.wav")
audio2,fs = librosa.core.load("Chord_Horns.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [.3, .7, .7, .3, .7, .7, .7, .3] # Relaxed Rule 110
blended, mask = alloy_sounds(audio1,audio2,20,60,p_array,2,n_fft=2048, attack=0.1)
ipd.Audio(data=blended, rate=fs)
sf.write('06_blend_maracatu_orchestral.wav', blended, fs)


# In[17]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# ## Spooky Chords

# In[30]:


# Spooky Chords
audio1,fs = librosa.core.load("usar_no_exemplo_7_som_spooki.wav")
audio2,fs = librosa.core.load("Chord_Clarinet.wav")
audio1, audio2 = resize_audio(audio1,audio2)
p_array = [0.1, 0.1, 0.5, 0.9, 0.1, 0.5, 0.8, 0.9] # Stains
blended, mask = alloy_sounds(audio1,audio2,20,60,p_array,2,n_fft=2048, attack=0.01)
ipd.Audio(data=blended, rate=fs)
sf.write('07_spooky_chords.wav', blended, fs)


# In[19]:


# Plain mix for comparison
ipd.Audio(data=audio1+audio2, rate=fs)


# In[ ]:





# # Credits for the wavfiles used here
# 
# * Chord_Clarinet.wav, Chord_Horns.wav, Melody_Oboe.wav, Melody_Violin.wav, Ostinato_Clarinet.wav, and Ostinato_Violin_1.wav were rendered by ourselves using a VST synthesizer.
# * usar_no_exemplo_4_oceano.wav was downloaded from https://freesound.org/people/INNORECORDS/sounds/456899/, downmixed to mono and cropped to around 1min length;
# * usar_no_exemplo_5_cidade.wav was downloaded from https://freesound.org/people/chimerical/sounds/106839/, downmixed to mono and cropped to around 1min length;
# * usar_no_exemplo_5_trem.wav was downloaded from https://freesound.org/people/alexkandrell/sounds/277755/, downmixed to mono and cropped to around 1min length;
# * usar_no_exemplo_6_maracatu.wav was downloaded from https://freesound.org/people/reinsamba/sounds/204197/, downmixed to mono and cropped to around 1min length;
# * usar_no_exemplo_7_spooki.wav was downloaded from https://freesound.org/people/KRISTIANKULTA/sounds/326962/, downmixed to mono and cropped to around 1min length;

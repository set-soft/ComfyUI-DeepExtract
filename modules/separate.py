from __future__ import annotations
from typing import TYPE_CHECKING
import os
import gc
import logging
import numpy as np
import soundfile as sf
import librosa
import torch
import onnxruntime
from onnx import load
from onnx2pytorch import ConvertModel

from .stft import STFT

if TYPE_CHECKING:
    from scripts.vocal_and_sound_remover import ModelData

logger = logging.getLogger(__name__)

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    gc.collect()
    torch.cuda.empty_cache()

def verify_audio(audio_file):
    """Verify if the provided file is a valid audio file."""
    is_audio = True

    if not isinstance(audio_file, (tuple, list)):
        audio_file = [audio_file]

    for file in audio_file:
        if os.path.isfile(file):
            try:
                librosa.load(file, duration=3, mono=False, sr=44100)
            except Exception as e:
                print(f'Failed to load {file}: {e}')
                is_audio = False

    return is_audio

def prepare_mix(mix):
    """Prepare the mix for processing."""
    if isinstance(mix, dict) and 'waveform' in mix and 'sample_rate' in mix:
        mix = mix['waveform'].numpy()
    elif isinstance(mix, str):
        mix, _ = librosa.load(mix, mono=False, sr=44100)
    elif isinstance(mix, np.ndarray):
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
    else:
        raise ValueError("Unsupported input type for mix.")

    return mix

class SeparateAttributes:
    """Class to hold attributes for separation."""
    def __init__(self, model_data: ModelData, process_data: dict):
        self.model_basename = model_data.model_basename
        self.model_path = model_data.model_path
        self.primary_stem = model_data.primary_stem
        self.mdx_segment_size = model_data.mdx_segment_size
        self.mdx_batch_size = model_data.mdx_batch_size
        self.compensate = model_data.compensate
        self.dim_f = model_data.mdx_dim_f_set
        self.dim_t = 2 ** model_data.mdx_dim_t_set
        self.n_fft = model_data.mdx_n_fft_scale_set
        self.hop = 1024
        self.adjust = 1
        self.audio_file = process_data['audio_file']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run_type = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

class SeparateMDX(SeparateAttributes):
    """Class to handle separation using MDX models."""
    def separate(self):
        samplerate = 44100

        self.model_run = ConvertModel(load(self.model_path))
        self.model_run.to(self.device).eval()
        
        mix = prepare_mix(self.audio_file)

        logger.info(mix)
        logger.info(mix.shape)

        vocals = self.demix(mix)
        background = mix - vocals

        # Tempo düzeltmesi yapalım
        original_length = mix.shape[-1]
        vocals = fix_tempo(vocals, original_length)
        background = fix_tempo(background, original_length)


        clear_gpu_cache()

        vocals_tensor = torch.tensor(vocals, dtype=torch.float32).unsqueeze(0)
        background_tensor = torch.tensor(background, dtype=torch.float32).unsqueeze(0)
    
        
        logger.info(vocals_tensor.shape)
        logger.info(background_tensor.shape)
        return {'waveform': vocals_tensor, 'sample_rate': samplerate}, {'waveform': background_tensor.squeeze(0), 'sample_rate': samplerate}

    def initialize_model_settings(self):
        """Initialize model settings for STFT and chunking."""
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop * (self.mdx_segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)

    def demix(self, mix):
        """Demix the audio into vocals and background."""
        self.initialize_model_settings()

        # Ensure mix is 2D
        if mix.ndim == 3:
            mix = np.squeeze(mix, axis=0)  # Remove the first dimension if it's 1

        tar_waves_ = []
        chunk_size = self.chunk_size
        gen_size = self.gen_size

        # Pad the mixture
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate(
            (np.zeros((2, self.trim), dtype='float32'), 
            mix, 
            np.zeros((2, pad), dtype='float32')), 
            axis=1
        )

        step = self.chunk_size - self.n_fft
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        for i in range(0, mixture.shape[-1], step):
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start

            # Apply Hanning window
            window = np.hanning(chunk_size_actual)
            window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate(
                    (mix_part_, np.zeros((2, pad_size), dtype='float32')), 
                    axis=-1
                )

            mix_part = torch.tensor(mix_part_, dtype=torch.float32).unsqueeze(0).to(self.device)
            mix_waves = mix_part.split(self.mdx_batch_size)

            with torch.no_grad():
                for mix_wave in mix_waves:
                    tar_waves = self.run_model(mix_wave)
                    tar_waves[..., :chunk_size_actual] *= window
                    divider[..., start:end] += window
                    result[..., start:end] += tar_waves[..., :end - start]

        # Normalize the output
        epsilon = 1e-8  # Prevent division by zero
        tar_waves = result / (divider + epsilon)
        tar_waves_.append(tar_waves)

        # Concatenate and trim the output
        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim:-self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]

        # Apply compensation
        source = tar_waves[:, 0:None] * self.compensate

        return source
    
   
    
    def run_model(self, mix):
        spek = self.stft(mix.to(self.device))*self.adjust
        spek[:, :, :3, :] *= 0

        spec_pred = self.model_run(spek)

        return self.stft.inverse(torch.tensor(spec_pred).to(self.device)).cpu().detach().numpy()
    
def fix_tempo(waveform: np.ndarray, original_length: int) -> np.ndarray:
    """
    Corrects the tempo of the separated audio after source separation.

    Args:
        waveform (np.ndarray): The separated waveform data (2, N).
        original_length (int): The number of samples in the original mixed audio.

    Returns:
        np.ndarray: Tempo-corrected waveform.
    """
    # If the difference is very small, no correction is needed
    if abs(waveform.shape[-1] - original_length) < 10:
        return waveform

    # Calculate the stretch rate
    rate = waveform.shape[-1] / original_length

    # Correct each channel separately
    fixed_waveform = []
    for ch in waveform:
        ch_fixed = librosa.effects.time_stretch(ch, rate)
        fixed_waveform.append(ch_fixed)
    
    # Cut to match the exact original length
    fixed_waveform = np.stack(fixed_waveform)[:, :original_length]
    return fixed_waveform

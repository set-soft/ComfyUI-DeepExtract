from __future__ import annotations
from typing import TYPE_CHECKING
import gc
import logging
import numpy as np
import librosa   # Used only when we need to adjust the rate
import torch
from onnx import load
from onnx2pytorch import ConvertModel
from tqdm import tqdm # Import tqdm

from .stft import STFT
from .resample import resample_audio_numpy

if TYPE_CHECKING:
    from scripts.vocal_and_sound_remover import ModelData

logger = logging.getLogger(__name__)
TARGET_SR = 44100


class SeparateAttributes:
    """Class to hold attributes for separation."""
    def __init__(self, model_data: ModelData, audio_data):
        self.model_basename = model_data.model_basename
        self.model_path = model_data.model_path
        self.mdx_segment_size = model_data.mdx_segment_size
        self.mdx_batch_size = model_data.mdx_batch_size
        self.compensate = model_data.compensate
        self.dim_f = model_data.mdx_dim_f_set
        self.dim_t = 2 ** model_data.mdx_dim_t_set
        self.n_fft = model_data.mdx_n_fft_scale_set
        self.hop = 1024
        self.adjust = 1
        self.audio_data = audio_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run_type = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']


class SeparateMDX(SeparateAttributes):
    """Class to handle separation using MDX models."""
    def separate(self):
        samplerate = 44100
        self.model_run = ConvertModel(load(self.model_path))
        self.model_run.to(self.device).eval()
        mix = np.squeeze(self.audio_data['waveform'].numpy(), axis=0)
        original_sr = self.audio_data['sample_rate']

        channels, original_length = mix.shape
        logger.info(f"Input audio: {channels} ch {original_length} samples @ {original_sr} Hz")

        # Ensure 44.1 kHz SR
        if original_sr != TARGET_SR:
            mix = resample_audio_numpy(mix, original_sr, TARGET_SR)

        # Ensure stereo
        back_to_mono = False
        if channels == 1:
            logger.info(f"Repeating channel to make fake stereo")
            mix = np.repeat(mix, 2, axis=0)
            back_to_mono = True

        channels, adapted_length = mix.shape
        logger.info(f"Audio to process {channels} ch {adapted_length} samples @ {TARGET_SR} Hz")

        main_audio = self.demix(mix)
        complement_audio = mix - main_audio

        # Do we need to convert to mono?
        if back_to_mono:
            main_audio = main_audio[0:1, :]
            complement_audio = complement_audio[0:1, :]

        # Do we need to resample?
        if original_sr != TARGET_SR:
            main_audio = resample_audio_numpy(main_audio, TARGET_SR, original_sr)
            complement_audio = resample_audio_numpy(complement_audio, TARGET_SR, original_sr)

        channels, final_length = main_audio.shape
        logger.info(f"Result {channels} ch {final_length} samples @ {original_sr} Hz")

        # Clear GPU cache to free up memory
        gc.collect()
        torch.cuda.empty_cache()

        main_audio_tensor = torch.tensor(main_audio, dtype=torch.float32).unsqueeze(0)
        complement_audio_tensor = torch.tensor(complement_audio, dtype=torch.float32).unsqueeze(0)

        return ({'waveform': main_audio_tensor, 'sample_rate': original_sr},
                {'waveform': complement_audio_tensor, 'sample_rate': original_sr})

    def initialize_model_settings(self):
        """Initialize model settings for STFT and chunking."""
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop * (self.mdx_segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)

    def demix(self, mix):
        """Demix the audio, separate the interesting thing (might be vocals, instruments, etc.)"""
        self.initialize_model_settings()

        tar_waves_ = []
        chunk_size = self.chunk_size
        gen_size = self.gen_size

        # Pad the mixture
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate((np.zeros((2, self.trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')),
                                 axis=1)

        step = self.chunk_size - self.n_fft
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        pbar_total = (mixture.shape[-1] - 1) // step + 1  # Number of loop executions

        with tqdm(total=pbar_total, desc="Demixing Chunks") as pbar:
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
                    mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

                mix_part = torch.tensor(mix_part_, dtype=torch.float32).unsqueeze(0).to(self.device)
                mix_waves = mix_part.split(self.mdx_batch_size)

                with torch.no_grad():
                    for mix_wave in mix_waves:
                        tar_waves = self.run_model(mix_wave)
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                        result[..., start:end] += tar_waves[..., :end - start]

                pbar.update(1)  # Update progress bar by one iteration

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

import logging
import os

from modules.separate import SeparateMDX
from .downloader import Downloader
# Here I found various MDX-Net's
BASE_URL = 'https://huggingface.co/seanghay/uvr_models/resolve/main/'
logger = logging.getLogger(__name__)


class ModelData:
    def __init__(self, model_path: str, segments: int):
        self.model_path = model_path
        self.model_basename = os.path.splitext(os.path.basename(model_path))[0]
        self.compensate = 1.009
        self.mdx_dim_f_set = 3072
        self.mdx_dim_t_set = 8
        self.mdx_n_fft_scale_set = 6144

        self.mdx_segment_size = 256 * segments
        self.mdx_batch_size = 1

        if not os.path.exists(self.model_path):
            download_url = os.path.join(BASE_URL, os.path.basename(model_path))
            print(f"Model not found, downloading: {download_url}")
            self.download_model(download_url)

    def download_model(self, url: str):
        try:
            Downloader(self.model_path).download_model(url)
            print(f"Model downloaded successfully: {self.model_path}")
        except Exception as e:
            raise Exception(f"An error occurred while downloading the model: {e}")


class VocalAndSoundRemover:
    def __init__(self, input_data, segments, model_path='Kim_Vocal_2.onnx'):
        self.input_data = input_data
        self.model_path = model_path
        self.segments = segments  # How many segments to process at once, bigger less overlap problems, but more memory

    def execute(self):
        try:
            logger.info("Starting vocal and sound separation process.")

            model = ModelData(self.model_path, self.segments)

            logger.info("Model loaded successfully, starting separation.")
            separator = SeparateMDX(model, self.input_data)
            result = separator.separate()

            logger.info("Separation process completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Error during separation: {str(e)}")
            raise e

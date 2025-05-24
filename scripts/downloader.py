# Simple URL downloader with TQDM progress
# Gemini 2.5 Pro code
import os
import urllib.request
from tqdm import tqdm  # Import tqdm


class Downloader:  # Example class structure
    def __init__(self, model_path):
        self.model_path = model_path
        # Ensure the directory for the model_path exists before __init__ if used elsewhere
        # or create it at the start of download_model

    # A TQDM helper class for urlretrieve reporthook
    # This is a common pattern for this use case.
    class TqdmUpTo(tqdm):
        """
        Provides `update_to(block_num, block_size, total_size)`
        and updates the TQDM bar.
        """
        def update_to(self, block_num=1, block_size=1, total_size=None):
            """
            block_num  : int, optional
                Number of blocks transferred so far [default: 1].
            block_size : int, optional
                Size of each block (in tqdm units) [default: 1].
            total_size : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if total_size is not None:
                self.total = total_size
            # self.update() will take the *difference* from the last call.
            # So we pass the number of new blocks * block_size.
            # Since block_num is cumulative, we calculate the new amount.
            self.update(block_num * block_size - self.n)  # self.n is current progress

    def download_model(self, url: str):
        try:
            # Ensure the directory exists
            # Use or '.' for current dir if dirname is empty
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)

            # Get filename for tqdm description
            filename = os.path.basename(self.model_path)

            # Use TqdmUpTo as a context manager
            with self.TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                               desc=f"Downloading {filename}") as t:
                # urlretrieve(url, filename=None, reporthook=None, data=None)
                # reporthook is called with (block_num, block_size, total_size)
                urllib.request.urlretrieve(url, self.model_path, reporthook=t.update_to)

            # The 'with' statement ensures t.close() is called.
            print(f"\nModel downloaded successfully: {self.model_path}")  # Add newline for cleaner output after tqdm

        except urllib.error.URLError as e:  # More specific exception for network issues
            # Clean up partially downloaded file if an error occurs
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            raise Exception(f"An error occurred while downloading the model (URL Error): {e.reason} from {url}")
        except Exception as e:
            # Clean up partially downloaded file if an error occurs
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            raise Exception(f"An unexpected error occurred while downloading the model: {e}")


# Example Usage:
if __name__ == '__main__':
    # Create a dummy directory and model path for testing
    test_model_dir = "test_models_temp"
    # os.makedirs(test_model_dir, exist_ok=True)  # Downloader will create it
    test_model_path = os.path.join(test_model_dir, "dummy_model.zip")  # Using a .zip as an example

    downloader_instance = Downloader(model_path=test_model_path)

    # Replace with a real, sizable file URL for testing progress
    # For example, a public dataset or a large image.
    # Using a small test file from a reliable source:
    test_url = "https://filesamples.com/samples/code/json/sample2.json"  # Small file, progress might be too fast
    # test_url = "https://effeleven.s3.amazonaws.com/sample.mp4" # ~10MB video for better progress view
    # Large file
    # test_url = "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors"

    print(f"Attempting to download from: {test_url}")
    print(f"Target path: {downloader_instance.model_path}")

    try:
        downloader_instance.download_model(test_url)
    except Exception as e:
        print(f"Download failed: {e}")

    # Clean up the dummy file and directory after test
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(test_model_dir) and not os.listdir(test_model_dir):  # only remove if empty
        os.rmdir(test_model_dir)

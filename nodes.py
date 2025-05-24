import logging
import os
import folder_paths  # ComfyUI's way to access model paths
from .scripts.vocal_and_sound_remover import VocalAndSoundRemover

logger = logging.getLogger(__name__)
DEF_MODEL = 'Kim_Vocal_2.onnx'
DEF_ENTRY = 'Default'


class VocalAndSoundRemoverNode:
    @classmethod
    def _get_available_audio_models(cls):
        """ Scans for models in the 'models/audio/MDX/' directory and caches the result.
            Gemini 2.5 Pro code """
        models_dir = os.path.join(folder_paths.models_dir, "audio", "MDX")

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            cls._audio_models_list = [DEF_ENTRY]
            return cls._audio_models_list

        supported_extensions = [".onnx"]  # , ".pth", ".pt", ".safetensors"]
        models = [DEF_ENTRY]
        try:
            for f in os.listdir(models_dir):
                if os.path.isfile(os.path.join(models_dir, f)):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in supported_extensions:
                        models.append(f)
        except Exception as e:
            logger.error(f"Error scanning audio models directory '{models_dir}': {e}")
            cls._audio_models_list = [DEF_ENTRY]  # Fallback
            return cls._audio_models_list

        return models if len(models) > 1 else [DEF_ENTRY]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_sound": ("AUDIO",),
                "model_filename": (cls._get_available_audio_models(), {"default": DEF_ENTRY}),  # Dropdown for model selection
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO",)
    RETURN_NAMES = ("Main", "Complement",)
    FUNCTION = "execute"
    CATEGORY = "DeepExtract"

    def execute(self, input_sound, model_filename):
        if model_filename == DEF_ENTRY or model_filename is None:
            model_filename = DEF_MODEL
        model_path = os.path.join(folder_paths.models_dir, "audio", "MDX", model_filename)
        demixer = VocalAndSoundRemover(input_sound, model_path)
        main_audio, complement_audio = demixer.execute()
        return (main_audio, complement_audio,)


NODE_CLASS_MAPPINGS = {
    "VocalAndSoundRemoverNode": VocalAndSoundRemoverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VocalAndSoundRemoverNode": "Vocal and Sound Separator",
}

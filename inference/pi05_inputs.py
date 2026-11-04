"""What a pi0.5 policy needs besides its weights: camera frames shrunk as openpi shrinks them, and the PaliGemma
tokenizer.

LeRobot's pi05 pads frames to 224x224 while they are still in [0, 1] with the value -1 and only then maps them to
[-1, 1], so its bars reach SigLIP as -3 instead of the black (-1) pi05_base was trained on. A frame that is already
224x224 skips that step, so frames go through resize() before the policy, in training and on the arm alike.
The tokenizer on the Hub (google/paligemma-3b-pt-224) is gated; openpi itself downloads the same SentencePiece model
from a public Google Cloud bucket, and the fast tokenizer built from it gives the same ids (checked on our prompts).
"""

from pathlib import Path
import urllib.request

import torch

IMAGE_SIZE = 224  # PI05Config.image_resolution of pi05_base
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_PATH = Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"  # where openpi keeps it


def resize(images: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] float32 in [0, 1] to [B, 3, 224, 224]: aspect ratio kept, black bars centred, as in openpi."""
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch  # needs the transformers fork for pi

    return (resize_with_pad_torch(images * 2 - 1, IMAGE_SIZE, IMAGE_SIZE) + 1) / 2


def shrink_cameras(frame: dict) -> dict:
    """A dataset frame with its camera images (HWC uint8) put through resize(), still HWC uint8."""
    shrunk = dict(frame)
    for key, image in frame.items():
        if key.startswith("observation.images."):
            floats = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
            resized = (resize(floats)[0].permute(1, 2, 0) * 255).round().clamp(0, 255)
            shrunk[key] = resized.numpy().astype("uint8")
    return shrunk


def load_tokenizer():
    """The PaliGemma tokenizer, downloaded once to TOKENIZER_PATH (4 MB)."""
    from transformers import GemmaTokenizerFast

    if not TOKENIZER_PATH.is_file():
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial_path = TOKENIZER_PATH.with_suffix(".part")
        urllib.request.urlretrieve(TOKENIZER_URL, partial_path)
        partial_path.rename(TOKENIZER_PATH)
    return GemmaTokenizerFast(vocab_file=str(TOKENIZER_PATH))

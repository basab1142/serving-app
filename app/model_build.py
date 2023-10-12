
## Importing important packages
import requests
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

import urllib.parse as parse
import os

def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)


    img = image_processor(image, return_tensors="pt").to(device)


    output = model.generate(**img)

    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return caption

def get_caption_2(model, image_processor, tokenizer, image):
    img = image_processor(image, return_tensors="pt").to(device)


    output = model.generate(**img)

    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return caption


# url = "https://images.pexels.com/photos/101667/pexels-photo-101667.jpeg?auto=compress&cs=tinysrgb&w=600"
# warnings.filterwarnings('ignore')
# caption = get_caption(model, image_processor, tokenizer, url)
# print(caption)
from typing import Dict,List,Optional,Union,Tuple,Iterable
import numpy as np
from PIL import Image
import torch

IMAGE_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGE_STANDARD_STD = [0.5,0.5,0.5]

def add_image_tokens_to_prompt(prefix_prompt,bos_token,image_seq_len,image_token):
    
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(
    image:Image,
    size:Tuple[int,int],
    resample:Image.Resampling=None,
    reducing_gap:Optional=None, # type: ignore
)-> np.ndarray :
    height , width = size
    resized_image = image.resize(
        (width,height),resample=resample,reducing_gap=reducing_gap
    )
    
    return resized_image

def rescale(
    image: Image,
    rescale_factor:float,
    dtype:np.dtype=np.float32
    )-> np.ndarray:
    rescaled_image = image * rescale_factor
    rescale_factor = rescale_factor.astype(dtype)
    
    return rescaled_image

def normalize(
    image: Image,
    mean:Union[float,Iterable[float]],
    std:Union[float,Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean,dtype=image.dtype)
    std = np.array(std,dtype=image.dtype)
    image = (image - mean)/ std
    
    return image

def process_images(
    images: List[Image.Image],
    size:Dict[str,int]=None,
    resample :  Image.Resampling = None,
    rescale_factor : float=None,
    image_mean : Optional[Union[float,List[float]]]= None,
    image_std : Optional[Union[float,List[float]]]= None,
    )-> List[np.ndarray]:
    height , width = size[0] , size[1]
    images = [
        resize(image=image, size=(height,width), resample=resample) for image in images
    ]
    
    images = [np.array(image) for image in images]
    images = [rescale(image,rescale_factor=rescale_factor) for image in images]
    images = [normalize(image,mean=image_mean,std=image_std) for image in images]
    images = [image.transpose(2,0,1) for image in images] ## (3,224,224)
    return images
         

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"
    def __init__(self,tokenizer,num_images_tokens:int,image_size:int):
        super().__init__()
        self.num_images_tokens =num_images_tokens
        self.image_size = image_size
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        
    def __call__(
        self,
        text:List[str],
        images: List[Image.Image],
        padding:str="longest",
        truncation:bool=True,
    ) -> dict : 
        assert len(images) ==1 and len(text) ==1 , f"Received {len(images)} for {len(text)} prompts"
        
        
        pixel_values = process_images(
            images,
            size=(self.image_size,self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGE_STANDARD_MEAN,
            image_std = IMAGE_STANDARD_STD,
        ) ## returns a list od tensors 
        
        pixel_values = np.stack(pixel_values,axis=0) 
        pixel_values = torch.tensor(pixel_values) ## (batch_size, channels, height, width)
        
        
        input_strings = [ # creates the tokens of the input text and tje place holder for the image tokens
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_tokens=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        
        inputs = self.tokenizer(
            input_strings,
            return_tensors='pt',
            padding=padding,
            truncation=truncation,
        )
        

from typing import Optional, Tuple
import torch 
import torch.nn as nn

class SigLIPVisionConfig():
    def __init__(
        self,
        hidden_size = 768, ## Embedding size
        inter_size = 3072, ## Linear layer in the feed forward network
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels=3, ## RGB images
        image_size = 224,
        patch_size=16, ## size of the patch (each image will be divided into patches)
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens:int=None, ## number of embeddings for each image
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens =num_image_tokens


class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self , config:SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.emb_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d( ## extracting information from the image , patch by patch without overlapping
            in_channels=config.num_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", ## ensure no padding is added
        )
        self.num_patches = (self.image_size // self.patch_size) **2
        self.num_positions = self.num_patches
        self.Pos_Embedding = nn.Embedding(self.num_positions,self.emb_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )
        
        def forward(self,pixel_values:torch.Tensor) -> torch.Tensor : 
            _,_, height , width = pixel_values.shape ## (batch_size, channels, height, width) 
            ## (batch_size, channels, height, width) ==> (batch_size, emb_dim, Num_patches_H, Num_patches_W)
            patch_embeddings = self.patch_embedding(pixel_values)
            ## (batch_size, emb_dim, Num_patches_H, Num_patches_W) ==> (batch_size, emb_dim, Num_patches) / Num_patches = Num_patches_H * Num_patches_W
            embeddings = patch_embeddings.flatten(2)
            ## (batch_size, emb_dim, Num_patches) ==> (batch_size, Num_patches, emb_dim)
            embeddings = embeddings.transpose(1,2)
            ## Add positional embedding for each patch
            embeddings = embeddings + self.Pos_Embedding(self.position_ids)
            ## (batch_size, Num_patches, emb_dim)
            return embeddings 
            
class SigLIPVisionEncoder(nn.Module):
    def __init__(self , config:SigLIPVisionConfig):
        super().__init__()
        self.emb_dim = config.hidden_size
        self.self_attn = SigLIPSelfAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.emb_dim,eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.emb_dim,eps=config.layer_norm_eps)
        
    def forward(self,embeddings:torch.Tensor) -> torch.Tensor :
        ## residual connection (batch_size, Num_patches, emb_dim)
        residual = embeddings
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim)
        embeddings = self.layer_norm1(embeddings)
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim) cc
        embeddings, _ = self.self_attn(embeddings)
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim) 
        embeddings = embeddings + residual
        residual = embeddings
        embeddings = self.layer_norm2(embeddings)
        embeddings = self.mlp(embeddings)
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim) 
        embeddings = embeddings + residual
        
        return embeddings

class SigLIPVisionTransformer(nn.Module):
    def __init__(self , config:SigLIPVisionConfig):
        super().__init__()
        self.config = config
        emb_dim = config.hidden_size
        self.embeddings = SigLIPVisionEmbeddings(config) ## extract patches
        self.encoder = SigLIPVisionEncoder(config)       ## encoder of the transformer (series of layers)
        self.post_layerNorm = nn.LayerNorm(emb_dim,eps=config.layer_norm_eps)
    def forward(self,pixel_values:torch.Tensor) -> torch.Tensor:
        ## pixel values : (batch_size, channels, height, width) ==> (batch_size, Num_patches, emb_dim)
        embeddings = self.embeddings(pixel_values)
        
        encoded_embeddings = self.encoder(embeddings) ## Convolution , split into patches , flattening + Positional Encoding
        encoded_embeddings = self.post_layerNorm(encoded_embeddings) 
        return encoded_embeddings
        
        
class SigLIPVisionModel(nn.Module):
    def __init__(self , config:SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.VisionModel = SigLIPVisionTransformer(config)
        
    def forward(self,pixel_values) -> Tuple:
        ## (batch_size, channels , height ,width) ==> (batch_size, num_patches, emb_dim)
        return self.VisionModel(pixel_values)
        
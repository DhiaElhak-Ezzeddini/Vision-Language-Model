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

class SigLIPMLP(nn.Module):
    def __init__(self , config:SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size , config.inter_size)
        self.fc2 = nn.Linear(config.inter_size , config.hidden_size)

    def forward(self,embeddings) : 
        upper_embeddings = self.fc1(embeddings)
        upper_embeddings = nn.functional.gelu(upper_embeddings,approximate="tanh")
        return self.fc2(upper_embeddings)
            
        
class SigLIPSelfAttention(nn.Module):
    def __init__(self , config:SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.emb_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.emb_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 ## (for the scaled dot product attention)
        self.dropout = config.attention_dropout
        
        self.W_k = nn.Linear(self.emb_dim,self.emb_dim) 
        self.W_q = nn.Linear(self.emb_dim,self.emb_dim) 
        self.W_v = nn.Linear(self.emb_dim,self.emb_dim) 
        self.W_o = nn.Linear(self.emb_dim,self.emb_dim) 
    
    def forward(self,embeddings:torch.Tensor) -> Tuple[torch.Tensor,Optional[torch.Tensor]]:
        ## (batch_size, Num_patches, emb_dim)
        batch_size, N_patches, _ = embeddings.size()
        queries = self.W_q(embeddings)
        keys    = self.W_k(embeddings)
        values  = self.W_v(embeddings)
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_heads, Num_patches, head_dim)
        queries = queries.view(batch_size,N_patches,self.num_heads,self.head_dim).transpose(1,2)
        keys    = keys.view(batch_size,N_patches,self.num_heads,self.head_dim).transpose(1,2)
        values  = values.view(batch_size,N_patches,self.num_heads,self.head_dim).transpose(1,2)
        ## (batch_size, Num_heads, Num_patches, head_dim) @ (batch_size, Num_heads, head_dim, Num_patches)
        attn_weights = (torch.matmul(queries,keys.transpose(2,3))*self.scale) 
        ## attention weights : (batch_size, Num_heads, Num_patches, Num_patches)
        ## softmax , applied by rows
        attn_weights = nn.functional.softmax(attn_weights,dim=-1,dtype=torch.float32).to(queries.dtype)
        attn_weights = nn.functional.dropout(attn_weights,p=self.dropout,training=self.training)
        ## (batch_size, Num_heads, Num_patches, Num_patches) @ (batch_size, Num_heads, Num_patches, head_dim)
        ## attention scores : (batch_size, Num_heads, Num_patches, head_dim)
        attn_scores = torch.matmul(attn_weights,values)
        ## attention scores : (batch_size, Num_heads, Num_patches, head_dim) ==> (batch_size, Num_patches, Num_heads, head_dim)
        attn_scores = attn_scores.transpose(1,2).contiguous()  
        ## (batch_size, Num_patches, Num_heads, head_dim) ==> (batch_size, Num_patches, emb_dim)
        attn_scores = attn_scores.reshape(batch_size,N_patches,self.emb_dim)
        ## Mixing the results of each head with the rest of heads that were calculated separately 
        output = self.W_o(attn_scores)
        
        return output , attn_weights 
    
        
class SigLIPVisionEncoderLayer(nn.Module):
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
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim) 
        embeddings, _ = self.self_attn(embeddings)
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim) 
        embeddings = embeddings + residual
        residual = embeddings
        embeddings = self.layer_norm2(embeddings)
        embeddings = self.mlp(embeddings)
        ## (batch_size, Num_patches, emb_dim) ==> (batch_size, Num_patches, emb_dim) 
        embeddings = embeddings + residual
        
        return embeddings
    
    
    
class SigLIPVisionEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.layers = nn.ModuleList(
            [SigLIPVisionEncoderLayer(self.config) for _ in range(config.num_hidden_layers)]
        )
    def forward(self,embeddings:torch.Tensor) -> torch.Tensor:
        for layer in self.layers : 
            embeddings = layer(embeddings)
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
        
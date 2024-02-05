# wget https://ommer-lab.com/files/latent-diffusion/celeba.zip

import torch
from diffusers import DiffusionPipeline

def change_weight_unet(HFmodel, SDmodel, ema=False):
    '''
    args: 
        :HFmodel: HF model state_dict (ex, DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").unet.state_dict())
        :SDmodel: SD model state_dict (ex, torch.load(open("/path/to/celebahq-ldm-vq-4.ckpt", 'rb'))["state_dict"])
    return:
        :HFmodel: changed HF model state_dict
        :HFChanged: list of changed weight key
    '''

    SD2HF = '''
    model.diffusion_model.input_blocks.0.0.weight,conv_in.weight
    model.diffusion_model.input_blocks.0.0.bias,conv_in.bias
    model.diffusion_model.time_embed.0.weight,time_embedding.linear_1.weight
    model.diffusion_model.time_embed.0.bias,time_embedding.linear_1.bias
    model.diffusion_model.time_embed.2.weight,time_embedding.linear_2.weight
    model.diffusion_model.time_embed.2.bias,time_embedding.linear_2.bias
    model.diffusion_model.input_blocks.1.0.in_layers.0.weight,down_blocks.0.resnets.0.norm1.weight
    model.diffusion_model.input_blocks.1.0.in_layers.0.bias,down_blocks.0.resnets.0.norm1.bias
    model.diffusion_model.input_blocks.1.0.in_layers.2.weight,down_blocks.0.resnets.0.conv1.weight
    model.diffusion_model.input_blocks.1.0.in_layers.2.bias,down_blocks.0.resnets.0.conv1.bias
    model.diffusion_model.input_blocks.1.0.emb_layers.1.weight,down_blocks.0.resnets.0.time_emb_proj.weight
    model.diffusion_model.input_blocks.1.0.emb_layers.1.bias,down_blocks.0.resnets.0.time_emb_proj.bias
    model.diffusion_model.input_blocks.1.0.out_layers.0.weight,down_blocks.0.resnets.0.norm2.weight
    model.diffusion_model.input_blocks.1.0.out_layers.0.bias,down_blocks.0.resnets.0.norm2.bias
    model.diffusion_model.input_blocks.1.0.out_layers.3.weight,down_blocks.0.resnets.0.conv2.weight
    model.diffusion_model.input_blocks.1.0.out_layers.3.bias,down_blocks.0.resnets.0.conv2.bias
    model.diffusion_model.input_blocks.2.0.in_layers.0.weight,down_blocks.0.resnets.1.norm1.weight
    model.diffusion_model.input_blocks.2.0.in_layers.0.bias,down_blocks.0.resnets.1.norm1.bias
    model.diffusion_model.input_blocks.2.0.in_layers.2.weight,down_blocks.0.resnets.1.conv1.weight
    model.diffusion_model.input_blocks.2.0.in_layers.2.bias,down_blocks.0.resnets.1.conv1.bias
    model.diffusion_model.input_blocks.2.0.emb_layers.1.weight,down_blocks.0.resnets.1.time_emb_proj.weight
    model.diffusion_model.input_blocks.2.0.emb_layers.1.bias,down_blocks.0.resnets.1.time_emb_proj.bias
    model.diffusion_model.input_blocks.2.0.out_layers.0.weight,down_blocks.0.resnets.1.norm2.weight
    model.diffusion_model.input_blocks.2.0.out_layers.0.bias,down_blocks.0.resnets.1.norm2.bias
    model.diffusion_model.input_blocks.2.0.out_layers.3.weight,down_blocks.0.resnets.1.conv2.weight
    model.diffusion_model.input_blocks.2.0.out_layers.3.bias,down_blocks.0.resnets.1.conv2.bias
    model.diffusion_model.input_blocks.3.0.op.weight,down_blocks.0.downsamplers.0.conv.weight
    model.diffusion_model.input_blocks.3.0.op.bias,down_blocks.0.downsamplers.0.conv.bias
    model.diffusion_model.input_blocks.4.1.norm.weight,down_blocks.1.attentions.0.group_norm.weight
    model.diffusion_model.input_blocks.4.1.norm.bias,down_blocks.1.attentions.0.group_norm.bias
    model.diffusion_model.input_blocks.4.1.proj_out.bias,down_blocks.1.attentions.0.to_out.0.bias
    model.diffusion_model.input_blocks.5.1.norm.weight,down_blocks.1.attentions.1.group_norm.weight
    model.diffusion_model.input_blocks.5.1.norm.bias,down_blocks.1.attentions.1.group_norm.bias
    model.diffusion_model.input_blocks.5.1.proj_out.bias,down_blocks.1.attentions.1.to_out.0.bias
    model.diffusion_model.input_blocks.4.0.in_layers.0.weight,down_blocks.1.resnets.0.norm1.weight
    model.diffusion_model.input_blocks.4.0.in_layers.0.bias,down_blocks.1.resnets.0.norm1.bias
    model.diffusion_model.input_blocks.4.0.in_layers.2.weight,down_blocks.1.resnets.0.conv1.weight
    model.diffusion_model.input_blocks.4.0.in_layers.2.bias,down_blocks.1.resnets.0.conv1.bias
    model.diffusion_model.input_blocks.4.0.emb_layers.1.weight,down_blocks.1.resnets.0.time_emb_proj.weight
    model.diffusion_model.input_blocks.4.0.emb_layers.1.bias,down_blocks.1.resnets.0.time_emb_proj.bias
    model.diffusion_model.input_blocks.4.0.out_layers.0.weight,down_blocks.1.resnets.0.norm2.weight
    model.diffusion_model.input_blocks.4.0.out_layers.0.bias,down_blocks.1.resnets.0.norm2.bias
    model.diffusion_model.input_blocks.4.0.out_layers.3.weight,down_blocks.1.resnets.0.conv2.weight
    model.diffusion_model.input_blocks.4.0.out_layers.3.bias,down_blocks.1.resnets.0.conv2.bias
    model.diffusion_model.input_blocks.4.0.skip_connection.weight,down_blocks.1.resnets.0.conv_shortcut.weight
    model.diffusion_model.input_blocks.4.0.skip_connection.bias,down_blocks.1.resnets.0.conv_shortcut.bias
    model.diffusion_model.input_blocks.5.0.in_layers.0.weight,down_blocks.1.resnets.1.norm1.weight
    model.diffusion_model.input_blocks.5.0.in_layers.0.bias,down_blocks.1.resnets.1.norm1.bias
    model.diffusion_model.input_blocks.5.0.in_layers.2.weight,down_blocks.1.resnets.1.conv1.weight
    model.diffusion_model.input_blocks.5.0.in_layers.2.bias,down_blocks.1.resnets.1.conv1.bias
    model.diffusion_model.input_blocks.5.0.emb_layers.1.weight,down_blocks.1.resnets.1.time_emb_proj.weight
    model.diffusion_model.input_blocks.5.0.emb_layers.1.bias,down_blocks.1.resnets.1.time_emb_proj.bias
    model.diffusion_model.input_blocks.5.0.out_layers.0.weight,down_blocks.1.resnets.1.norm2.weight
    model.diffusion_model.input_blocks.5.0.out_layers.0.bias,down_blocks.1.resnets.1.norm2.bias
    model.diffusion_model.input_blocks.5.0.out_layers.3.weight,down_blocks.1.resnets.1.conv2.weight
    model.diffusion_model.input_blocks.5.0.out_layers.3.bias,down_blocks.1.resnets.1.conv2.bias
    model.diffusion_model.input_blocks.6.0.op.weight,down_blocks.1.downsamplers.0.conv.weight
    model.diffusion_model.input_blocks.6.0.op.bias,down_blocks.1.downsamplers.0.conv.bias
    model.diffusion_model.input_blocks.7.1.norm.weight,down_blocks.2.attentions.0.group_norm.weight
    model.diffusion_model.input_blocks.7.1.norm.bias,down_blocks.2.attentions.0.group_norm.bias
    model.diffusion_model.input_blocks.7.1.proj_out.bias,down_blocks.2.attentions.0.to_out.0.bias
    model.diffusion_model.input_blocks.8.1.norm.weight,down_blocks.2.attentions.1.group_norm.weight
    model.diffusion_model.input_blocks.8.1.norm.bias,down_blocks.2.attentions.1.group_norm.bias
    model.diffusion_model.input_blocks.8.1.proj_out.bias,down_blocks.2.attentions.1.to_out.0.bias
    model.diffusion_model.input_blocks.7.0.in_layers.0.weight,down_blocks.2.resnets.0.norm1.weight
    model.diffusion_model.input_blocks.7.0.in_layers.0.bias,down_blocks.2.resnets.0.norm1.bias
    model.diffusion_model.input_blocks.7.0.in_layers.2.weight,down_blocks.2.resnets.0.conv1.weight
    model.diffusion_model.input_blocks.7.0.in_layers.2.bias,down_blocks.2.resnets.0.conv1.bias
    model.diffusion_model.input_blocks.7.0.emb_layers.1.weight,down_blocks.2.resnets.0.time_emb_proj.weight
    model.diffusion_model.input_blocks.7.0.emb_layers.1.bias,down_blocks.2.resnets.0.time_emb_proj.bias
    model.diffusion_model.input_blocks.7.0.out_layers.0.weight,down_blocks.2.resnets.0.norm2.weight
    model.diffusion_model.input_blocks.7.0.out_layers.0.bias,down_blocks.2.resnets.0.norm2.bias
    model.diffusion_model.input_blocks.7.0.out_layers.3.weight,down_blocks.2.resnets.0.conv2.weight
    model.diffusion_model.input_blocks.7.0.out_layers.3.bias,down_blocks.2.resnets.0.conv2.bias
    model.diffusion_model.input_blocks.7.0.skip_connection.weight,down_blocks.2.resnets.0.conv_shortcut.weight
    model.diffusion_model.input_blocks.7.0.skip_connection.bias,down_blocks.2.resnets.0.conv_shortcut.bias
    model.diffusion_model.input_blocks.8.0.in_layers.0.weight,down_blocks.2.resnets.1.norm1.weight
    model.diffusion_model.input_blocks.8.0.in_layers.0.bias,down_blocks.2.resnets.1.norm1.bias
    model.diffusion_model.input_blocks.8.0.in_layers.2.weight,down_blocks.2.resnets.1.conv1.weight
    model.diffusion_model.input_blocks.8.0.in_layers.2.bias,down_blocks.2.resnets.1.conv1.bias
    model.diffusion_model.input_blocks.8.0.emb_layers.1.weight,down_blocks.2.resnets.1.time_emb_proj.weight
    model.diffusion_model.input_blocks.8.0.emb_layers.1.bias,down_blocks.2.resnets.1.time_emb_proj.bias
    model.diffusion_model.input_blocks.8.0.out_layers.0.weight,down_blocks.2.resnets.1.norm2.weight
    model.diffusion_model.input_blocks.8.0.out_layers.0.bias,down_blocks.2.resnets.1.norm2.bias
    model.diffusion_model.input_blocks.8.0.out_layers.3.weight,down_blocks.2.resnets.1.conv2.weight
    model.diffusion_model.input_blocks.8.0.out_layers.3.bias,down_blocks.2.resnets.1.conv2.bias
    model.diffusion_model.input_blocks.9.0.op.weight,down_blocks.2.downsamplers.0.conv.weight
    model.diffusion_model.input_blocks.9.0.op.bias,down_blocks.2.downsamplers.0.conv.bias
    model.diffusion_model.input_blocks.10.1.norm.weight,down_blocks.3.attentions.0.group_norm.weight
    model.diffusion_model.input_blocks.10.1.norm.bias,down_blocks.3.attentions.0.group_norm.bias
    model.diffusion_model.input_blocks.10.1.proj_out.bias,down_blocks.3.attentions.0.to_out.0.bias
    model.diffusion_model.input_blocks.11.1.norm.weight,down_blocks.3.attentions.1.group_norm.weight
    model.diffusion_model.input_blocks.11.1.norm.bias,down_blocks.3.attentions.1.group_norm.bias
    model.diffusion_model.input_blocks.11.1.proj_out.bias,down_blocks.3.attentions.1.to_out.0.bias
    model.diffusion_model.input_blocks.10.0.in_layers.0.weight,down_blocks.3.resnets.0.norm1.weight
    model.diffusion_model.input_blocks.10.0.in_layers.0.bias,down_blocks.3.resnets.0.norm1.bias
    model.diffusion_model.input_blocks.10.0.in_layers.2.weight,down_blocks.3.resnets.0.conv1.weight
    model.diffusion_model.input_blocks.10.0.in_layers.2.bias,down_blocks.3.resnets.0.conv1.bias
    model.diffusion_model.input_blocks.10.0.emb_layers.1.weight,down_blocks.3.resnets.0.time_emb_proj.weight
    model.diffusion_model.input_blocks.10.0.emb_layers.1.bias,down_blocks.3.resnets.0.time_emb_proj.bias
    model.diffusion_model.input_blocks.10.0.out_layers.0.weight,down_blocks.3.resnets.0.norm2.weight
    model.diffusion_model.input_blocks.10.0.out_layers.0.bias,down_blocks.3.resnets.0.norm2.bias
    model.diffusion_model.input_blocks.10.0.out_layers.3.weight,down_blocks.3.resnets.0.conv2.weight
    model.diffusion_model.input_blocks.10.0.out_layers.3.bias,down_blocks.3.resnets.0.conv2.bias
    model.diffusion_model.input_blocks.10.0.skip_connection.weight,down_blocks.3.resnets.0.conv_shortcut.weight
    model.diffusion_model.input_blocks.10.0.skip_connection.bias,down_blocks.3.resnets.0.conv_shortcut.bias
    model.diffusion_model.input_blocks.11.0.in_layers.0.weight,down_blocks.3.resnets.1.norm1.weight
    model.diffusion_model.input_blocks.11.0.in_layers.0.bias,down_blocks.3.resnets.1.norm1.bias
    model.diffusion_model.input_blocks.11.0.in_layers.2.weight,down_blocks.3.resnets.1.conv1.weight
    model.diffusion_model.input_blocks.11.0.in_layers.2.bias,down_blocks.3.resnets.1.conv1.bias
    model.diffusion_model.input_blocks.11.0.emb_layers.1.weight,down_blocks.3.resnets.1.time_emb_proj.weight
    model.diffusion_model.input_blocks.11.0.emb_layers.1.bias,down_blocks.3.resnets.1.time_emb_proj.bias
    model.diffusion_model.input_blocks.11.0.out_layers.0.weight,down_blocks.3.resnets.1.norm2.weight
    model.diffusion_model.input_blocks.11.0.out_layers.0.bias,down_blocks.3.resnets.1.norm2.bias
    model.diffusion_model.input_blocks.11.0.out_layers.3.weight,down_blocks.3.resnets.1.conv2.weight
    model.diffusion_model.input_blocks.11.0.out_layers.3.bias,down_blocks.3.resnets.1.conv2.bias
    model.diffusion_model.output_blocks.0.1.norm.weight,up_blocks.0.attentions.0.group_norm.weight
    model.diffusion_model.output_blocks.0.1.norm.bias,up_blocks.0.attentions.0.group_norm.bias
    model.diffusion_model.output_blocks.0.1.proj_out.bias,up_blocks.0.attentions.0.to_out.0.bias
    model.diffusion_model.output_blocks.1.1.norm.weight,up_blocks.0.attentions.1.group_norm.weight
    model.diffusion_model.output_blocks.1.1.norm.bias,up_blocks.0.attentions.1.group_norm.bias
    model.diffusion_model.output_blocks.1.1.proj_out.bias,up_blocks.0.attentions.1.to_out.0.bias
    model.diffusion_model.output_blocks.2.1.norm.weight,up_blocks.0.attentions.2.group_norm.weight
    model.diffusion_model.output_blocks.2.1.norm.bias,up_blocks.0.attentions.2.group_norm.bias
    model.diffusion_model.output_blocks.2.1.proj_out.bias,up_blocks.0.attentions.2.to_out.0.bias
    model.diffusion_model.output_blocks.0.0.in_layers.0.weight,up_blocks.0.resnets.0.norm1.weight
    model.diffusion_model.output_blocks.0.0.in_layers.0.bias,up_blocks.0.resnets.0.norm1.bias
    model.diffusion_model.output_blocks.0.0.in_layers.2.weight,up_blocks.0.resnets.0.conv1.weight
    model.diffusion_model.output_blocks.0.0.in_layers.2.bias,up_blocks.0.resnets.0.conv1.bias
    model.diffusion_model.output_blocks.0.0.emb_layers.1.weight,up_blocks.0.resnets.0.time_emb_proj.weight
    model.diffusion_model.output_blocks.0.0.emb_layers.1.bias,up_blocks.0.resnets.0.time_emb_proj.bias
    model.diffusion_model.output_blocks.0.0.out_layers.0.weight,up_blocks.0.resnets.0.norm2.weight
    model.diffusion_model.output_blocks.0.0.out_layers.0.bias,up_blocks.0.resnets.0.norm2.bias
    model.diffusion_model.output_blocks.0.0.out_layers.3.weight,up_blocks.0.resnets.0.conv2.weight
    model.diffusion_model.output_blocks.0.0.out_layers.3.bias,up_blocks.0.resnets.0.conv2.bias
    model.diffusion_model.output_blocks.0.0.skip_connection.weight,up_blocks.0.resnets.0.conv_shortcut.weight
    model.diffusion_model.output_blocks.0.0.skip_connection.bias,up_blocks.0.resnets.0.conv_shortcut.bias
    model.diffusion_model.output_blocks.1.0.in_layers.0.weight,up_blocks.0.resnets.1.norm1.weight
    model.diffusion_model.output_blocks.1.0.in_layers.0.bias,up_blocks.0.resnets.1.norm1.bias
    model.diffusion_model.output_blocks.1.0.in_layers.2.weight,up_blocks.0.resnets.1.conv1.weight
    model.diffusion_model.output_blocks.1.0.in_layers.2.bias,up_blocks.0.resnets.1.conv1.bias
    model.diffusion_model.output_blocks.1.0.emb_layers.1.weight,up_blocks.0.resnets.1.time_emb_proj.weight
    model.diffusion_model.output_blocks.1.0.emb_layers.1.bias,up_blocks.0.resnets.1.time_emb_proj.bias
    model.diffusion_model.output_blocks.1.0.out_layers.0.weight,up_blocks.0.resnets.1.norm2.weight
    model.diffusion_model.output_blocks.1.0.out_layers.0.bias,up_blocks.0.resnets.1.norm2.bias
    model.diffusion_model.output_blocks.1.0.out_layers.3.weight,up_blocks.0.resnets.1.conv2.weight
    model.diffusion_model.output_blocks.1.0.out_layers.3.bias,up_blocks.0.resnets.1.conv2.bias
    model.diffusion_model.output_blocks.1.0.skip_connection.weight,up_blocks.0.resnets.1.conv_shortcut.weight
    model.diffusion_model.output_blocks.1.0.skip_connection.bias,up_blocks.0.resnets.1.conv_shortcut.bias
    model.diffusion_model.output_blocks.2.0.in_layers.0.weight,up_blocks.0.resnets.2.norm1.weight
    model.diffusion_model.output_blocks.2.0.in_layers.0.bias,up_blocks.0.resnets.2.norm1.bias
    model.diffusion_model.output_blocks.2.0.in_layers.2.weight,up_blocks.0.resnets.2.conv1.weight
    model.diffusion_model.output_blocks.2.0.in_layers.2.bias,up_blocks.0.resnets.2.conv1.bias
    model.diffusion_model.output_blocks.2.0.emb_layers.1.weight,up_blocks.0.resnets.2.time_emb_proj.weight
    model.diffusion_model.output_blocks.2.0.emb_layers.1.bias,up_blocks.0.resnets.2.time_emb_proj.bias
    model.diffusion_model.output_blocks.2.0.out_layers.0.weight,up_blocks.0.resnets.2.norm2.weight
    model.diffusion_model.output_blocks.2.0.out_layers.0.bias,up_blocks.0.resnets.2.norm2.bias
    model.diffusion_model.output_blocks.2.0.out_layers.3.weight,up_blocks.0.resnets.2.conv2.weight
    model.diffusion_model.output_blocks.2.0.out_layers.3.bias,up_blocks.0.resnets.2.conv2.bias
    model.diffusion_model.output_blocks.2.0.skip_connection.weight,up_blocks.0.resnets.2.conv_shortcut.weight
    model.diffusion_model.output_blocks.2.0.skip_connection.bias,up_blocks.0.resnets.2.conv_shortcut.bias
    model.diffusion_model.output_blocks.2.2.conv.weight,up_blocks.0.upsamplers.0.conv.weight
    model.diffusion_model.output_blocks.2.2.conv.bias,up_blocks.0.upsamplers.0.conv.bias
    model.diffusion_model.output_blocks.3.1.norm.weight,up_blocks.1.attentions.0.group_norm.weight
    model.diffusion_model.output_blocks.3.1.norm.bias,up_blocks.1.attentions.0.group_norm.bias
    model.diffusion_model.output_blocks.3.1.proj_out.bias,up_blocks.1.attentions.0.to_out.0.bias
    model.diffusion_model.output_blocks.4.1.norm.weight,up_blocks.1.attentions.1.group_norm.weight
    model.diffusion_model.output_blocks.4.1.norm.bias,up_blocks.1.attentions.1.group_norm.bias
    model.diffusion_model.output_blocks.4.1.proj_out.bias,up_blocks.1.attentions.1.to_out.0.bias
    model.diffusion_model.output_blocks.5.1.norm.weight,up_blocks.1.attentions.2.group_norm.weight
    model.diffusion_model.output_blocks.5.1.norm.bias,up_blocks.1.attentions.2.group_norm.bias
    model.diffusion_model.output_blocks.5.1.proj_out.bias,up_blocks.1.attentions.2.to_out.0.bias
    model.diffusion_model.output_blocks.3.0.in_layers.0.weight,up_blocks.1.resnets.0.norm1.weight
    model.diffusion_model.output_blocks.3.0.in_layers.0.bias,up_blocks.1.resnets.0.norm1.bias
    model.diffusion_model.output_blocks.3.0.in_layers.2.weight,up_blocks.1.resnets.0.conv1.weight
    model.diffusion_model.output_blocks.3.0.in_layers.2.bias,up_blocks.1.resnets.0.conv1.bias
    model.diffusion_model.output_blocks.3.0.emb_layers.1.weight,up_blocks.1.resnets.0.time_emb_proj.weight
    model.diffusion_model.output_blocks.3.0.emb_layers.1.bias,up_blocks.1.resnets.0.time_emb_proj.bias
    model.diffusion_model.output_blocks.3.0.out_layers.0.weight,up_blocks.1.resnets.0.norm2.weight
    model.diffusion_model.output_blocks.3.0.out_layers.0.bias,up_blocks.1.resnets.0.norm2.bias
    model.diffusion_model.output_blocks.3.0.out_layers.3.weight,up_blocks.1.resnets.0.conv2.weight
    model.diffusion_model.output_blocks.3.0.out_layers.3.bias,up_blocks.1.resnets.0.conv2.bias
    model.diffusion_model.output_blocks.3.0.skip_connection.weight,up_blocks.1.resnets.0.conv_shortcut.weight
    model.diffusion_model.output_blocks.3.0.skip_connection.bias,up_blocks.1.resnets.0.conv_shortcut.bias
    model.diffusion_model.output_blocks.4.0.in_layers.0.weight,up_blocks.1.resnets.1.norm1.weight
    model.diffusion_model.output_blocks.4.0.in_layers.0.bias,up_blocks.1.resnets.1.norm1.bias
    model.diffusion_model.output_blocks.4.0.in_layers.2.weight,up_blocks.1.resnets.1.conv1.weight
    model.diffusion_model.output_blocks.4.0.in_layers.2.bias,up_blocks.1.resnets.1.conv1.bias
    model.diffusion_model.output_blocks.4.0.emb_layers.1.weight,up_blocks.1.resnets.1.time_emb_proj.weight
    model.diffusion_model.output_blocks.4.0.emb_layers.1.bias,up_blocks.1.resnets.1.time_emb_proj.bias
    model.diffusion_model.output_blocks.4.0.out_layers.0.weight,up_blocks.1.resnets.1.norm2.weight
    model.diffusion_model.output_blocks.4.0.out_layers.0.bias,up_blocks.1.resnets.1.norm2.bias
    model.diffusion_model.output_blocks.4.0.out_layers.3.weight,up_blocks.1.resnets.1.conv2.weight
    model.diffusion_model.output_blocks.4.0.out_layers.3.bias,up_blocks.1.resnets.1.conv2.bias
    model.diffusion_model.output_blocks.4.0.skip_connection.weight,up_blocks.1.resnets.1.conv_shortcut.weight
    model.diffusion_model.output_blocks.4.0.skip_connection.bias,up_blocks.1.resnets.1.conv_shortcut.bias
    model.diffusion_model.output_blocks.5.0.in_layers.0.weight,up_blocks.1.resnets.2.norm1.weight
    model.diffusion_model.output_blocks.5.0.in_layers.0.bias,up_blocks.1.resnets.2.norm1.bias
    model.diffusion_model.output_blocks.5.0.in_layers.2.weight,up_blocks.1.resnets.2.conv1.weight
    model.diffusion_model.output_blocks.5.0.in_layers.2.bias,up_blocks.1.resnets.2.conv1.bias
    model.diffusion_model.output_blocks.5.0.emb_layers.1.weight,up_blocks.1.resnets.2.time_emb_proj.weight
    model.diffusion_model.output_blocks.5.0.emb_layers.1.bias,up_blocks.1.resnets.2.time_emb_proj.bias
    model.diffusion_model.output_blocks.5.0.out_layers.0.weight,up_blocks.1.resnets.2.norm2.weight
    model.diffusion_model.output_blocks.5.0.out_layers.0.bias,up_blocks.1.resnets.2.norm2.bias
    model.diffusion_model.output_blocks.5.0.out_layers.3.weight,up_blocks.1.resnets.2.conv2.weight
    model.diffusion_model.output_blocks.5.0.out_layers.3.bias,up_blocks.1.resnets.2.conv2.bias
    model.diffusion_model.output_blocks.5.0.skip_connection.weight,up_blocks.1.resnets.2.conv_shortcut.weight
    model.diffusion_model.output_blocks.5.0.skip_connection.bias,up_blocks.1.resnets.2.conv_shortcut.bias
    model.diffusion_model.output_blocks.5.2.conv.weight,up_blocks.1.upsamplers.0.conv.weight
    model.diffusion_model.output_blocks.5.2.conv.bias,up_blocks.1.upsamplers.0.conv.bias
    model.diffusion_model.output_blocks.6.1.norm.weight,up_blocks.2.attentions.0.group_norm.weight
    model.diffusion_model.output_blocks.6.1.norm.bias,up_blocks.2.attentions.0.group_norm.bias
    model.diffusion_model.output_blocks.6.1.proj_out.bias,up_blocks.2.attentions.0.to_out.0.bias
    model.diffusion_model.output_blocks.7.1.norm.weight,up_blocks.2.attentions.1.group_norm.weight
    model.diffusion_model.output_blocks.7.1.norm.bias,up_blocks.2.attentions.1.group_norm.bias
    model.diffusion_model.output_blocks.7.1.proj_out.bias,up_blocks.2.attentions.1.to_out.0.bias
    model.diffusion_model.output_blocks.8.1.norm.weight,up_blocks.2.attentions.2.group_norm.weight
    model.diffusion_model.output_blocks.8.1.norm.bias,up_blocks.2.attentions.2.group_norm.bias
    model.diffusion_model.output_blocks.8.1.proj_out.bias,up_blocks.2.attentions.2.to_out.0.bias
    model.diffusion_model.output_blocks.6.0.in_layers.0.weight,up_blocks.2.resnets.0.norm1.weight
    model.diffusion_model.output_blocks.6.0.in_layers.0.bias,up_blocks.2.resnets.0.norm1.bias
    model.diffusion_model.output_blocks.6.0.in_layers.2.weight,up_blocks.2.resnets.0.conv1.weight
    model.diffusion_model.output_blocks.6.0.in_layers.2.bias,up_blocks.2.resnets.0.conv1.bias
    model.diffusion_model.output_blocks.6.0.emb_layers.1.weight,up_blocks.2.resnets.0.time_emb_proj.weight
    model.diffusion_model.output_blocks.6.0.emb_layers.1.bias,up_blocks.2.resnets.0.time_emb_proj.bias
    model.diffusion_model.output_blocks.6.0.out_layers.0.weight,up_blocks.2.resnets.0.norm2.weight
    model.diffusion_model.output_blocks.6.0.out_layers.0.bias,up_blocks.2.resnets.0.norm2.bias
    model.diffusion_model.output_blocks.6.0.out_layers.3.weight,up_blocks.2.resnets.0.conv2.weight
    model.diffusion_model.output_blocks.6.0.out_layers.3.bias,up_blocks.2.resnets.0.conv2.bias
    model.diffusion_model.output_blocks.6.0.skip_connection.weight,up_blocks.2.resnets.0.conv_shortcut.weight
    model.diffusion_model.output_blocks.6.0.skip_connection.bias,up_blocks.2.resnets.0.conv_shortcut.bias
    model.diffusion_model.output_blocks.7.0.in_layers.0.weight,up_blocks.2.resnets.1.norm1.weight
    model.diffusion_model.output_blocks.7.0.in_layers.0.bias,up_blocks.2.resnets.1.norm1.bias
    model.diffusion_model.output_blocks.7.0.in_layers.2.weight,up_blocks.2.resnets.1.conv1.weight
    model.diffusion_model.output_blocks.7.0.in_layers.2.bias,up_blocks.2.resnets.1.conv1.bias
    model.diffusion_model.output_blocks.7.0.emb_layers.1.weight,up_blocks.2.resnets.1.time_emb_proj.weight
    model.diffusion_model.output_blocks.7.0.emb_layers.1.bias,up_blocks.2.resnets.1.time_emb_proj.bias
    model.diffusion_model.output_blocks.7.0.out_layers.0.weight,up_blocks.2.resnets.1.norm2.weight
    model.diffusion_model.output_blocks.7.0.out_layers.0.bias,up_blocks.2.resnets.1.norm2.bias
    model.diffusion_model.output_blocks.7.0.out_layers.3.weight,up_blocks.2.resnets.1.conv2.weight
    model.diffusion_model.output_blocks.7.0.out_layers.3.bias,up_blocks.2.resnets.1.conv2.bias
    model.diffusion_model.output_blocks.7.0.skip_connection.weight,up_blocks.2.resnets.1.conv_shortcut.weight
    model.diffusion_model.output_blocks.7.0.skip_connection.bias,up_blocks.2.resnets.1.conv_shortcut.bias
    model.diffusion_model.output_blocks.8.0.in_layers.0.weight,up_blocks.2.resnets.2.norm1.weight
    model.diffusion_model.output_blocks.8.0.in_layers.0.bias,up_blocks.2.resnets.2.norm1.bias
    model.diffusion_model.output_blocks.8.0.in_layers.2.weight,up_blocks.2.resnets.2.conv1.weight
    model.diffusion_model.output_blocks.8.0.in_layers.2.bias,up_blocks.2.resnets.2.conv1.bias
    model.diffusion_model.output_blocks.8.0.emb_layers.1.weight,up_blocks.2.resnets.2.time_emb_proj.weight
    model.diffusion_model.output_blocks.8.0.emb_layers.1.bias,up_blocks.2.resnets.2.time_emb_proj.bias
    model.diffusion_model.output_blocks.8.0.out_layers.0.weight,up_blocks.2.resnets.2.norm2.weight
    model.diffusion_model.output_blocks.8.0.out_layers.0.bias,up_blocks.2.resnets.2.norm2.bias
    model.diffusion_model.output_blocks.8.0.out_layers.3.weight,up_blocks.2.resnets.2.conv2.weight
    model.diffusion_model.output_blocks.8.0.out_layers.3.bias,up_blocks.2.resnets.2.conv2.bias
    model.diffusion_model.output_blocks.8.0.skip_connection.weight,up_blocks.2.resnets.2.conv_shortcut.weight
    model.diffusion_model.output_blocks.8.0.skip_connection.bias,up_blocks.2.resnets.2.conv_shortcut.bias
    model.diffusion_model.output_blocks.8.2.conv.weight,up_blocks.2.upsamplers.0.conv.weight
    model.diffusion_model.output_blocks.8.2.conv.bias,up_blocks.2.upsamplers.0.conv.bias
    model.diffusion_model.output_blocks.9.0.in_layers.0.weight,up_blocks.3.resnets.0.norm1.weight
    model.diffusion_model.output_blocks.9.0.in_layers.0.bias,up_blocks.3.resnets.0.norm1.bias
    model.diffusion_model.output_blocks.9.0.in_layers.2.weight,up_blocks.3.resnets.0.conv1.weight
    model.diffusion_model.output_blocks.9.0.in_layers.2.bias,up_blocks.3.resnets.0.conv1.bias
    model.diffusion_model.output_blocks.9.0.emb_layers.1.weight,up_blocks.3.resnets.0.time_emb_proj.weight
    model.diffusion_model.output_blocks.9.0.emb_layers.1.bias,up_blocks.3.resnets.0.time_emb_proj.bias
    model.diffusion_model.output_blocks.9.0.out_layers.0.weight,up_blocks.3.resnets.0.norm2.weight
    model.diffusion_model.output_blocks.9.0.out_layers.0.bias,up_blocks.3.resnets.0.norm2.bias
    model.diffusion_model.output_blocks.9.0.out_layers.3.weight,up_blocks.3.resnets.0.conv2.weight
    model.diffusion_model.output_blocks.9.0.out_layers.3.bias,up_blocks.3.resnets.0.conv2.bias
    model.diffusion_model.output_blocks.9.0.skip_connection.weight,up_blocks.3.resnets.0.conv_shortcut.weight
    model.diffusion_model.output_blocks.9.0.skip_connection.bias,up_blocks.3.resnets.0.conv_shortcut.bias
    model.diffusion_model.output_blocks.10.0.in_layers.0.weight,up_blocks.3.resnets.1.norm1.weight
    model.diffusion_model.output_blocks.10.0.in_layers.0.bias,up_blocks.3.resnets.1.norm1.bias
    model.diffusion_model.output_blocks.10.0.in_layers.2.weight,up_blocks.3.resnets.1.conv1.weight
    model.diffusion_model.output_blocks.10.0.in_layers.2.bias,up_blocks.3.resnets.1.conv1.bias
    model.diffusion_model.output_blocks.10.0.emb_layers.1.weight,up_blocks.3.resnets.1.time_emb_proj.weight
    model.diffusion_model.output_blocks.10.0.emb_layers.1.bias,up_blocks.3.resnets.1.time_emb_proj.bias
    model.diffusion_model.output_blocks.10.0.out_layers.0.weight,up_blocks.3.resnets.1.norm2.weight
    model.diffusion_model.output_blocks.10.0.out_layers.0.bias,up_blocks.3.resnets.1.norm2.bias
    model.diffusion_model.output_blocks.10.0.out_layers.3.weight,up_blocks.3.resnets.1.conv2.weight
    model.diffusion_model.output_blocks.10.0.out_layers.3.bias,up_blocks.3.resnets.1.conv2.bias
    model.diffusion_model.output_blocks.10.0.skip_connection.weight,up_blocks.3.resnets.1.conv_shortcut.weight
    model.diffusion_model.output_blocks.10.0.skip_connection.bias,up_blocks.3.resnets.1.conv_shortcut.bias
    model.diffusion_model.output_blocks.11.0.in_layers.0.weight,up_blocks.3.resnets.2.norm1.weight
    model.diffusion_model.output_blocks.11.0.in_layers.0.bias,up_blocks.3.resnets.2.norm1.bias
    model.diffusion_model.output_blocks.11.0.in_layers.2.weight,up_blocks.3.resnets.2.conv1.weight
    model.diffusion_model.output_blocks.11.0.in_layers.2.bias,up_blocks.3.resnets.2.conv1.bias
    model.diffusion_model.output_blocks.11.0.emb_layers.1.weight,up_blocks.3.resnets.2.time_emb_proj.weight
    model.diffusion_model.output_blocks.11.0.emb_layers.1.bias,up_blocks.3.resnets.2.time_emb_proj.bias
    model.diffusion_model.output_blocks.11.0.out_layers.0.weight,up_blocks.3.resnets.2.norm2.weight
    model.diffusion_model.output_blocks.11.0.out_layers.0.bias,up_blocks.3.resnets.2.norm2.bias
    model.diffusion_model.output_blocks.11.0.out_layers.3.weight,up_blocks.3.resnets.2.conv2.weight
    model.diffusion_model.output_blocks.11.0.out_layers.3.bias,up_blocks.3.resnets.2.conv2.bias
    model.diffusion_model.output_blocks.11.0.skip_connection.weight,up_blocks.3.resnets.2.conv_shortcut.weight
    model.diffusion_model.output_blocks.11.0.skip_connection.bias,up_blocks.3.resnets.2.conv_shortcut.bias
    model.diffusion_model.middle_block.1.norm.weight,mid_block.attentions.0.group_norm.weight
    model.diffusion_model.middle_block.1.norm.bias,mid_block.attentions.0.group_norm.bias
    model.diffusion_model.middle_block.1.proj_out.bias,mid_block.attentions.0.to_out.0.bias
    model.diffusion_model.middle_block.0.in_layers.0.weight,mid_block.resnets.0.norm1.weight
    model.diffusion_model.middle_block.0.in_layers.0.bias,mid_block.resnets.0.norm1.bias
    model.diffusion_model.middle_block.0.in_layers.2.weight,mid_block.resnets.0.conv1.weight
    model.diffusion_model.middle_block.0.in_layers.2.bias,mid_block.resnets.0.conv1.bias
    model.diffusion_model.middle_block.0.emb_layers.1.weight,mid_block.resnets.0.time_emb_proj.weight
    model.diffusion_model.middle_block.0.emb_layers.1.bias,mid_block.resnets.0.time_emb_proj.bias
    model.diffusion_model.middle_block.0.out_layers.0.weight,mid_block.resnets.0.norm2.weight
    model.diffusion_model.middle_block.0.out_layers.0.bias,mid_block.resnets.0.norm2.bias
    model.diffusion_model.middle_block.0.out_layers.3.weight,mid_block.resnets.0.conv2.weight
    model.diffusion_model.middle_block.0.out_layers.3.bias,mid_block.resnets.0.conv2.bias
    model.diffusion_model.middle_block.2.in_layers.0.weight,mid_block.resnets.1.norm1.weight
    model.diffusion_model.middle_block.2.in_layers.0.bias,mid_block.resnets.1.norm1.bias
    model.diffusion_model.middle_block.2.in_layers.2.weight,mid_block.resnets.1.conv1.weight
    model.diffusion_model.middle_block.2.in_layers.2.bias,mid_block.resnets.1.conv1.bias
    model.diffusion_model.middle_block.2.emb_layers.1.weight,mid_block.resnets.1.time_emb_proj.weight
    model.diffusion_model.middle_block.2.emb_layers.1.bias,mid_block.resnets.1.time_emb_proj.bias
    model.diffusion_model.middle_block.2.out_layers.0.weight,mid_block.resnets.1.norm2.weight
    model.diffusion_model.middle_block.2.out_layers.0.bias,mid_block.resnets.1.norm2.bias
    model.diffusion_model.middle_block.2.out_layers.3.weight,mid_block.resnets.1.conv2.weight
    model.diffusion_model.middle_block.2.out_layers.3.bias,mid_block.resnets.1.conv2.bias
    model.diffusion_model.out.0.weight,conv_norm_out.weight
    model.diffusion_model.out.0.bias,conv_norm_out.bias
    model.diffusion_model.out.2.weight,conv_out.weight
    model.diffusion_model.out.2.bias,conv_out.bias'''

    if ema:
        def rename_to_ema(name):
            name.replace("model.", "")
            name.replace(".", "")
            name.replace("diffusion_model", "model_ema.diffusion_model")
        SD2HF = {rename_to_ema(x.split(',')[0].strip()): x.split(',')[1].strip() for x in SD2HF.strip().split('\n')}
                    
    else:
        SD2HF = {x.split(',')[0].strip(): x.split(',')[1].strip() for x in SD2HF.strip().split('\n')}

    SDModified =  '''
    model.diffusion_model.input_blocks.4.1.qkv.weight
    model.diffusion_model.input_blocks.4.1.qkv.bias
    model.diffusion_model.input_blocks.4.1.proj_out.weight
    model.diffusion_model.input_blocks.5.1.qkv.weight
    model.diffusion_model.input_blocks.5.1.qkv.bias
    model.diffusion_model.input_blocks.5.1.proj_out.weight
    model.diffusion_model.input_blocks.7.1.qkv.weight
    model.diffusion_model.input_blocks.7.1.qkv.bias
    model.diffusion_model.input_blocks.7.1.proj_out.weight
    model.diffusion_model.input_blocks.8.1.qkv.weight
    model.diffusion_model.input_blocks.8.1.qkv.bias
    model.diffusion_model.input_blocks.8.1.proj_out.weight
    model.diffusion_model.input_blocks.10.1.qkv.weight
    model.diffusion_model.input_blocks.10.1.qkv.bias
    model.diffusion_model.input_blocks.10.1.proj_out.weight
    model.diffusion_model.input_blocks.11.1.qkv.weight
    model.diffusion_model.input_blocks.11.1.qkv.bias
    model.diffusion_model.input_blocks.11.1.proj_out.weight
    model.diffusion_model.middle_block.1.qkv.weight
    model.diffusion_model.middle_block.1.qkv.bias
    model.diffusion_model.middle_block.1.proj_out.weight
    model.diffusion_model.output_blocks.0.1.qkv.weight
    model.diffusion_model.output_blocks.0.1.qkv.bias
    model.diffusion_model.output_blocks.0.1.proj_out.weight
    model.diffusion_model.output_blocks.1.1.qkv.weight
    model.diffusion_model.output_blocks.1.1.qkv.bias
    model.diffusion_model.output_blocks.1.1.proj_out.weight
    model.diffusion_model.output_blocks.2.1.qkv.weight
    model.diffusion_model.output_blocks.2.1.qkv.bias
    model.diffusion_model.output_blocks.2.1.proj_out.weight
    model.diffusion_model.output_blocks.3.1.qkv.weight
    model.diffusion_model.output_blocks.3.1.qkv.bias
    model.diffusion_model.output_blocks.3.1.proj_out.weight
    model.diffusion_model.output_blocks.4.1.qkv.weight
    model.diffusion_model.output_blocks.4.1.qkv.bias
    model.diffusion_model.output_blocks.4.1.proj_out.weight
    model.diffusion_model.output_blocks.5.1.qkv.weight
    model.diffusion_model.output_blocks.5.1.qkv.bias
    model.diffusion_model.output_blocks.5.1.proj_out.weight
    model.diffusion_model.output_blocks.6.1.qkv.weight
    model.diffusion_model.output_blocks.6.1.qkv.bias
    model.diffusion_model.output_blocks.6.1.proj_out.weight
    model.diffusion_model.output_blocks.7.1.qkv.weight
    model.diffusion_model.output_blocks.7.1.qkv.bias
    model.diffusion_model.output_blocks.7.1.proj_out.weight
    model.diffusion_model.output_blocks.8.1.qkv.weight
    model.diffusion_model.output_blocks.8.1.qkv.bias
    model.diffusion_model.output_blocks.8.1.proj_out.weight
    '''.strip().split()

    HFModified = '''
    down_blocks.1.attentions.0.to_q.weight
    down_blocks.1.attentions.0.to_q.bias
    down_blocks.1.attentions.0.to_k.weight
    down_blocks.1.attentions.0.to_k.bias
    down_blocks.1.attentions.0.to_v.weight
    down_blocks.1.attentions.0.to_v.bias
    down_blocks.1.attentions.0.to_out.0.weight
    down_blocks.1.attentions.1.to_q.weight
    down_blocks.1.attentions.1.to_q.bias
    down_blocks.1.attentions.1.to_k.weight
    down_blocks.1.attentions.1.to_k.bias
    down_blocks.1.attentions.1.to_v.weight
    down_blocks.1.attentions.1.to_v.bias
    down_blocks.1.attentions.1.to_out.0.weight
    down_blocks.2.attentions.0.to_q.weight
    down_blocks.2.attentions.0.to_q.bias
    down_blocks.2.attentions.0.to_k.weight
    down_blocks.2.attentions.0.to_k.bias
    down_blocks.2.attentions.0.to_v.weight
    down_blocks.2.attentions.0.to_v.bias
    down_blocks.2.attentions.0.to_out.0.weight
    down_blocks.2.attentions.1.to_q.weight
    down_blocks.2.attentions.1.to_q.bias
    down_blocks.2.attentions.1.to_k.weight
    down_blocks.2.attentions.1.to_k.bias
    down_blocks.2.attentions.1.to_v.weight
    down_blocks.2.attentions.1.to_v.bias
    down_blocks.2.attentions.1.to_out.0.weight
    down_blocks.3.attentions.0.to_q.weight
    down_blocks.3.attentions.0.to_q.bias
    down_blocks.3.attentions.0.to_k.weight
    down_blocks.3.attentions.0.to_k.bias
    down_blocks.3.attentions.0.to_v.weight
    down_blocks.3.attentions.0.to_v.bias
    down_blocks.3.attentions.0.to_out.0.weight
    down_blocks.3.attentions.1.to_q.weight
    down_blocks.3.attentions.1.to_q.bias
    down_blocks.3.attentions.1.to_k.weight
    down_blocks.3.attentions.1.to_k.bias
    down_blocks.3.attentions.1.to_v.weight
    down_blocks.3.attentions.1.to_v.bias
    down_blocks.3.attentions.1.to_out.0.weight
    mid_block.attentions.0.to_q.weight
    mid_block.attentions.0.to_q.bias
    mid_block.attentions.0.to_k.weight
    mid_block.attentions.0.to_k.bias
    mid_block.attentions.0.to_v.weight
    mid_block.attentions.0.to_v.bias
    mid_block.attentions.0.to_out.0.weight
    up_blocks.0.attentions.0.to_q.weight
    up_blocks.0.attentions.0.to_q.bias
    up_blocks.0.attentions.0.to_k.weight
    up_blocks.0.attentions.0.to_k.bias
    up_blocks.0.attentions.0.to_v.weight
    up_blocks.0.attentions.0.to_v.bias
    up_blocks.0.attentions.0.to_out.0.weight
    up_blocks.0.attentions.1.to_q.weight
    up_blocks.0.attentions.1.to_q.bias
    up_blocks.0.attentions.1.to_k.weight
    up_blocks.0.attentions.1.to_k.bias
    up_blocks.0.attentions.1.to_v.weight
    up_blocks.0.attentions.1.to_v.bias
    up_blocks.0.attentions.1.to_out.0.weight
    up_blocks.0.attentions.2.to_q.weight
    up_blocks.0.attentions.2.to_q.bias
    up_blocks.0.attentions.2.to_k.weight
    up_blocks.0.attentions.2.to_k.bias
    up_blocks.0.attentions.2.to_v.weight
    up_blocks.0.attentions.2.to_v.bias
    up_blocks.0.attentions.2.to_out.0.weight
    up_blocks.1.attentions.0.to_q.weight
    up_blocks.1.attentions.0.to_q.bias
    up_blocks.1.attentions.0.to_k.weight
    up_blocks.1.attentions.0.to_k.bias
    up_blocks.1.attentions.0.to_v.weight
    up_blocks.1.attentions.0.to_v.bias
    up_blocks.1.attentions.0.to_out.0.weight
    up_blocks.1.attentions.1.to_q.weight
    up_blocks.1.attentions.1.to_q.bias
    up_blocks.1.attentions.1.to_k.weight
    up_blocks.1.attentions.1.to_k.bias
    up_blocks.1.attentions.1.to_v.weight
    up_blocks.1.attentions.1.to_v.bias
    up_blocks.1.attentions.1.to_out.0.weight
    up_blocks.1.attentions.2.to_q.weight
    up_blocks.1.attentions.2.to_q.bias
    up_blocks.1.attentions.2.to_k.weight
    up_blocks.1.attentions.2.to_k.bias
    up_blocks.1.attentions.2.to_v.weight
    up_blocks.1.attentions.2.to_v.bias
    up_blocks.1.attentions.2.to_out.0.weight
    up_blocks.2.attentions.0.to_q.weight
    up_blocks.2.attentions.0.to_q.bias
    up_blocks.2.attentions.0.to_k.weight
    up_blocks.2.attentions.0.to_k.bias
    up_blocks.2.attentions.0.to_v.weight
    up_blocks.2.attentions.0.to_v.bias
    up_blocks.2.attentions.0.to_out.0.weight
    up_blocks.2.attentions.1.to_q.weight
    up_blocks.2.attentions.1.to_q.bias
    up_blocks.2.attentions.1.to_k.weight
    up_blocks.2.attentions.1.to_k.bias
    up_blocks.2.attentions.1.to_v.weight
    up_blocks.2.attentions.1.to_v.bias
    up_blocks.2.attentions.1.to_out.0.weight
    up_blocks.2.attentions.2.to_q.weight
    up_blocks.2.attentions.2.to_q.bias
    up_blocks.2.attentions.2.to_k.weight
    up_blocks.2.attentions.2.to_k.bias
    up_blocks.2.attentions.2.to_v.weight
    up_blocks.2.attentions.2.to_v.bias
    up_blocks.2.attentions.2.to_out.0.weight
    '''.strip().split()
    
    HFindex = 0
    SDindex = 0
    while SDindex < len(SDModified):
        SDkey = SDModified[SDindex:SDindex+3]
        SD2HF[SDkey[0]] = HFModified[HFindex:HFindex+5:2]
        SD2HF[SDkey[1]] = HFModified[HFindex+1:HFindex+6:2]
        SD2HF[SDkey[2]] = HFModified[HFindex+6:HFindex+7]
        HFindex += 7
        SDindex += 3

    
    HFChanged = []
    for key in SD2HF:
        if type(SD2HF[key]) == str:
            # assert HFmodel[SD2HF[key]].equal(SDmodel[key])
            HFmodel[SD2HF[key]] = SDmodel[key]
            HFChanged.append(SD2HF[key])
        
        else:
            if len(SD2HF[key]) == 1:
                # assert HFmodel[SD2HF[key][0]].equal(SDmodel[key].squeeze())
                HFmodel[SD2HF[key][0]] = SDmodel[key].squeeze()
                HFChanged += SD2HF[key]
            elif len(SD2HF[key]) == 3:
                tensors = SDmodel[key].split(32)
                tensor_q, tensor_k, tensor_v = [], [], []
                for i in range(len(tensors)):
                    if i % 3 == 0:
                        tensor_q.append(tensors[i])
                    elif i % 3 == 1:
                        tensor_k.append(tensors[i])
                    else:
                        tensor_v.append(tensors[i])
                tensor_q = torch.cat(tensor_q, dim=0)
                tensor_k = torch.cat(tensor_k, dim=0)
                tensor_v = torch.cat(tensor_v, dim=0)

                # assert HFmodel[SD2HF[key][0]].equal(tensor_q.squeeze())
                # assert HFmodel[SD2HF[key][1]].equal(tensor_k.squeeze())
                # assert HFmodel[SD2HF[key][2]].equal(tensor_v.squeeze())
                HFmodel[SD2HF[key][0]] = tensor_q.squeeze()
                HFmodel[SD2HF[key][1]] = tensor_k.squeeze()
                HFmodel[SD2HF[key][2]] = tensor_v.squeeze()
                HFChanged += SD2HF[key]

    assert sorted(list(HFmodel.keys())) == sorted(HFChanged)
    return HFmodel, HFChanged

def change_weight_vqvae(HFmodel, SDmodel):
    HF2SD = '''
    encoder.conv_in.weight,first_stage_model.encoder.conv_in.weight
    encoder.conv_in.bias,first_stage_model.encoder.conv_in.bias
    encoder.down_blocks.0.resnets.0.norm1.weight,first_stage_model.encoder.down.0.block.0.norm1.weight
    encoder.down_blocks.0.resnets.0.norm1.bias,first_stage_model.encoder.down.0.block.0.norm1.bias
    encoder.down_blocks.0.resnets.0.conv1.weight,first_stage_model.encoder.down.0.block.0.conv1.weight
    encoder.down_blocks.0.resnets.0.conv1.bias,first_stage_model.encoder.down.0.block.0.conv1.bias
    encoder.down_blocks.0.resnets.0.norm2.weight,first_stage_model.encoder.down.0.block.0.norm2.weight
    encoder.down_blocks.0.resnets.0.norm2.bias,first_stage_model.encoder.down.0.block.0.norm2.bias
    encoder.down_blocks.0.resnets.0.conv2.weight,first_stage_model.encoder.down.0.block.0.conv2.weight
    encoder.down_blocks.0.resnets.0.conv2.bias,first_stage_model.encoder.down.0.block.0.conv2.bias
    encoder.down_blocks.0.resnets.1.norm1.weight,first_stage_model.encoder.down.0.block.1.norm1.weight
    encoder.down_blocks.0.resnets.1.norm1.bias,first_stage_model.encoder.down.0.block.1.norm1.bias
    encoder.down_blocks.0.resnets.1.conv1.weight,first_stage_model.encoder.down.0.block.1.conv1.weight
    encoder.down_blocks.0.resnets.1.conv1.bias,first_stage_model.encoder.down.0.block.1.conv1.bias
    encoder.down_blocks.0.resnets.1.norm2.weight,first_stage_model.encoder.down.0.block.1.norm2.weight
    encoder.down_blocks.0.resnets.1.norm2.bias,first_stage_model.encoder.down.0.block.1.norm2.bias
    encoder.down_blocks.0.resnets.1.conv2.weight,first_stage_model.encoder.down.0.block.1.conv2.weight
    encoder.down_blocks.0.resnets.1.conv2.bias,first_stage_model.encoder.down.0.block.1.conv2.bias
    encoder.down_blocks.0.downsamplers.0.conv.weight,first_stage_model.encoder.down.0.downsample.conv.weight
    encoder.down_blocks.0.downsamplers.0.conv.bias,first_stage_model.encoder.down.0.downsample.conv.bias
    encoder.down_blocks.1.resnets.0.norm1.weight,first_stage_model.encoder.down.1.block.0.norm1.weight
    encoder.down_blocks.1.resnets.0.norm1.bias,first_stage_model.encoder.down.1.block.0.norm1.bias
    encoder.down_blocks.1.resnets.0.conv1.weight,first_stage_model.encoder.down.1.block.0.conv1.weight
    encoder.down_blocks.1.resnets.0.conv1.bias,first_stage_model.encoder.down.1.block.0.conv1.bias
    encoder.down_blocks.1.resnets.0.norm2.weight,first_stage_model.encoder.down.1.block.0.norm2.weight
    encoder.down_blocks.1.resnets.0.norm2.bias,first_stage_model.encoder.down.1.block.0.norm2.bias
    encoder.down_blocks.1.resnets.0.conv2.weight,first_stage_model.encoder.down.1.block.0.conv2.weight
    encoder.down_blocks.1.resnets.0.conv2.bias,first_stage_model.encoder.down.1.block.0.conv2.bias
    encoder.down_blocks.1.resnets.0.conv_shortcut.weight,first_stage_model.encoder.down.1.block.0.nin_shortcut.weight
    encoder.down_blocks.1.resnets.0.conv_shortcut.bias,first_stage_model.encoder.down.1.block.0.nin_shortcut.bias
    encoder.down_blocks.1.resnets.1.norm1.weight,first_stage_model.encoder.down.1.block.1.norm1.weight
    encoder.down_blocks.1.resnets.1.norm1.bias,first_stage_model.encoder.down.1.block.1.norm1.bias
    encoder.down_blocks.1.resnets.1.conv1.weight,first_stage_model.encoder.down.1.block.1.conv1.weight
    encoder.down_blocks.1.resnets.1.conv1.bias,first_stage_model.encoder.down.1.block.1.conv1.bias
    encoder.down_blocks.1.resnets.1.norm2.weight,first_stage_model.encoder.down.1.block.1.norm2.weight
    encoder.down_blocks.1.resnets.1.norm2.bias,first_stage_model.encoder.down.1.block.1.norm2.bias
    encoder.down_blocks.1.resnets.1.conv2.weight,first_stage_model.encoder.down.1.block.1.conv2.weight
    encoder.down_blocks.1.resnets.1.conv2.bias,first_stage_model.encoder.down.1.block.1.conv2.bias
    encoder.down_blocks.1.downsamplers.0.conv.weight,first_stage_model.encoder.down.1.downsample.conv.weight
    encoder.down_blocks.1.downsamplers.0.conv.bias,first_stage_model.encoder.down.1.downsample.conv.bias
    encoder.down_blocks.2.resnets.0.norm1.weight,first_stage_model.encoder.down.2.block.0.norm1.weight
    encoder.down_blocks.2.resnets.0.norm1.bias,first_stage_model.encoder.down.2.block.0.norm1.bias
    encoder.down_blocks.2.resnets.0.conv1.weight,first_stage_model.encoder.down.2.block.0.conv1.weight
    encoder.down_blocks.2.resnets.0.conv1.bias,first_stage_model.encoder.down.2.block.0.conv1.bias
    encoder.down_blocks.2.resnets.0.norm2.weight,first_stage_model.encoder.down.2.block.0.norm2.weight
    encoder.down_blocks.2.resnets.0.norm2.bias,first_stage_model.encoder.down.2.block.0.norm2.bias
    encoder.down_blocks.2.resnets.0.conv2.weight,first_stage_model.encoder.down.2.block.0.conv2.weight
    encoder.down_blocks.2.resnets.0.conv2.bias,first_stage_model.encoder.down.2.block.0.conv2.bias
    encoder.down_blocks.2.resnets.0.conv_shortcut.weight,first_stage_model.encoder.down.2.block.0.nin_shortcut.weight
    encoder.down_blocks.2.resnets.0.conv_shortcut.bias,first_stage_model.encoder.down.2.block.0.nin_shortcut.bias
    encoder.down_blocks.2.resnets.1.norm1.weight,first_stage_model.encoder.down.2.block.1.norm1.weight
    encoder.down_blocks.2.resnets.1.norm1.bias,first_stage_model.encoder.down.2.block.1.norm1.bias
    encoder.down_blocks.2.resnets.1.conv1.weight,first_stage_model.encoder.down.2.block.1.conv1.weight
    encoder.down_blocks.2.resnets.1.conv1.bias,first_stage_model.encoder.down.2.block.1.conv1.bias
    encoder.down_blocks.2.resnets.1.norm2.weight,first_stage_model.encoder.down.2.block.1.norm2.weight
    encoder.down_blocks.2.resnets.1.norm2.bias,first_stage_model.encoder.down.2.block.1.norm2.bias
    encoder.down_blocks.2.resnets.1.conv2.weight,first_stage_model.encoder.down.2.block.1.conv2.weight
    encoder.down_blocks.2.resnets.1.conv2.bias,first_stage_model.encoder.down.2.block.1.conv2.bias
    encoder.mid_block.attentions.0.group_norm.weight,first_stage_model.encoder.mid.attn_1.norm.weight
    encoder.mid_block.attentions.0.group_norm.bias,first_stage_model.encoder.mid.attn_1.norm.bias
    encoder.mid_block.attentions.0.to_q.bias,first_stage_model.encoder.mid.attn_1.q.bias
    encoder.mid_block.attentions.0.to_k.bias,first_stage_model.encoder.mid.attn_1.k.bias
    encoder.mid_block.attentions.0.to_v.bias,first_stage_model.encoder.mid.attn_1.v.bias
    encoder.mid_block.attentions.0.to_out.0.bias,first_stage_model.encoder.mid.attn_1.proj_out.bias
    encoder.mid_block.resnets.0.norm1.weight,first_stage_model.encoder.mid.block_1.norm1.weight
    encoder.mid_block.resnets.0.norm1.bias,first_stage_model.encoder.mid.block_1.norm1.bias
    encoder.mid_block.resnets.0.conv1.weight,first_stage_model.encoder.mid.block_1.conv1.weight
    encoder.mid_block.resnets.0.conv1.bias,first_stage_model.encoder.mid.block_1.conv1.bias
    encoder.mid_block.resnets.0.norm2.weight,first_stage_model.encoder.mid.block_1.norm2.weight
    encoder.mid_block.resnets.0.norm2.bias,first_stage_model.encoder.mid.block_1.norm2.bias
    encoder.mid_block.resnets.0.conv2.weight,first_stage_model.encoder.mid.block_1.conv2.weight
    encoder.mid_block.resnets.0.conv2.bias,first_stage_model.encoder.mid.block_1.conv2.bias
    encoder.mid_block.resnets.1.norm1.weight,first_stage_model.encoder.mid.block_2.norm1.weight
    encoder.mid_block.resnets.1.norm1.bias,first_stage_model.encoder.mid.block_2.norm1.bias
    encoder.mid_block.resnets.1.conv1.weight,first_stage_model.encoder.mid.block_2.conv1.weight
    encoder.mid_block.resnets.1.conv1.bias,first_stage_model.encoder.mid.block_2.conv1.bias
    encoder.mid_block.resnets.1.norm2.weight,first_stage_model.encoder.mid.block_2.norm2.weight
    encoder.mid_block.resnets.1.norm2.bias,first_stage_model.encoder.mid.block_2.norm2.bias
    encoder.mid_block.resnets.1.conv2.weight,first_stage_model.encoder.mid.block_2.conv2.weight
    encoder.mid_block.resnets.1.conv2.bias,first_stage_model.encoder.mid.block_2.conv2.bias
    encoder.conv_norm_out.weight,first_stage_model.encoder.norm_out.weight
    encoder.conv_norm_out.bias,first_stage_model.encoder.norm_out.bias
    encoder.conv_out.weight,first_stage_model.encoder.conv_out.weight
    encoder.conv_out.bias,first_stage_model.encoder.conv_out.bias
    quant_conv.weight,first_stage_model.quant_conv.weight
    quant_conv.bias,first_stage_model.quant_conv.bias
    quantize.embedding.weight,first_stage_model.quantize.embedding.weight
    post_quant_conv.weight,first_stage_model.post_quant_conv.weight
    post_quant_conv.bias,first_stage_model.post_quant_conv.bias
    decoder.conv_in.weight,first_stage_model.decoder.conv_in.weight
    decoder.conv_in.bias,first_stage_model.decoder.conv_in.bias
    decoder.up_blocks.0.resnets.0.norm1.weight,first_stage_model.decoder.up.2.block.0.norm1.weight
    decoder.up_blocks.0.resnets.0.norm1.bias,first_stage_model.decoder.up.2.block.0.norm1.bias
    decoder.up_blocks.0.resnets.0.conv1.weight,first_stage_model.decoder.up.2.block.0.conv1.weight
    decoder.up_blocks.0.resnets.0.conv1.bias,first_stage_model.decoder.up.2.block.0.conv1.bias
    decoder.up_blocks.0.resnets.0.norm2.weight,first_stage_model.decoder.up.2.block.0.norm2.weight
    decoder.up_blocks.0.resnets.0.norm2.bias,first_stage_model.decoder.up.2.block.0.norm2.bias
    decoder.up_blocks.0.resnets.0.conv2.weight,first_stage_model.decoder.up.2.block.0.conv2.weight
    decoder.up_blocks.0.resnets.0.conv2.bias,first_stage_model.decoder.up.2.block.0.conv2.bias
    decoder.up_blocks.0.resnets.1.norm1.weight,first_stage_model.decoder.up.2.block.1.norm1.weight
    decoder.up_blocks.0.resnets.1.norm1.bias,first_stage_model.decoder.up.2.block.1.norm1.bias
    decoder.up_blocks.0.resnets.1.conv1.weight,first_stage_model.decoder.up.2.block.1.conv1.weight
    decoder.up_blocks.0.resnets.1.conv1.bias,first_stage_model.decoder.up.2.block.1.conv1.bias
    decoder.up_blocks.0.resnets.1.norm2.weight,first_stage_model.decoder.up.2.block.1.norm2.weight
    decoder.up_blocks.0.resnets.1.norm2.bias,first_stage_model.decoder.up.2.block.1.norm2.bias
    decoder.up_blocks.0.resnets.1.conv2.weight,first_stage_model.decoder.up.2.block.1.conv2.weight
    decoder.up_blocks.0.resnets.1.conv2.bias,first_stage_model.decoder.up.2.block.1.conv2.bias
    decoder.up_blocks.0.resnets.2.norm1.weight,first_stage_model.decoder.up.2.block.2.norm1.weight
    decoder.up_blocks.0.resnets.2.norm1.bias,first_stage_model.decoder.up.2.block.2.norm1.bias
    decoder.up_blocks.0.resnets.2.conv1.weight,first_stage_model.decoder.up.2.block.2.conv1.weight
    decoder.up_blocks.0.resnets.2.conv1.bias,first_stage_model.decoder.up.2.block.2.conv1.bias
    decoder.up_blocks.0.resnets.2.norm2.weight,first_stage_model.decoder.up.2.block.2.norm2.weight
    decoder.up_blocks.0.resnets.2.norm2.bias,first_stage_model.decoder.up.2.block.2.norm2.bias
    decoder.up_blocks.0.resnets.2.conv2.weight,first_stage_model.decoder.up.2.block.2.conv2.weight
    decoder.up_blocks.0.resnets.2.conv2.bias,first_stage_model.decoder.up.2.block.2.conv2.bias
    decoder.up_blocks.0.upsamplers.0.conv.weight,first_stage_model.decoder.up.2.upsample.conv.weight
    decoder.up_blocks.0.upsamplers.0.conv.bias,first_stage_model.decoder.up.2.upsample.conv.bias
    decoder.up_blocks.1.resnets.0.norm1.weight,first_stage_model.decoder.up.1.block.0.norm1.weight
    decoder.up_blocks.1.resnets.0.norm1.bias,first_stage_model.decoder.up.1.block.0.norm1.bias
    decoder.up_blocks.1.resnets.0.conv1.weight,first_stage_model.decoder.up.1.block.0.conv1.weight
    decoder.up_blocks.1.resnets.0.conv1.bias,first_stage_model.decoder.up.1.block.0.conv1.bias
    decoder.up_blocks.1.resnets.0.norm2.weight,first_stage_model.decoder.up.1.block.0.norm2.weight
    decoder.up_blocks.1.resnets.0.norm2.bias,first_stage_model.decoder.up.1.block.0.norm2.bias
    decoder.up_blocks.1.resnets.0.conv2.weight,first_stage_model.decoder.up.1.block.0.conv2.weight
    decoder.up_blocks.1.resnets.0.conv2.bias,first_stage_model.decoder.up.1.block.0.conv2.bias
    decoder.up_blocks.1.resnets.0.conv_shortcut.weight,first_stage_model.decoder.up.1.block.0.nin_shortcut.weight
    decoder.up_blocks.1.resnets.0.conv_shortcut.bias,first_stage_model.decoder.up.1.block.0.nin_shortcut.bias
    decoder.up_blocks.1.resnets.1.norm1.weight,first_stage_model.decoder.up.1.block.1.norm1.weight
    decoder.up_blocks.1.resnets.1.norm1.bias,first_stage_model.decoder.up.1.block.1.norm1.bias
    decoder.up_blocks.1.resnets.1.conv1.weight,first_stage_model.decoder.up.1.block.1.conv1.weight
    decoder.up_blocks.1.resnets.1.conv1.bias,first_stage_model.decoder.up.1.block.1.conv1.bias
    decoder.up_blocks.1.resnets.1.norm2.weight,first_stage_model.decoder.up.1.block.1.norm2.weight
    decoder.up_blocks.1.resnets.1.norm2.bias,first_stage_model.decoder.up.1.block.1.norm2.bias
    decoder.up_blocks.1.resnets.1.conv2.weight,first_stage_model.decoder.up.1.block.1.conv2.weight
    decoder.up_blocks.1.resnets.1.conv2.bias,first_stage_model.decoder.up.1.block.1.conv2.bias
    decoder.up_blocks.1.resnets.2.norm1.weight,first_stage_model.decoder.up.1.block.2.norm1.weight
    decoder.up_blocks.1.resnets.2.norm1.bias,first_stage_model.decoder.up.1.block.2.norm1.bias
    decoder.up_blocks.1.resnets.2.conv1.weight,first_stage_model.decoder.up.1.block.2.conv1.weight
    decoder.up_blocks.1.resnets.2.conv1.bias,first_stage_model.decoder.up.1.block.2.conv1.bias
    decoder.up_blocks.1.resnets.2.norm2.weight,first_stage_model.decoder.up.1.block.2.norm2.weight
    decoder.up_blocks.1.resnets.2.norm2.bias,first_stage_model.decoder.up.1.block.2.norm2.bias
    decoder.up_blocks.1.resnets.2.conv2.weight,first_stage_model.decoder.up.1.block.2.conv2.weight
    decoder.up_blocks.1.resnets.2.conv2.bias,first_stage_model.decoder.up.1.block.2.conv2.bias
    decoder.up_blocks.1.upsamplers.0.conv.weight,first_stage_model.decoder.up.1.upsample.conv.weight
    decoder.up_blocks.1.upsamplers.0.conv.bias,first_stage_model.decoder.up.1.upsample.conv.bias
    decoder.up_blocks.2.resnets.0.norm1.weight,first_stage_model.decoder.up.0.block.0.norm1.weight
    decoder.up_blocks.2.resnets.0.norm1.bias,first_stage_model.decoder.up.0.block.0.norm1.bias
    decoder.up_blocks.2.resnets.0.conv1.weight,first_stage_model.decoder.up.0.block.0.conv1.weight
    decoder.up_blocks.2.resnets.0.conv1.bias,first_stage_model.decoder.up.0.block.0.conv1.bias
    decoder.up_blocks.2.resnets.0.norm2.weight,first_stage_model.decoder.up.0.block.0.norm2.weight
    decoder.up_blocks.2.resnets.0.norm2.bias,first_stage_model.decoder.up.0.block.0.norm2.bias
    decoder.up_blocks.2.resnets.0.conv2.weight,first_stage_model.decoder.up.0.block.0.conv2.weight
    decoder.up_blocks.2.resnets.0.conv2.bias,first_stage_model.decoder.up.0.block.0.conv2.bias
    decoder.up_blocks.2.resnets.0.conv_shortcut.weight,first_stage_model.decoder.up.0.block.0.nin_shortcut.weight
    decoder.up_blocks.2.resnets.0.conv_shortcut.bias,first_stage_model.decoder.up.0.block.0.nin_shortcut.bias
    decoder.up_blocks.2.resnets.1.norm1.weight,first_stage_model.decoder.up.0.block.1.norm1.weight
    decoder.up_blocks.2.resnets.1.norm1.bias,first_stage_model.decoder.up.0.block.1.norm1.bias
    decoder.up_blocks.2.resnets.1.conv1.weight,first_stage_model.decoder.up.0.block.1.conv1.weight
    decoder.up_blocks.2.resnets.1.conv1.bias,first_stage_model.decoder.up.0.block.1.conv1.bias
    decoder.up_blocks.2.resnets.1.norm2.weight,first_stage_model.decoder.up.0.block.1.norm2.weight
    decoder.up_blocks.2.resnets.1.norm2.bias,first_stage_model.decoder.up.0.block.1.norm2.bias
    decoder.up_blocks.2.resnets.1.conv2.weight,first_stage_model.decoder.up.0.block.1.conv2.weight
    decoder.up_blocks.2.resnets.1.conv2.bias,first_stage_model.decoder.up.0.block.1.conv2.bias
    decoder.up_blocks.2.resnets.2.norm1.weight,first_stage_model.decoder.up.0.block.2.norm1.weight
    decoder.up_blocks.2.resnets.2.norm1.bias,first_stage_model.decoder.up.0.block.2.norm1.bias
    decoder.up_blocks.2.resnets.2.conv1.weight,first_stage_model.decoder.up.0.block.2.conv1.weight
    decoder.up_blocks.2.resnets.2.conv1.bias,first_stage_model.decoder.up.0.block.2.conv1.bias
    decoder.up_blocks.2.resnets.2.norm2.weight,first_stage_model.decoder.up.0.block.2.norm2.weight
    decoder.up_blocks.2.resnets.2.norm2.bias,first_stage_model.decoder.up.0.block.2.norm2.bias
    decoder.up_blocks.2.resnets.2.conv2.weight,first_stage_model.decoder.up.0.block.2.conv2.weight
    decoder.up_blocks.2.resnets.2.conv2.bias,first_stage_model.decoder.up.0.block.2.conv2.bias
    decoder.mid_block.attentions.0.group_norm.weight,first_stage_model.decoder.mid.attn_1.norm.weight
    decoder.mid_block.attentions.0.group_norm.bias,first_stage_model.decoder.mid.attn_1.norm.bias
    decoder.mid_block.attentions.0.to_q.bias,first_stage_model.decoder.mid.attn_1.q.bias
    decoder.mid_block.attentions.0.to_k.bias,first_stage_model.decoder.mid.attn_1.k.bias
    decoder.mid_block.attentions.0.to_v.bias,first_stage_model.decoder.mid.attn_1.v.bias
    decoder.mid_block.attentions.0.to_out.0.bias,first_stage_model.decoder.mid.attn_1.proj_out.bias
    decoder.mid_block.resnets.0.norm1.weight,first_stage_model.decoder.mid.block_1.norm1.weight
    decoder.mid_block.resnets.0.norm1.bias,first_stage_model.decoder.mid.block_1.norm1.bias
    decoder.mid_block.resnets.0.conv1.weight,first_stage_model.decoder.mid.block_1.conv1.weight
    decoder.mid_block.resnets.0.conv1.bias,first_stage_model.decoder.mid.block_1.conv1.bias
    decoder.mid_block.resnets.0.norm2.weight,first_stage_model.decoder.mid.block_1.norm2.weight
    decoder.mid_block.resnets.0.norm2.bias,first_stage_model.decoder.mid.block_1.norm2.bias
    decoder.mid_block.resnets.0.conv2.weight,first_stage_model.decoder.mid.block_1.conv2.weight
    decoder.mid_block.resnets.0.conv2.bias,first_stage_model.decoder.mid.block_1.conv2.bias
    decoder.mid_block.resnets.1.norm1.weight,first_stage_model.decoder.mid.block_2.norm1.weight
    decoder.mid_block.resnets.1.norm1.bias,first_stage_model.decoder.mid.block_2.norm1.bias
    decoder.mid_block.resnets.1.conv1.weight,first_stage_model.decoder.mid.block_2.conv1.weight
    decoder.mid_block.resnets.1.conv1.bias,first_stage_model.decoder.mid.block_2.conv1.bias
    decoder.mid_block.resnets.1.norm2.weight,first_stage_model.decoder.mid.block_2.norm2.weight
    decoder.mid_block.resnets.1.norm2.bias,first_stage_model.decoder.mid.block_2.norm2.bias
    decoder.mid_block.resnets.1.conv2.weight,first_stage_model.decoder.mid.block_2.conv2.weight
    decoder.mid_block.resnets.1.conv2.bias,first_stage_model.decoder.mid.block_2.conv2.bias
    decoder.conv_norm_out.weight,first_stage_model.decoder.norm_out.weight
    decoder.conv_norm_out.bias,first_stage_model.decoder.norm_out.bias
    decoder.conv_out.weight,first_stage_model.decoder.conv_out.weight
    decoder.conv_out.bias,first_stage_model.decoder.conv_out.bias
    '''

    SD2HF = {x.split(',')[1].strip(): x.split(',')[0].strip() for x in HF2SD.strip().split('\n')}

    SDModified = '''
    first_stage_model.encoder.mid.attn_1.q.weight
    first_stage_model.encoder.mid.attn_1.k.weight
    first_stage_model.encoder.mid.attn_1.v.weight
    first_stage_model.encoder.mid.attn_1.proj_out.weight
    first_stage_model.decoder.mid.attn_1.q.weight
    first_stage_model.decoder.mid.attn_1.k.weight
    first_stage_model.decoder.mid.attn_1.v.weight
    first_stage_model.decoder.mid.attn_1.proj_out.weight
    '''.strip().split()

    HFModified = '''
    encoder.mid_block.attentions.0.to_q.weight
    encoder.mid_block.attentions.0.to_k.weight
    encoder.mid_block.attentions.0.to_v.weight
    encoder.mid_block.attentions.0.to_out.0.weight
    decoder.mid_block.attentions.0.to_q.weight
    decoder.mid_block.attentions.0.to_k.weight
    decoder.mid_block.attentions.0.to_v.weight
    decoder.mid_block.attentions.0.to_out.0.weight    
    '''.strip().split()


    HFindex = 0
    SDindex = 0
    while SDindex < len(SDModified):
        SDkey = SDModified[SDindex:SDindex+1]
        SD2HF[SDkey[0]] = HFModified[HFindex:HFindex+1]
        HFindex += 1
        SDindex += 1

    HFChanged = []
    for key in SD2HF:
        if type(SD2HF[key]) == str:
            # assert HFmodel[SD2HF[key]].equal(SDmodel[key])
            HFmodel[SD2HF[key]] = SDmodel[key]
            HFChanged.append(SD2HF[key])
        
        else:
            if len(SD2HF[key]) == 1:
                # assert HFmodel[SD2HF[key][0]].equal(SDmodel[key].squeeze())
                HFmodel[SD2HF[key][0]] = SDmodel[key].squeeze()
                HFChanged += SD2HF[key]
            else:
                breakpoint()
                raise NotImplementedError
            
    assert sorted(list(HFmodel.keys())) == sorted(HFChanged)
    return HFmodel, HFChanged

def compare_key_name_by_weight(HFmodel, SDmodel):
    SDkeys = SDmodel.keys()
    HFkeys = HFmodel.keys()

    HFChanged = []
    for HFkey in HFkeys:
        exist = False
        for SDkey in SDkeys:
            if HFmodel[HFkey].equal(SDmodel[SDkey]):
                print(HFkey, SDkey, sep=',')
                HFChanged.append(HFkey)
                exist = True
                break
        if not exist:
            # print(HFkey)
            pass
            

    return HFmodel, HFChanged

    

if __name__ == '__main__':
    SDmodel = torch.load(open("/raid/workspace/cvml_user/rhg/ECCV2024/local_shared/pretrained_models/sd_models/celebahq-ldm-vq-4.ckpt", 'rb'))["state_dict"]
    pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256")
    HFmodelunet = pipe.unet.state_dict()
    HFmodelunet, HFChangedunet = change_weight_unet(HFmodelunet, SDmodel)

    HFmodelvqvae = pipe.vqvae.state_dict()
    HFmodelvqvae, HFChangedvqvae = change_weight_vqvae(HFmodelvqvae, SDmodel)
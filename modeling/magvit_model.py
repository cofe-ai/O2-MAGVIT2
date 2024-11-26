import torch.distributed
from modeling.magvit2_2d import Encoder as VisionEncoder2D, Decoder as VisionDecoder2D
from modeling.magvit2 import Encoder as VisionEncoder3D, Decoder as VisionDecoder3D
from modeling.lfq_quantizer import LFQ
from modeling.discriminator import StyleGANDiscriminator3D
from modeling.discriminator_2d import StyleGANDiscriminator2D
from src import losses
from torch import nn
from torchvision.models import resnet50, vgg16
import torch.nn.functional as F
from collections import namedtuple
from einops import rearrange
from contextlib import contextmanager
import torch
import time



GenLossBreakdown = namedtuple('GenLossBreakdown', [
    'recon_loss',
    'quantize_loss',
    'perceptual_loss',
    'g_adversarial_loss',
])

DiscriLossBreakdown = namedtuple('DiscriLossBreakdown', [
    'd_adversarial_loss',
    'lecam_reg_loss',
    'gradient_penalty_loss',
])

class VisionTokenizerModule(nn.Module):
    def __init__(self, config, modal, quantizer_type, codebook_size, commitment_cost, diversity_gamma):
        super().__init__()
        self.modal = modal
        if modal == "video":
            filters = config.model.decoder.filters

            self.encoder = VisionEncoder3D(
                filters=filters,
                chan_in=3,
                chan_out=config.model.quantize_model.token_size,
                num_res_blocks=config.model.encoder.num_res_blocks,
                temporal_downsample=config.model.encoder.temporal_downsample,
                channel_multipliers=config.model.encoder.channel_multipliers,
                skip_conv_first=False,
                skip_conv_last=False
            )
            self.decoder = VisionDecoder3D(
                filters=filters,
                chan_in=config.model.quantize_model.token_size,
                chan_out=3,
                num_res_blocks=config.model.decoder.num_res_blocks,
                temporal_downsample=config.model.encoder.temporal_downsample,
                channel_multipliers=config.model.decoder.channel_multipliers,
                skip_conv_first=False,
            )
        else:
            filters = config.model.decoder.filters

            self.encoder = VisionEncoder2D(
                filters=filters,
                chan_in=3,
                chan_out=config.model.quantize_model.token_size,
                num_res_blocks=config.model.encoder.num_res_blocks,
                channel_multipliers=config.model.encoder.channel_multipliers,
                skip_conv_first=False,
                skip_conv_last=False
            )
            self.decoder = VisionDecoder2D(
                filters=filters,
                chan_in=config.model.quantize_model.token_size,
                chan_out=3,
                num_res_blocks=config.model.decoder.num_res_blocks,
                channel_multipliers=config.model.decoder.channel_multipliers,
                skip_conv_first=False,
            )

        if quantizer_type == "lfq":
            self.quantize = LFQ(
                codebook_size=codebook_size,
                dim=config.model.quantize_model.token_size,
                commitment_loss_weight=commitment_cost,
                diversity_gamma=diversity_gamma,
                spherical=config.model.quantize_model.use_l2_norm,
                force_quantization_f32=True,
                num_codebooks=1
            )
        else:
            raise NotImplementedError("quantizer type not implemented yet.")

    def encode(self, x, entropy_loss_weight=0.1, use_distributed_batch_entropy=None, calculate_quantize_loss=True):
        z = self.encoder(x)
        h, w = z.shape[-2:]
        if self.modal == "image":
            z_flattened = rearrange(z, "b c h w -> b (h w) c")
        elif self.modal == "video":
            z_flattened = rearrange(z, "b c t h w -> b (t h w) c")
        quantized_output, loss_breakdown = self.quantize(z_flattened, entropy_loss_weight=entropy_loss_weight, calculate_loss=calculate_quantize_loss, return_loss_breakdown=True, use_distributed_batch_entropy=use_distributed_batch_entropy)
        if self.modal == "image":
            z_quantized = rearrange(quantized_output.quantized, "b (h w) c -> b c h w", h=h, w=w)
        elif self.modal == "video":
            z_quantized = rearrange(quantized_output.quantized, "b (t h w) c -> b c t h w", h=h, w=w)
        return z_quantized, quantized_output, loss_breakdown
    
    def decode(self, *args):
        decoded_pixel_output = self.decoder(*args)
        return decoded_pixel_output

    def forward(self, x, entropy_loss_weight: float, calculate_quantize_loss=True):
        z_quantized, quantized_output, quantize_loss_breakdown = self.encode(x, entropy_loss_weight, calculate_quantize_loss)
        decoded_pixel_output = self.decode(z_quantized)
        return decoded_pixel_output, z_quantized, quantized_output, quantize_loss_breakdown

class VisionTokenizer(nn.Module):
    def __init__(self, config, commitment_cost, diversity_gamma=0.0, use_gan=False, use_lecam_ema=False, use_perceptual=False, perceptual_ckpt_path=None):
        super().__init__()
        self.config = config
        self.modal = config.modal
        self.use_gan = use_gan
        self.use_lecam_ema = use_lecam_ema
        self.use_perceptual = use_perceptual
        assert self.modal in ("image", "video")

        self.skip_quantize = self.config.model.quantize_model.skip_quantize
        if self.skip_quantize:
            raise NotImplementedError
        self.codebook_size = self.config.model.quantize_model.codebook_size
        self.quantizer_type = (self.config.model.quantize_model.quantizer_type).lower()

        self.tokenizer = VisionTokenizerModule(
            config=self.config,
            modal=self.modal,
            quantizer_type=self.quantizer_type,
            codebook_size=self.codebook_size,
            commitment_cost=commitment_cost,
            diversity_gamma=diversity_gamma
        )
        # self.tokenizer.apply(self._init_weights) # TODO        
        
        if self.use_perceptual:
            self.perceptual_model = self.prepare_perceptual(ckpt_path=perceptual_ckpt_path)


        if self.use_gan:
            if self.modal  == "video":
                discriminator_cls = StyleGANDiscriminator3D
            elif self.modal == "image":
                discriminator_cls = StyleGANDiscriminator2D
            else:
                raise NotImplementedError
            self.discriminator = discriminator_cls(config, use_lecam_ema=use_lecam_ema)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """

        if isinstance(module, nn.Linear):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def prepare_perceptual(self, ckpt_path):
        weights = torch.load(ckpt_path, map_location="cpu")
        # model = resnet50()
        model = vgg16()
        model.load_state_dict(weights)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model    



    def _forward_discriminator(self, input_data: torch.Tensor, recon_output: torch.Tensor, apply_gradient_penalty: bool, discriminator_loss_type: str, loss_weights: dict):
        
        discri_loss = 0.0
        if apply_gradient_penalty:
            input_data.requires_grad_()
        real_logit = self.discriminator(input_data)
        fake_logit = self.discriminator(recon_output)
        real_pred = torch.mean(real_logit)
        fake_pred = torch.mean(fake_logit)
        if self.use_lecam_ema:
            lecam_loss_weight = loss_weights["lecam_loss_weight"]
            lecam_ema = self.discriminator.lecam_ema
            lecam_ema.update(real_logit, fake_logit)
            lecam_loss = losses.lecam_reg(real_pred, fake_pred,
                                            lecam_ema.logits_real_ema,
                                            lecam_ema.logits_fake_ema)
            discri_loss += lecam_loss * lecam_loss_weight
        else:
            lecam_loss = None
        d_adversarial_loss = losses.discriminator_loss(
            real_logit=real_logit,
            fake_logit=fake_logit,
            loss_type=discriminator_loss_type)

        discri_loss += d_adversarial_loss * loss_weights["d_adversarial_loss_weight"]

        if apply_gradient_penalty:
            gradient_penalty_loss = losses.r1_gradient_penalty(input_data, real_logit, loss_weights["gradient_penalty_cost"])
        
            discri_loss += gradient_penalty_loss
        else:
            gradient_penalty_loss = None
        loss_breakdown = DiscriLossBreakdown(lecam_reg_loss=lecam_loss, d_adversarial_loss=d_adversarial_loss, gradient_penalty_loss=gradient_penalty_loss)
        return discri_loss, loss_breakdown

    def _forward_generator(self, input_data, recon_output: torch.Tensor, recon_loss, quantize_loss, generator_loss_type, loss_weights):
        
        gen_loss = recon_loss * loss_weights["recon_loss_weight"]
        if quantize_loss is not None:
            gen_loss += quantize_loss * loss_weights["quantizer_aux_loss_weight"]
        if self.use_perceptual:
            perceptual_loss = self._forward_perceptual(input_data=input_data, recon_output=recon_output)
            gen_loss += perceptual_loss * loss_weights["perceptual_loss_weight"]
        fake_logit = self.discriminator(recon_output)
        g_adversarial_loss = losses.generator_loss(
            fake_logit=fake_logit, loss_type=generator_loss_type
        )
        gen_loss += g_adversarial_loss * loss_weights["g_adversarial_loss_weight"]
        loss_breakdown = GenLossBreakdown(g_adversarial_loss=g_adversarial_loss, recon_loss=recon_loss, quantize_loss=quantize_loss, perceptual_loss=perceptual_loss)
        return gen_loss, loss_breakdown

    def _forward_perceptual(self, input_data: torch.Tensor, recon_output: torch.Tensor):
        
        if input_data.ndim == 5:
            real_perceptual_inputs = rearrange(input_data, "b c t h w -> (b t) c h w")
            fake_perceptual_inputs = rearrange(recon_output, "b c t h w -> (b t) c h w")
        else:
            real_perceptual_inputs = input_data
            fake_perceptual_inputs = recon_output
        perceptual_loss = losses.calculate_perceptual_loss(real_perceptual_inputs, fake_perceptual_inputs, self.perceptual_model)
        return perceptual_loss

    def _forward_reconstruction(self, input_data, calculate_loss=False, loss_weights: dict=None, use_distributed_batch_entropy=False):
        z_quantized, quantized_output, quantize_loss_breakdown = self.tokenizer.encode(input_data, entropy_loss_weight=loss_weights["quantizer_entropy_loss_weight"], use_distributed_batch_entropy=use_distributed_batch_entropy, calculate_quantize_loss=calculate_loss)
        decoded_pixel_output = self.tokenizer.decode(z_quantized)

        if self.skip_quantize:
            quantize_loss = None
            quantized_token_ids = []
        else:
            quantized_token_ids =  quantized_output.indices
            quantize_loss = quantized_output.entropy_aux_loss

        if calculate_loss:
            recon_loss = F.mse_loss(input_data, decoded_pixel_output)
        else:
            recon_loss = None
        return decoded_pixel_output, quantized_token_ids, recon_loss, quantize_loss, quantize_loss_breakdown


    def forward(self, forward_mode: str, **forward_kwargs): 
        """
        Parameters:
        - forward_mode (str): Determines the mode of operation for the forward method.
                            Acceptable values are ["reconstruction", "generator", "discriminator"].
        - *forward_args: Positional arguments for the forward method.
        - **forward_kwargs: Keyword arguments for the forward method.
        """
        if forward_mode == "reconstruction":
            return self._forward_reconstruction(**forward_kwargs)
        elif forward_mode == "generator":
            return self._forward_generator(**forward_kwargs)
        elif forward_mode == "discriminator":
            return self._forward_discriminator(**forward_kwargs)
        else:
            raise NotImplementedError(f"unsuported {forward_mode}")


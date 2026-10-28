"""
PickMoment: FLUX.2-klein-4B with a MeanFlow-style time-pair embedding.

The network predicts the interval-mean image over [s, t] of the exposure:

    f_theta(x_blur, s, t) = mean of the sharp signal over [s, t],  s, t in [-1, 1]

Architecture:
    z_blur    = vae.encode(x_blur)                       # packed, BN-normalized latent
    time_pair = normalize([t, s])                         # [0, 1] endpoint pair
    temb      = (t_embedder(t) + r_embedder(s)) / 2       # replaces FLUX time/guidance embed
    v         = transformer(z_blur, timestep=time_pair)   # null text tokens for cross-attention
    z_pred    = z_blur + v
    output    = vae.decode(z_pred)

The VAE is frozen. The full transformer and the null tokens are trained.
"""

import math
import os

import torch
import torch.nn as nn
from diffusers.models import Flux2Transformer2DModel, AutoencoderKLFlux2


HF_PREFIX = "hf://"


def resolve_checkpoint(path):
    """Return a local checkpoint path.

    Accepts either a local file or `hf://<user>/<repo>/<filename>`, which is
    downloaded from the Hugging Face Hub.
    """
    if path is None:
        return None
    if path.startswith(HF_PREFIX):
        from huggingface_hub import hf_hub_download
        parts = path[len(HF_PREFIX):].split("/")
        if len(parts) < 3:
            raise ValueError(f"Expected hf://<user>/<repo>/<filename>, got {path}")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return hf_hub_download(repo_id=repo_id, filename=filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


# ---------------------------------------------------------------------------
# MeanFlow-style time-pair embedding
# ---------------------------------------------------------------------------

class MeanFlowTimestepEmbedder(nn.Module):
    """Sinusoidal embedding + 2-layer MLP for one scalar time input."""

    def __init__(self, dim: int, nfreq: int = 256, max_period: float = 10000.0,
                 bias: bool = False):
        super().__init__()
        self.nfreq = nfreq
        self.max_period = max_period
        self.linear_1 = nn.Linear(nfreq, dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim, bias=bias)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.nfreq // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.nfreq % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t).to(dtype=self.linear_1.weight.dtype)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class MeanFlowTimePairEmbeddings(nn.Module):
    """Pair embedder: (t_embedder(t) + r_embedder(r)) / 2."""

    def __init__(self, embedding_dim: int, nfreq: int = 256, bias: bool = False):
        super().__init__()
        self.t_embedder = MeanFlowTimestepEmbedder(embedding_dim, nfreq=nfreq, bias=bias)
        self.r_embedder = MeanFlowTimestepEmbedder(embedding_dim, nfreq=nfreq, bias=bias)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor | None) -> torch.Tensor:
        del guidance
        if timestep.ndim != 2 or timestep.shape[-1] != 2:
            raise ValueError(
                "MeanFlowTimePairEmbeddings expects `timestep` with shape (B, 2) "
                "containing the pair [t, r]."
            )
        t, r = timestep.unbind(dim=-1)
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(r)
        return (t_emb + r_emb) / 2


def configure_flux_meanflow_timestep_embedding(transformer):
    """Replace FLUX's time/guidance embedder with the time-pair embedder.

    Both embedders are initialized from the pretrained FLUX.2 `timestep_embedder`
    so the original time-conditioning prior is preserved.
    """
    transformer_ref = transformer.module if hasattr(transformer, 'module') else transformer
    old_embed = transformer_ref.time_guidance_embed
    nfreq = getattr(
        old_embed.time_proj,
        'num_channels',
        getattr(transformer_ref.config, 'timestep_guidance_channels', 256),
    )
    embedding_dim = getattr(transformer_ref, 'inner_dim', None)
    if embedding_dim is None:
        embedding_dim = (
            transformer_ref.config.num_attention_heads
            * transformer_ref.config.attention_head_dim
        )

    use_bias = bool(getattr(old_embed.timestep_embedder.linear_1, 'bias', None) is not None)
    new_embed = MeanFlowTimePairEmbeddings(
        embedding_dim=embedding_dim,
        nfreq=nfreq,
        bias=use_bias,
    )

    if hasattr(old_embed, 'timestep_embedder'):
        def _copy_linear(dst, src):
            dst.weight.data.copy_(src.weight.data)
            if dst.bias is not None and src.bias is not None:
                dst.bias.data.copy_(src.bias.data)

        _copy_linear(new_embed.t_embedder.linear_1, old_embed.timestep_embedder.linear_1)
        _copy_linear(new_embed.t_embedder.linear_2, old_embed.timestep_embedder.linear_2)
        _copy_linear(new_embed.r_embedder.linear_1, old_embed.timestep_embedder.linear_1)
        _copy_linear(new_embed.r_embedder.linear_2, old_embed.timestep_embedder.linear_2)

    embed_ref = old_embed.timestep_embedder.linear_1.weight
    new_embed.to(device=embed_ref.device, dtype=embed_ref.dtype)

    transformer_ref.time_guidance_embed = new_embed

    for p in transformer_ref.time_guidance_embed.parameters():
        p.requires_grad = True


def remap_meanflow_time_embed_state_dict(state_dict):
    """Map FLUX time/guidance embedder parameter names to the pair-embedder names.

    This is a no-op for checkpoints that already use the pair-embedder names.
    """
    key_map = {
        'time_guidance_embed.timestep_embedder.linear_1.weight': 'time_guidance_embed.t_embedder.linear_1.weight',
        'time_guidance_embed.timestep_embedder.linear_1.bias': 'time_guidance_embed.t_embedder.linear_1.bias',
        'time_guidance_embed.timestep_embedder.linear_2.weight': 'time_guidance_embed.t_embedder.linear_2.weight',
        'time_guidance_embed.timestep_embedder.linear_2.bias': 'time_guidance_embed.t_embedder.linear_2.bias',
        'time_guidance_embed.guidance_embedder.linear_1.weight': 'time_guidance_embed.r_embedder.linear_1.weight',
        'time_guidance_embed.guidance_embedder.linear_1.bias': 'time_guidance_embed.r_embedder.linear_1.bias',
        'time_guidance_embed.guidance_embedder.linear_2.weight': 'time_guidance_embed.r_embedder.linear_2.weight',
        'time_guidance_embed.guidance_embedder.linear_2.bias': 'time_guidance_embed.r_embedder.linear_2.bias',
    }
    return {key_map.get(k, k): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Null conditioning tokens for cross-attention
# ---------------------------------------------------------------------------

class NullConditioner(nn.Module):
    """Learnable null tokens used in place of text embeddings.

    Interval conditioning goes only through the time-pair embedding.
    """

    def __init__(self, num_tokens: int = 4, joint_attention_dim: int = 7680):
        super().__init__()
        self.null_tokens = nn.Parameter(torch.zeros(1, num_tokens, joint_attention_dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.null_tokens.expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# FLUX transformer
# ---------------------------------------------------------------------------

def initialize_flux_transformer(args):
    """Load the FLUX.2-klein transformer for full fine-tuning."""
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False)
    transformer.train()
    transformer.requires_grad_(True)

    if args.enable_grad_checkpointing:
        transformer.enable_gradient_checkpointing()
    return transformer


# ---------------------------------------------------------------------------
# PickMoment model
# ---------------------------------------------------------------------------

class PickMoment(nn.Module):
    """
    Residual prediction in the FLUX.2 VAE latent space:
        z_blur = vae.encode(x_blur)
        v      = transformer(z_blur, (s, t))
        z_pred = z_blur + v
        pred   = vae.decode(z_pred)      # f_theta(x_blur, s, t)

    Training losses (see train.py):
        recon:   f_theta(x_blur, s, t) vs the mean of GT sub-frames in [s, t]
        sharp:   f_theta(x_blur, t - eps, t + eps) vs the sharp frame at t
        consist: additivity (t-s) f(s,t) = (m-s) f(s,m) + (t-m) f(m,t)
    """

    def __init__(self, args, mode='train', checkpoint_path=None):
        super().__init__()

        self.args = args
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ----- VAE (frozen) -----
        print(f"Loading FLUX.2 VAE from {args.pretrained_model_name_or_path}...")
        self.vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ----- FLUX transformer (full fine-tuning) -----
        print("Loading FLUX.2 transformer...")
        self.transformer = initialize_flux_transformer(args)
        configure_flux_meanflow_timestep_embedding(self.transformer)

        # ----- Cross-attention null tokens -----
        transformer_ref = (
            self.transformer.module
            if hasattr(self.transformer, 'module')
            else self.transformer
        )
        joint_dim = getattr(transformer_ref.config, 'joint_attention_dim', 7680)
        self.null_conditioner = NullConditioner(
            num_tokens=args.num_cond_tokens,
            joint_attention_dim=joint_dim,
        )

        # VAE properties
        self.latent_channels = self.vae.config.latent_channels
        self.downsampling_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.vae.to(self.device)
        self.transformer.to(self.device)
        self.null_conditioner.to(self.device)

        if checkpoint_path is not None:
            self.load_ckpt(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    def set_train(self):
        self.transformer.train()
        self.null_conditioner.train()
        self.transformer.requires_grad_(True)
        self.null_conditioner.requires_grad_(True)

    def set_eval(self):
        self.transformer.eval()
        self.null_conditioner.eval()

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, images, mode='sample'):
        images = images.to(dtype=torch.bfloat16, device=self.device)
        latent_dist = self.vae.encode(images).latent_dist

        if mode == 'sample':
            latents = latent_dist.sample()
        else:
            latents = latent_dist.mean

        latents = self._pack_latents(latents)

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std

        return latents

    def decode(self, latents):
        latents = latents.to(dtype=torch.bfloat16, device=self.device)

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean

        latents = self._unpack_latents(latents)
        images = self.vae.decode(latents).sample
        return images.clamp(-1, 1)

    # ------------------------------------------------------------------
    # Packing / unpacking (2x2 patchify)
    # ------------------------------------------------------------------

    def _pack_latents(self, latents):
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        return latents

    def _unpack_latents(self, latents):
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
        return latents

    # ------------------------------------------------------------------
    # Positional IDs for FLUX
    # ------------------------------------------------------------------

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        t = torch.arange(1, device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        l = torch.arange(1, device=device)
        latent_image_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
        latent_image_ids = latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_image_ids

    def _prepare_text_ids(self, encoder_hidden_states, device, dtype):
        batch_size, seq_len, _ = encoder_hidden_states.shape
        t = torch.arange(1, device=device)
        h = torch.arange(1, device=device)
        w = torch.arange(1, device=device)
        l = torch.arange(seq_len, device=device)
        txt_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
        txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return txt_ids

    # ------------------------------------------------------------------
    # Core: predict the latent residual conditioned on (s, t)
    # ------------------------------------------------------------------

    def predict_velocity(self, z_blur, s, t):
        batch_size = z_blur.shape[0]
        H, W = z_blur.shape[2], z_blur.shape[3]

        s = s.to(device=z_blur.device, dtype=torch.float32)
        t = t.to(device=z_blur.device, dtype=torch.float32)
        s_norm = ((s + 1.0) / 2.0).clamp_(0.0, 1.0)
        t_norm = ((t + 1.0) / 2.0).clamp_(0.0, 1.0)
        timestep = torch.stack([t_norm, s_norm], dim=-1)

        encoder_hidden_states = self.null_conditioner(batch_size)
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16, device=z_blur.device)

        z_seq = z_blur.permute(0, 2, 3, 1).reshape(batch_size, H * W, z_blur.shape[1])

        img_ids = self._prepare_latent_image_ids(batch_size, H, W, z_seq.device, z_seq.dtype)
        txt_ids = self._prepare_text_ids(encoder_hidden_states, z_seq.device, z_seq.dtype)

        output = self.transformer(
            hidden_states=z_seq,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )

        result = output[0] if isinstance(output, tuple) else output
        C = result.shape[-1]
        velocity = result.reshape(batch_size, H, W, C).permute(0, 3, 1, 2)
        return velocity

    def predict_mean_image(self, z_blur, s, t):
        """Latent of the interval-mean image over [s, t]."""
        v = self.predict_velocity(z_blur, s, t)
        return z_blur + v

    # ------------------------------------------------------------------
    # Inference helpers (inputs/outputs in [-1, 1], H and W multiples of 32)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def deblur(self, x_blur, eps=0.01):
        """Center sharp frame: query [-eps, +eps]."""
        B = x_blur.shape[0]
        s = torch.full((B,), -eps, device=x_blur.device)
        t = torch.full((B,), +eps, device=x_blur.device)
        z_blur = self.encode(x_blur, mode='mean')
        z_pred = self.predict_mean_image(z_blur, s, t)
        return self.decode(z_pred)

    @torch.no_grad()
    def get_frame_at(self, x_blur, t_target, eps=0.01):
        """Sharp frame at an arbitrary moment t_target in [-1, 1]."""
        B = x_blur.shape[0]
        s = torch.full((B,), t_target - eps, device=x_blur.device)
        t = torch.full((B,), t_target + eps, device=x_blur.device)
        z_blur = self.encode(x_blur, mode='mean')
        z_pred = self.predict_mean_image(z_blur, s, t)
        return self.decode(z_pred)

    @torch.no_grad()
    def get_interval_mean(self, x_blur, s_val, t_val):
        """Interval-mean (re-blurred) image over [s_val, t_val]."""
        B = x_blur.shape[0]
        s = torch.full((B,), float(s_val), device=x_blur.device)
        t = torch.full((B,), float(t_val), device=x_blur.device)
        z_blur = self.encode(x_blur, mode='mean')
        z_pred = self.predict_mean_image(z_blur, s, t)
        return self.decode(z_pred)

    @torch.no_grad()
    def extract_video(self, x_blur, n_frames=11, eps=0.01, max_batch=None):
        """N sharp frames at uniformly spaced moments in [-1, 1]. Returns (B, N, 3, H, W)."""
        B = x_blur.shape[0]
        z_blur = self.encode(x_blur, mode='mean')

        t_targets = torch.linspace(-1.0, 1.0, n_frames, device=x_blur.device)
        s_all = (t_targets - eps).repeat(B)
        t_all = (t_targets + eps).repeat(B)
        z_blur_exp = z_blur.repeat_interleave(n_frames, dim=0)

        if max_batch is None or max_batch >= B * n_frames:
            z_pred = self.predict_mean_image(z_blur_exp, s_all, t_all)
            frames = self.decode(z_pred)
        else:
            chunks = []
            for start in range(0, B * n_frames, max_batch):
                end = min(start + max_batch, B * n_frames)
                z_pred_chunk = self.predict_mean_image(
                    z_blur_exp[start:end], s_all[start:end], t_all[start:end]
                )
                chunks.append(self.decode(z_pred_chunk))
            frames = torch.cat(chunks, dim=0)

        frames = frames.view(B, n_frames, *frames.shape[1:])
        return frames

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_model(self, output_path):
        state_dict = {
            'use_lora': False,
            'null_conditioner': self.null_conditioner.state_dict(),
            'state_dict_transformer': self.transformer.state_dict(),
        }
        torch.save(state_dict, output_path)
        print(f"Saved checkpoint to {output_path}")

    def load_ckpt(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if ckpt.get('use_lora', True):
            raise ValueError("Expected a full fine-tuning checkpoint (use_lora=False).")

        state_dict_transformer = remap_meanflow_time_embed_state_dict(
            ckpt['state_dict_transformer']
        )
        self.null_conditioner.load_state_dict(ckpt['null_conditioner'])
        self.transformer.load_state_dict(state_dict_transformer, strict=True)
        print(f"Loaded {len(state_dict_transformer)} transformer tensors and null tokens")

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def get_trainable_params(self):
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        params += [p for p in self.null_conditioner.parameters() if p.requires_grad]
        return params

    def get_trainable_param_count(self):
        return sum(p.numel() for p in self.get_trainable_params())


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

class PickMomentArgs:
    def __init__(
        self,
        pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-4B",
        num_cond_tokens=4,
        enable_grad_checkpointing=True,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_cond_tokens = num_cond_tokens
        self.enable_grad_checkpointing = enable_grad_checkpointing


def build_model(model_cfg, checkpoint=None, mode='test'):
    """Build PickMoment from the `model` section of a config."""
    args = PickMomentArgs(
        pretrained_model_name_or_path=model_cfg.get('backbone', 'black-forest-labs/FLUX.2-klein-4B'),
        num_cond_tokens=model_cfg.get('num_cond_tokens', 4),
        enable_grad_checkpointing=model_cfg.get('enable_grad_checkpointing', True),
    )
    ckpt_path = resolve_checkpoint(checkpoint) if checkpoint else None
    return PickMoment(args, mode=mode, checkpoint_path=ckpt_path)

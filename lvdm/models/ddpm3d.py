"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""

from functools import partial
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import logging
mainlogger = logging.getLogger('mainlogger')
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import pytorch_lightning as pl
from utils.utils import instantiate_from_config
from lvdm.ema import LitEma
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import (
    make_beta_schedule,
    rescale_zero_terminal_snr, 
    normal_kl,
    mean_flat,
    discretized_gaussian_log_likelihood,
    )
from lvdm.basics import disabled_train
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)
import random

from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel

import pdb

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor=None,
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 rescale_betas_zero_snr=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size  # try conv?
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings

        # TODO：
        self.enable_lora = True
        self.model = DiffusionWrapper(unet_config, conditioning_key, self.enable_lora)
        #count_params(self.model, verbose=True)
        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # hybrid, MSE/l2, KL
        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))

        if self.parameterization != 'v':
            self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        else:
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        # replace the first index with the second index
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        mainlogger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        '''
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        '''
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # TODO!: ORIGINAL p_mean_variance
    def p_mean_variance_1(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_input(self, batch, k):
        x = batch[k]
        # test
        # print(f'get_input x: {x}, batch: {batch}')
        # batch的值是{video: tenxxx, caption: ['xxx']} 当取到caption时会报错
        # AttributeError: 'list' object has no attribute 'to'
        # x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 uncond_prob=0.2,
                 uncond_type="empty_seq",
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 only_model=False,
                 noise_strength=0,
                 use_dynamic_rescale=False,
                 base_scale=0.7,
                 turning_step=400,
                 loop_video=False,
                 fps_condition_type='fs',
                 perframe_ae=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.noise_strength = noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.loop_video = loop_video
        self.fps_condition_type = fps_condition_type
        self.perframe_ae = perframe_ae
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config        
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert(encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
                

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            # test
            print(f'get_learned_conditioning cond_stage_forward is None')
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
   
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        
        ## consume more GPU memory but faster
        if not self.perframe_ae:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior).detach()
        else:  ## consume less GPU memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_batch = self.first_stage_model.encode(x[index:index+1,:,:,:])
                frame_result = self.get_first_stage_encoding(frame_batch).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        
        return results
    
    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
            
        if not self.perframe_ae:    
            z = 1. / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1. / self.scale_factor * z[index:index+1,:,:,:]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)
    
    def forward(self, x, c, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        # pdb.set_trace()
        # test
        # print(f'apply_model cond: {cond}')
        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W
        
        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid


    def p_mean_variance(self, x, c, t, 
                        clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, 
                        **kwargs):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        B, F, C = x.shape[:3]
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        # !!:score_corrector means?
        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            pred_xstart = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            pred_xstart = model_out
        else:
            raise NotImplementedError()

        # ?: Latte with denoised_fn?
        if clip_denoised:
            pred_xstart.clamp_(-1., 1.)

        # predict mean, variance, log variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        # TODO: iddpm, learned posterior variance, should add model_var_type in params
        if self.model_var_type == 'learned':
            model_output, model_var_values = torch.split(model_output, C, dim=2)
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            # !: iddpm的后验方差
            posterior_log_variance = frac * max_log + (1 - frac) * min_log
            posterior_variance = torch.exp(posterior_log_variance)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, pred_xstart
        else:
            return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, \
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]        
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img


    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                    At the first timestep return the decoder NLL，otherwise return KL
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        '''
            If trainning with way of iddpm:
                无论用在哪里 loss应该都是一样的
                loss_simple: MSE loss between target noise and model output
                loss_vlb: loss KL between q_mean, q_var, p_mean, p_var
                in this implement, we pass the option of future frame prediction with frame_cond param
                
        '''
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        loss_dict = {}

        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = noise
        else:
            raise NotImplementedError()

        # if self.frame_cond:
        #     ## [b,c,t,h,w]: only care about the predicted part (avoid disturbance)
        #     model_output = model_output[:, :, self.frame_cond:, :, :]
        #     target = target[:, :, self.frame_cond:, :, :]
        # pdb.set_trace()

        # choose loss type: hybrid / simple loss
        # hybrid = MSE + KL * 1/1000
        if self.loss_type == 'hybrid':
            model_output = self.apply_model(x_t, t, cond, **kwargs)

            B, F, C = x_t.shape[:3]
            assert model_output.shape == (B, F, C * 2, *x_t.shape[3:])
            # !!!
            # TODO: 确认是否生成的是 [model_output, model_var_values]，可能和latte的输出不同，会是其他维度*2，而不是channel维度*2
            # 看起来不是，需要额外finetune
            model_output, model_var_values = torch.split(model_output, C, dim=2)
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=2)
            loss_dict["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
            )["output"]

            # using a factor of 1/1000, balance the KL loss and MSE loss
            loss_dict["vb"] *= self.num_timesteps / 1000.0

            loss_dict["mse"] = mean_flat((target - model_output) ** 2)

            loss = loss_dict["mse"] + loss_dict["vb"]
        elif self.loss_type == 'MSE' or 'l2':
            # loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
            model_output = self.apply_model(x_t, t, cond, **kwargs)
            B, C, F = x_t.shape[:3] 
            # refer to iddpm, but dynamicrafter model trained by ddpm
            if model_output.shape == (B, C * 2, F, *x_t.shape[3:]):
                model_output, model_var_values = torch.split(model_output, C, dim=2)
                loss_dict["mse"] = mean_flat((target - model_output) ** 2)
                loss = loss_dict["mse"]
            # iddpm
            else:
                loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
                if torch.isnan(loss_simple).any():
                    print(f"loss_simple exists nan: {loss_simple}")
                    # import pdb; pdb.set_trace()
                    for i in range(loss_simple.shape[0]):
                        if torch.isnan(loss_simple[i]).any():
                            loss_simple[i] = torch.zeros_like(loss_simple[i])
                loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

                if self.logvar.device is not self.device:
                    self.logvar = self.logvar.to(self.device)
                logvar_t = self.logvar[t]
                # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
                loss = loss_simple / torch.exp(logvar_t) + logvar_t
                # loss = loss_simple / torch.exp(self.logvar) + self.logvar
                if self.learn_logvar:
                    loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
                    loss_dict.update({'logvar': self.logvar.data.mean()})

                loss = self.l_simple_weight * loss.mean()

                if self.original_elbo_weight > 0:
                    loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
                    loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
                    loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
                    loss += (self.original_elbo_weight * loss_vlb)
                loss_dict.update({f'{prefix}/loss': loss})

        elif self.loss_type == 'KL':
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        else:
            raise NotImplementedError(self.loss_type)
        
        return loss, loss_dict



class LatentVisualDiffusion(LatentDiffusion):
    def __init__(self, img_cond_stage_config, image_proj_stage_config, freeze_embedder=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self.image_proj_model = instantiate_from_config(image_proj_stage_config)

    def _init_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        ## sync_dist | rank_zero_only
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss

    def shared_step(self, batch, random_uncond, **kwargs):
        is_imgbatch = False
        if "loader_img" in batch.keys():
            ratio = 10.0 / self.temporal_length
            if random.uniform(0., 10.) < ratio:
                is_imgbatch = True
                batch = batch["loader_img"]
            else:
                batch = batch["loader_video"]
        else:
            pass

        # x, c = self.get_batch_input(batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch)
        x, c = self.get_input(batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch)
        loss, loss_dict = self(x, c, is_imgbatch=is_imgbatch, **kwargs)
        return loss, loss_dict

    @torch.no_grad()
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False,
                        is_imgbatch=False):
        ## image/video shape: b, c, t, h, w
        # TODO: get from img
        data_key = 'jpg' if is_imgbatch else self.first_stage_key
        x = super().get_input(batch, data_key)
        if is_imgbatch:
            ## pack image as video
            # x = x[:,:,None,:,:]
            b = x.shape[0] // self.temporal_length
            x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=self.temporal_length)
        x_ori = x
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)

        ## get caption condition
        cond_key = 'txt' if is_imgbatch else self.cond_stage_key
        cond = batch[cond_key]
        # test
        print(f'get_batch_input cond: {cond}， random_uncond: {random_uncond}, self.uncond_type: {self.uncond_type}, self.uncond_prob: {self.uncond_prob}')
        # if random_uncond and self.uncond_type == 'empty_seq':
        #     for i, ci in enumerate(cond):
        #         if random.random() < self.uncond_prob:
        #             cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        # test
        # print(f'get_batch_input cond_emb.shape: {cond_emb.shape}')
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(ci)

        out = [z, cond_emb]
        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x_ori, xrec])
        if return_original_cond:
            out.append(cond)

        return out

    def get_input(self, batch, random_uncond, is_imgbatch, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        # dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))
        # first：edited， cond： edit
        # batch['video', 'caption', 'random_frame', 'jpg']
        # todo: 添加读入jpg的功能
        data_key = 'jpg' if is_imgbatch else self.first_stage_key
        x = super().get_input(batch, data_key)
        # jpg/image, we load nums of frames images
        if is_imgbatch:
            b_ = x.shape[0] // self.temporal_length
            x = rearrange(x, '(b t) c h w -> b c t h w', b=b_, t=self.temporal_length)
        x_ori = x
        x_encode = self.encode_first_stage(x)

        # concat input, 如果是image，concat的东西都弄成当前图像的copy？
        if not is_imgbatch:
            random_frame = batch['random_frame']
            random_frame_copy = random_frame.clone()
            # test
            print(f'get_input begin random_frame.shape: {random_frame.shape}')
        
        if bs is not None:
            x = x[:bs]

        x = x.to(self.device)
        
        # # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        # random = torch.rand(x.size(0), device=x.device)
        # prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        # input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        #
        # null_prompt = self.get_learned_conditioning([""])
        # cond["c_crossattn"] = [
        #     torch.where(prompt_mask, null_prompt, self.get_learned_conditioning('prompt_xxx').detach())]
        # cond["c_concat"] = [input_mask * self.encode_first_stage((xc["c_concat"].to(self.device))).mode().detach()]

        # TODO: 
        # nashi - 4/11
        # image concat
        batch_size, len_frames = x.shape[0], x.shape[2]
        # TODO: 后续还需要考虑batch的问题，目前是get_input没有batch（实际上可能是几个clip的random frames分别凑在一起，然后做concat
        if not is_imgbatch:
            repeat_random_frame = random_frame_copy.permute(0, 3, 1, 2).unsqueeze(2)
            repeat_random_frame = repeat(repeat_random_frame, 'b c t h w -> b c (repeat t) h w', repeat=len_frames)
            repeat_random_frame = (repeat_random_frame / 255. - 0.5) * 2
        else:
            # is img batch, repeat imgbatch
            repeat_random_frame = x.clone()

        img_concat = self.encode_first_stage(repeat_random_frame)
        # OUTPUT: torch.Size([1, 4, 16, 32, 32])

        # todo: add caption
        cond_key = cond_key or self.cond_stage_key
        cond_caption = super().get_input(batch, cond_key)

        # empty seq
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond_caption):
                if random.random() < self.uncond_prob:
                    cond_caption[i] = ""
        
        cond = {}
        # preprocess cap_emb
        # 如果是视频的话，cap_emb是不是直接repeat frames?
        # pdb.set_trace()
        if isinstance(cond_caption, dict) or isinstance(cond_caption, list):
            cap_emb = self.get_learned_conditioning(cond_caption) # b 77 1024
        else:
            cap_emb = self.get_learned_conditioning(cond_caption.to(self.device))
        # zero embedding
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond_caption):
                if random.random() < self.uncond_prob:
                    cap_emb[i] = torch.zeros_like(ci)

        # img cond, contain 

        # pdb.set_trace()

        if not is_imgbatch: 
            img_tensor = random_frame_copy.permute(0, 3, 1, 2).float().to(self.device) # torch.Size([2, 3, 256, 256])
            img_tensor = (img_tensor / 255. - 0.5) * 2
            cond_images = self.embedder(img_tensor)  ## blc torch.Size([b, 257, 1280])
            img_emb = self.image_proj_model(cond_images) # torch.Size([2, 16, 1024])
        else:
            # 针对要输入 img的
            img_emb_list = []
            for i in range(self.temporal_length):
                # b c t h w
                img_tensor = repeat_random_frame[:, :, i, :, :]
                cond_images = self.embedder(img_tensor)  ## blc torch.Size([b, 257, 1280])
                img_emb = self.image_proj_model(cond_images) # torch.Size([2, 16, 1024])
                img_emb_list.append(img_emb)
            img_emb = torch.cat(img_emb_list, dim=1)
            

        # combine cap_emb and img_emb will using for cross-attention
        imtext_cond = torch.cat([cap_emb, img_emb], dim=1)
        

        cond = {"c_crossattn": [imtext_cond], "c_concat": [img_concat]}
        out = [x_encode, cond]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(x_encode)
            out.extend([x, xrec])
        if return_original_cond:
            # onlu cond_caption without img_caption
            out.append(cond_caption)
        # test
        # print(f'get_input out: {out}')
        return out

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        # if self.empty_params_only and hasattr(self, "empty_paras"):
        #     params = [p for n, p in self.model.named_parameters() if n in self.empty_paras]
        #     print('self.empty_paras', len(self.empty_paras))
        #     for n, p in self.model.named_parameters():
        #         if n not in self.empty_paras:
        #             p.requires_grad = False
        #     mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
        # else:
        #     params = list(self.model.parameters())
        #     mainlogger.info(f"@Training [{len(params)}] Full Paramters.")
        params = list(self.model.parameters())
        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)
        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key, enable_lora=False):
        super().__init__()
        self.diffusion_model = self.apply_lora(instantiate_from_config(diff_model_config), enable_lora)
        self.conditioning_key = conditioning_key

    
    def apply_lora(self, model, enable_lora=False):
        if not enable_lora:
            return model
        
        for name, param in model.named_parameters():
            param.requires_grad_(False)

        child_list = []
        for name, child in model.named_children():
            child_list.append(name)

        # diffusion model
        dm_child_list = []
        dm_child_list_ = []

        target_modules_list = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ]
        # pdb.set_trace()
        for name, child in model.named_parameters():
            if 'transformer_blocks' in name:
                for element in target_modules_list:
                    if element in name:
                        e_ind = name.index(element)
                        dm_child_list.append(name[:e_ind + len(element)])

        target_modules = list(set(dm_child_list))
        lora_config = LoraConfig(
            r=2,  # 4,8, ..., args.rank
            init_lora_weights="gaussian",
            target_modules=target_modules
        )
        # pdb.set_trace()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
        # pdb.set_trace()


    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,
                c_adm=None, s=None, mask=None, **kwargs):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        elif self.conditioning_key == 'resblockcond':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == 'hybrid-time':
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == 'concat-time-mask':
            # assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == 'concat-adm-mask':
            # assert s is not None
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-adm-mask':
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-time-adm': # adm means y, e.g., class index
            # assert s is not None
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        else:
            raise NotImplementedError()

        return out
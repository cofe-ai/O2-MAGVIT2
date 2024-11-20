# -*- coding:UTF-8 -*-
from modeling.magvit_model import VisionTokenizer
from src.data import VideoDataset, ImageDataset, DataLoader
from src import losses
from transformers import get_cosine_schedule_with_warmup
import torch
import os
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import json
from torchvision.io import write_video, write_png
from einops import rearrange
from modeling.ema import ModelEmaV3 as EMA
from torchvision.models import resnet50
import math
import time
from omegaconf import OmegaConf
import torch.nn.functional as F
import gc
import pathlib

from safetensors.torch import load_model

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,backend:native"
def cycle(dl):
    while True:
        for data in dl:
            yield data

class EntropyLossAnnealingScheduler():
    def __init__(self, value=0.1, scaling_factor=3.0, decay_steps=2000):
        self.decay_steps = decay_steps
        self.max_value = scaling_factor * value
        self.final_value = value

    def step(self, current_step):
        if current_step < self.decay_steps:
            current_value = (self.max_value - self.final_value) / self.decay_steps * (self.decay_steps - current_step) + self.final_value
        else:
            current_value = self.final_value
        return current_value


class CosineWarmupScheduler:
    def __init__(self, initial_value, final_value, warmup_steps, last_step=-1):
        """
        :param param: Parameter to apply the cosine warmup scheduler to.
        :param warmup_steps: Number of steps over which to perform the warmup.
        :param max_steps: Total number of steps for the entire schedule.
        :param initial_value: Initial value of the parameter at the start of warmup.
        :param final_value: Final value of the parameter at the end of warmup.
        :param last_step: The index of the last step. Default is -1.
        """
        self.value = initial_value
        self.warmup_steps = warmup_steps
        self.initial_value = initial_value
        self.final_value = final_value

    def step(self, current_step):
        if current_step <= self.warmup_steps:
            # Cosine warmup
            cos_inner = (math.pi * current_step) / (2 * self.warmup_steps)
            new_value = self.initial_value + (self.final_value - self.initial_value) * math.sin(cos_inner)
            self.value = new_value
            return self.value
        else:
            # Hold final value after warmup period
            self.value = self.final_value
            return self.value

    def get_last_value(self):
        return self.value


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class VideoTokenizerTrainer:

    def __init__(self, model_config, trainer_config):
        self.trainer_config = trainer_config
        self.model_config = model_config
        set_seed(self.model_config.seed)
        self.debug_only = False
        self.use_ema = self.trainer_config.ema.apply_ema
        self.use_gan = self.trainer_config.gan.use_gan
        self.quantizer_type = (self.model_config.model.quantize_model.quantizer_type).lower()

        grad_accum_plugin = GradientAccumulationPlugin(num_steps=self.trainer_config.optimizer.grad_accum_size, sync_each_batch=True, sync_with_dataloader=False)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(log_with="tensorboard", 
                                       project_dir=self.trainer_config.logging.tensorboard_dir,gradient_accumulation_plugin=grad_accum_plugin,
                                       kwargs_handlers=[ddp_kwargs])
        self.dtype = self.get_dtype(self.accelerator.mixed_precision)

        if self.model_config.architecture == "magvit2":
            from modeling.magvit_model import VisionTokenizer
        else:
            raise NotImplementedError
        if self.quantizer_type == "lfq":
            self.model = VisionTokenizer(model_config, 
                                         self.trainer_config.quantizer.commitment_cost, 
                                         self.trainer_config.quantizer.diversity_gamma,
                                         use_gan=self.trainer_config.gan.use_gan,
                                         use_lecam_ema=self.trainer_config.gan.use_lecam_ema,
                                         use_perceptual=self.trainer_config.perceptual.use_perceptual,
                                         perceptual_ckpt_path=self.trainer_config.perceptual.ckpt_path)
            
        elif self.quantizer_type == "vq":
            raise NotImplementedError
        
        if trainer_config.checkpointing.inflate_from_2d is True:
            self.model = self.inflate_2d_to_3d(self.model, trainer_config.checkpointing.inflate_ckpt_path)

        if self.trainer_config.perceptual.use_perceptual:
            self.model.perceptual_model.to(self.dtype)
        self.model.perceptual_model
        self.codebook_size = self.model.codebook_size


        if self.trainer_config.modal == "video":
            dataset = VideoDataset(meta_path=self.trainer_config.data.train_dir, image_size=self.trainer_config.data.spatial_size, num_frames=self.trainer_config.data.num_frames)
            valid_dataset = VideoDataset(meta_path=self.trainer_config.data.valid_dir, image_size=self.trainer_config.data.spatial_size, num_frames=self.trainer_config.data.num_frames)
        elif self.trainer_config.modal == "image":
            dataset = ImageDataset(meta_path=self.trainer_config.data.train_dir, image_size=self.trainer_config.data.spatial_size)
            valid_dataset = ImageDataset(meta_path=self.trainer_config.data.valid_dir, image_size=self.trainer_config.data.spatial_size)


        
        self.train_dataloader = DataLoader(dataset, shuffle=True, drop_last=True, pin_memory=True, num_workers=self.trainer_config.data.num_workers, batch_size=self.trainer_config.data.train_batch_size)
        self.valid_dataloader = DataLoader(valid_dataset, shuffle=False, drop_last=False, pin_memory=True, num_workers=self.trainer_config.data.num_workers, batch_size=self.trainer_config.data.valid_batch_size)


        self.num_epochs = self.trainer_config.data.num_epochs
        self.total_training_steps = self.trainer_config.data.num_epochs * len(self.train_dataloader) // (self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps)

        self.gen_optimizer = self.init_optimizer(self.model.tokenizer, lr_scale=self.trainer_config.lr_scheduler.gen_lr_scale)
        self.discri_optimizer =  self.init_optimizer(self.model.discriminator, lr_scale=self.trainer_config.lr_scheduler.disc_lr_scale)

        self.gen_lr_scheduler = self.init_scheduler(self.gen_optimizer)
        self.discri_lr_scheduler = self.init_scheduler(self.discri_optimizer)

        self.entropy_weight_scheduler = EntropyLossAnnealingScheduler(value=self.trainer_config.quantizer.entropy_loss_weight, scaling_factor=self.trainer_config.quantizer.entropy_loss_scale_factor, decay_steps=self.trainer_config.quantizer.entropy_loss_decay_steps)
        

        self.accelerator.init_trackers(self.trainer_config.exp_name)
        # self.trainer_config.logging.tensorboard_dir
        self.tracker = self.accelerator.get_tracker("tensorboard")

        accelerator_to_prepare = [self.train_dataloader, self.valid_dataloader, self.model, self.gen_optimizer, self.discri_optimizer]


        self.train_dataloader, self.valid_dataloader, self.model, self.gen_optimizer, self.discri_optimizer = self.accelerator.prepare(*accelerator_to_prepare)  
        # del accelerator_to_prepare
        self.accelerator.register_for_checkpointing(self.gen_lr_scheduler, self.discri_lr_scheduler)
        self.accelerator.register_save_state_pre_hook(self.save_state_pre_hook)
        self.accelerator.register_load_state_pre_hook(self.load_state_pre_hook)

        self.step = 0
        self.epoch = 0

        if self.trainer_config.checkpointing.pretrained is not None:
            if self.trainer_config.checkpointing.continue_training is True:
                self.accelerator.load_state(self.trainer_config.checkpointing.pretrained)
                self.continue_training = True
            else:
                # reset 
                load_model(self.model.module.tokenizer, self.trainer_config.checkpointing.pretrained)
                self.continue_training = False
            self.print_global_rank_0(f"successfully load from pretrained {self.trainer_config.checkpointing.pretrained}")

        else:
            self.continue_training = False


        if self.use_ema:
            tokenizer = self.model.module.tokenizer if self.accelerator.num_processes > 1 else self.model.tokenizer
            self.ema_model = EMA(tokenizer, decay=self.trainer_config.ema.decay_rate)
            self.ema_model.to(self.accelerator.device)

        self.ckpt_base_dir = os.path.join(self.trainer_config.io.ckpt_base_dir, self.trainer_config.exp_name)
        
        self.output_base_dir = os.path.join(self.trainer_config.io.output_base_dir, self.trainer_config.exp_name)
        if self.accelerator.process_index == 0:
            os.makedirs(self.ckpt_base_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_base_dir, "input"), exist_ok=True)
            os.makedirs(os.path.join(self.output_base_dir, "output"), exist_ok=True)

        self.codebook_tracker = set()
        gc.collect()
        torch.cuda.empty_cache()      

    @staticmethod
    def get_dtype(dtype_str: str):
        dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        return dtype_map[dtype_str]

    def print_global_rank_0(self, *message):
        if self.accelerator.process_index == 0:
            print(*message)
    @property
    def apply_gradient_penalty(self):
        return self.trainer_config.gan.apply_gradient_penalty and (self.step + 1) % self.trainer_config.gan.apply_gradient_penalty_every == 0

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @staticmethod
    def inflate_2d_to_3d(model: torch.nn.Module, ckpt_path):
        s = torch.load(ckpt_path)
        model.tokenizer.load_state_dict(s, strict=False)
        return model

    def save_state_pre_hook(self, models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], input_dir: str):
        self.trainer_config.current_step = self.step
        self.trainer_config.current_micro_step = self.accelerator.step
        self.trainer_config.current_epoch = self.epoch

        if self.accelerator.process_index == 0:
            OmegaConf.save(self.trainer_config, os.path.join(self.ckpt_base_dir, f"iter_{self.step}", "trainer_config.yaml"))
            OmegaConf.save(self.model_config, os.path.join(self.ckpt_base_dir, f"iter_{self.step}", "model_config.yaml"))
        self.accelerator.wait_for_everyone()

    def load_state_pre_hook(self, models: list[torch.nn.Module], input_dir: str) -> None:
        trainer_config = OmegaConf.load(os.path.join(input_dir, "trainer_config.yaml"))
        model_config = OmegaConf.load(os.path.join(input_dir, "model_config.yaml"))
        self.accelerator.wait_for_everyone()
        self.step = trainer_config.pop("current_step")
        self.epoch = trainer_config.pop("current_epoch")
        self.accelerator.step = trainer_config.pop("current_micro_step")
    

    def train_step(self, batch):
        data, index_strs = batch
        self.print_global_rank_0(f"RECORD: step {self.step}, indices: {list(map(int, index_strs))}")
        # ====
        with self.accelerator.accumulate(self.model):
            is_gradient_accum_boundary = (self.accelerator.step % (self.accelerator.gradient_accumulation_steps)== 0)

            quantizer_entropy_loss_weight = self.entropy_weight_scheduler.step(self.step)
            loss_weights = {
                "recon_loss_weight": self.trainer_config.recon_loss_weight,
                "quantizer_entropy_loss_weight": quantizer_entropy_loss_weight,
                "quantizer_aux_loss_weight": self.trainer_config.quantizer.aux_loss_weight
            }

            if not self.use_gan:
                run_generator=False,
                run_discriminator=False,
                apply_gradient_penalty=False,
            else:
                run_generator = (self.step % 2 == 0)
                run_discriminator = not run_generator

                if run_generator:
                    self.print_global_rank_0("run generator!!!")
                    apply_gradient_penalty = False
                    loss_weights.update(
                        {
                            "g_adversarial_loss_weight": self.trainer_config.gan["g_adversarial_loss_weight"],
                            "perceptual_loss_weight": self.trainer_config.perceptual["perceptual_loss_weight"],
                        }
                    )
                else: # run discriminator
                    self.print_global_rank_0("run discriminator!!!")

                    apply_gradient_penalty = self.apply_gradient_penalty and is_gradient_accum_boundary

                    loss_weights.update(
                        {
                            "lecam_loss_weight": self.trainer_config.gan["lecam_loss_weight"],
                            "gradient_penalty_cost": self.trainer_config.gan["gradient_penalty_cost"],
                            "d_adversarial_loss_weight": self.trainer_config.gan["d_adversarial_loss_weight"]
                         }
                    )
            forward_s = time.time()
            recon_output, indices, total_loss, loss_breakdown, quantize_loss_breakdown = self.model(
                data,
                run_generator=run_generator,
                run_discriminator=run_discriminator,
                apply_gradient_penalty=apply_gradient_penalty,
                discriminator_loss_type=self.trainer_config.gan.discriminator_loss_type,
                generator_loss_type=self.trainer_config.gan.generator_loss_type,
                loss_weights=loss_weights,
                use_distributed_batch_entropy=self.trainer_config.quantizer.use_distributed_batch_entropy
            )
            foward_e = time.time()

            loss_breakdowns = [loss_breakdown]
            if (not self.use_gan) or (run_generator): 
                loss_breakdowns.append(quantize_loss_breakdown)

            tracker_dict = dict()
            for loss_group in loss_breakdowns:
                loss_type = type(loss_group).__name__
                if loss_type == "QuantLossBreakdown":
                    tracker_prefix = "quantize/"
                elif loss_type == "GenLossBreakdown":
                    tracker_prefix = "generator/"
                elif loss_type == "DiscriLossBreakdown":
                    tracker_prefix = "discriminator/"
                else:
                    raise NotImplementedError
                for field, value in zip(loss_group._fields, loss_group):
                    if value is not None:                        
                        avg_val = self.accelerator.gather_for_metrics(value).mean()
                        tracker_dict[tracker_prefix+field] = avg_val.item()
            self.tracker.log(
                
                tracker_dict, step=self.step
            )

            self.tracker.log({
                "quantize/entropy_loss_weight": quantizer_entropy_loss_weight}, step=self.step)
            if self.quantizer_type == "lfq":
                self.tracker.log({
                    "quantize/diversity_gamma": self.trainer_config.quantizer.diversity_gamma}, step=self.step)
            self.accelerator.backward(total_loss)

            if (not self.use_gan) or (run_generator):
                with torch.no_grad():
                    tokens_per_batch = self.accelerator.gather_for_metrics(indices.detach().reshape(-1))
                    unique_tokens_per_batch = tokens_per_batch.unique().tolist()
                    self.tracker.log({"train/batch_unique_code_ratio": round(len(unique_tokens_per_batch) / min(self.codebook_size, len(tokens_per_batch.tolist())), 4)}, step=self.step)
                    del tokens_per_batch, unique_tokens_per_batch
            if is_gradient_accum_boundary:
                last_gen_lr = self.gen_lr_scheduler.get_last_lr()[0]
                last_disc_lr = self.discri_lr_scheduler.get_last_lr()[0]

                should_skip_current_step = (self.accelerator.optimizer_step_was_skipped or torch.isnan(total_loss))
                if not should_skip_current_step:
                    if self.trainer_config.max_grad_norm is not None:
                        self.print_global_rank_0("clip grad norm to 1.0...")
                        if run_generator:
                            self.accelerator.clip_grad_norm_(self.model.module.tokenizer.parameters(), self.trainer_config.max_grad_norm)
                        elif run_discriminator:
                            self.accelerator.clip_grad_norm_(self.model.module.discriminator.parameters(), self.trainer_config.max_grad_norm)

                    if run_generator:
                        self.gen_optimizer.step()
                        self.gen_optimizer.zero_grad()
                        self.gen_lr_scheduler.step()

                    elif run_discriminator:
                        self.discri_optimizer.step()
                        self.discri_optimizer.zero_grad()
                        self.discri_lr_scheduler.step()

                    if self.use_ema and ((not self.use_gan) or run_generator):
                        self.ema_model.update(self.model.module.tokenizer if self.accelerator.num_processes > 1 else self.model.tokenizer, step=self.step)
                else:
                    if run_generator:
                        self.gen_optimizer.zero_grad()
                    elif run_discriminator:
                        self.discri_optimizer.zero_grad()
                    self.print_global_rank_0("[WARN] optimizer_step_was_skipped! ")
                self.step += 1

                if self.accelerator.process_index == 0:
                    self.time_e = time.time()
                    duration = round(self.time_e - self.time_s, 2)
                    self.accum_duration += duration
                    if self.step % 100 == 0:
                        self.print_global_rank_0(f"time consumed for 100 step: {self.accum_duration} seconds")
                    self.time_s = self.time_e
                    info = {"global_step": self.step, "micro_step": self.accelerator.step, "epoch": self.epoch, "duration": duration, "forward_time": round(foward_e - forward_s, 4)}
                    self.print_global_rank_0(f">> TRAINING INFO: {json.dumps(info)}")
                if run_generator:
                    self.tracker.log({"train/generator_lr": last_gen_lr}, step=self.step)
                elif run_discriminator:
                    self.tracker.log({"train/discriminator_lr": last_disc_lr}, step=self.step)
                else:
                    pass

                mem_stats = torch.cuda.memory_stats()
                self.tracker.log({
                                "memory/reserved-bytes": mem_stats["reserved_bytes.all.current"],
                                "memory/allocated-bytes": mem_stats["allocated_bytes.all.current"],
                                "memory/allocated-count": mem_stats["allocation.all.current"]
                                },
                                step=self.step)
                input_video = data

                is_global_step_update = True
            else:
                input_video = None
                is_global_step_update = False
            if apply_gradient_penalty:
                gc.collect()
                # torch.cuda.empty_cache()
        return is_global_step_update, input_video, recon_output

    def get_accum_batches(self, train_dataloader):
        accum_batches = []
        for batch in train_dataloader:
            accum_batches.append(batch)
            if len(accum_batches) >= self.accelerator.gradient_accumulation_steps:
                yield accum_batches
                accum_batches = []
        if len(accum_batches):
            yield accum_batches


    def train(self):

        self.print_global_rank_0(f"using global batch size: {self.train_dataloader.batch_sampler.batch_size} x {self.accelerator.state.num_processes} x {self.trainer_config.optimizer.grad_accum_size}")
        self.model.train()
        if self.accelerator.process_index == 0:
            self.time_s = time.time()

        if self.continue_training:
            micro_steps_to_skip = (self.step * self.accelerator.gradient_accumulation_steps)//2 % len(self.train_dataloader)

            skipped_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, num_batches=micro_steps_to_skip)
            num_trained_epochs = self.epoch
            if micro_steps_to_skip == 0:
                num_trained_epochs += 1
            
        else:
            micro_steps_to_skip = 0
            num_trained_epochs = 0

        self.print_global_rank_0(f"num steps to skip {micro_steps_to_skip}")
        valid_data_iter = iter(cycle(self.valid_dataloader))

        

        for epoch in range(self.num_epochs):
            if epoch < num_trained_epochs:
                continue
            if micro_steps_to_skip > 0:
                self.print_global_rank_0(f"{micro_steps_to_skip} steps trained for epoch {epoch}, continue training for rest...")
                train_dataloader = skipped_dataloader
                micro_steps_to_skip = 0
            else:
                train_dataloader = self.train_dataloader
            self.epoch = epoch
            
            d_s = time.time()
            
            for accum_batches in self.get_accum_batches(train_dataloader):
                d_e = time.time()
                self.print_global_rank_0(f">> TIME: [data loading] {d_e - d_s}")
                for _ in range(2):
                    for batch in accum_batches:
                        with self.accelerator.autocast():
                            if self.step % 100 == 0:
                                self.accum_duration = 0
                            is_global_step_update, input_video, recon_output = self.train_step(batch)
                        self.accelerator.wait_for_everyone()
                        if is_global_step_update:
                            if self.step > 0 and self.step % self.trainer_config.logging.validate_every_step == 0:
                                self.valid_step(valid_data_iter=valid_data_iter)                        
                                self.model.train()
                            if self.step > 0 and self.step % self.trainer_config.checkpointing.checkpoint_every_step == 0:
                                self.save_ckpt(self.step)
                                self.accelerator.save_state(os.path.join(self.ckpt_base_dir, f"iter_{self.step}"))
                d_s = time.time()

    def write_recon_video(self, input_video, decoded_output, prefix=""):
        
        if self.accelerator.process_index == 0:
            write_video(os.path.join(self.output_base_dir, "input", f"input_step{self.step:05d}.mp4"), ((rearrange(input_video.detach().cpu(), "c t h w -> t h w c")+1)*127.5).to(torch.uint8), fps=10)
            write_video(os.path.join(self.output_base_dir, "output", f"{prefix}output_step{self.step:05d}.mp4"), ((rearrange(decoded_output.detach().cpu(), "c t h w -> t h w c").clamp(-1,1)+1) * 127.5).to(torch.uint8), fps=10)
        self.accelerator.wait_for_everyone()

    def write_recon_image(self, input_image, decoded_output, prefix=""):
        
        if self.accelerator.process_index == 0:
            input_fname = os.path.join(self.output_base_dir, "input", f"input_step{self.step:05d}.png")
            output_fname = os.path.join(self.output_base_dir, "output", f"{prefix}output_step{self.step:05d}.png")
            write_png(((input_image.detach().cpu() + 1)*127.5).to(torch.uint8), input_fname, compression_level=0)
            write_png(((decoded_output.detach().cpu().clamp(-1,1) +1)*127.5).to(torch.uint8), output_fname, compression_level=0)
        self.accelerator.wait_for_everyone()

    def valid_step(self, valid_data_iter):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            eval_data, *_ = next(valid_data_iter)
            all_eval_data = self.accelerator.gather_for_metrics(eval_data)

            if self.use_ema:
                eval_data = eval_data.to(torch.float32)
                ema_recon_output, *_ = self.ema_model.module(
                    eval_data,
                    entropy_loss_weight=0.0)
                all_ema_recon_output = self.accelerator.gather_for_metrics(ema_recon_output)
                valid_ema_recon_loss = F.mse_loss(all_eval_data, all_ema_recon_output)
                self.tracker.log({"val/ema_recon_loss": valid_ema_recon_loss.item()}, step=self.step)
            recon_output, *_ = self.model(
                eval_data,
                run_generator=False,
                run_discriminator=False,
                apply_gradient_penalty=False,
                loss_weights={"quantizer_entropy_loss_weight": 0.0, "recon_loss_weight": 0.0, "quantizer_aux_loss_weight": 0.0},
            )
            all_recon_output = self.accelerator.gather_for_metrics(recon_output)
            valid_recon_loss = F.mse_loss(all_eval_data, all_recon_output)
            self.tracker.log({"val/recon_loss": valid_recon_loss.item()}, step=self.step)

        if self.trainer_config.modal == "video":
            self.write_recon_video(eval_data[0].cpu(), recon_output[0].cpu())
            if self.use_ema:
                self.write_recon_video(eval_data[0].cpu(), ema_recon_output[0].cpu(), prefix="ema_")

        elif self.trainer_config.modal == "image":
            self.write_recon_image(eval_data[0].cpu(), recon_output[0].cpu())
            if self.use_ema:
                self.write_recon_image(eval_data[0].cpu(), ema_recon_output[0].cpu(), prefix="ema_")
        del eval_data, recon_output, ema_recon_output, all_eval_data, all_recon_output, all_ema_recon_output
        # gc.collect()
        # torch.cuda.empty_cache()


    def init_optimizer(self, model, lr_scale: float):

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=self.trainer_config.lr_scheduler.learning_rate * lr_scale,
            betas=(self.trainer_config.optimizer.beta1, self.trainer_config.optimizer.beta2),
            weight_decay=0,
            eps=self.trainer_config.optimizer.epsilon,
            foreach=True
        )
        return optimizer

    def init_scheduler(self, optimizer):
        warmup_steps = self.trainer_config.lr_scheduler.warmup_steps
        lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_training_steps)
        return lr_scheduler


    def save_ckpt(self, step):
        if self.use_ema:
            self.accelerator.save(self.ema_model.state_dict(), os.path.join(self.ckpt_base_dir, f"iter_{step}_ema.ckpt"))
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.ckpt_base_dir, f"iter_{step}.ckpt"))
        self.accelerator.wait_for_everyone()
        del unwrapped_model
        gc.collect()
        torch.cuda.empty_cache()

    def load_ckpt(self, step):
        torch.load(os.path.join(self.ckpt_base_dir, f"iter_{step}.ckpt"))

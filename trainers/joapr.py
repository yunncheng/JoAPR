import time
import os.path as osp
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from sklearn.mixture import GaussianMixture
from datasets.data_manager import Dataloader_XU, Dataloader_eval
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, load_checkpoint, load_pretrained_weights
)

from dassl.modeling.ops.utils import sharpen_prob, create_onehot

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.JOAPR.N_CTX
        ctx_init = cfg.TRAINER.JOAPR.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.JOAPR.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.JOAPR.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))



@TRAINER_REGISTRY.register()
class JoAPR(TrainerXU):
    """
    JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models
    https://openaccess.thecvf.com/content/CVPR2024/html/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.html
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.temp = cfg.TRAINER.JOAPR.TEMP
        self.rampup = cfg.TRAINER.JOAPR.RAMPUP
        self.warmup_epoch = cfg.TRAINER.JOAPR.WARMUP_EPOCH
        self.beta = cfg.TRAINER.JOAPR.BETA
        self.alpha1 = cfg.TRAINER.JOAPR.ALPHA1
        self.alpha2 = cfg.TRAINER.JOAPR.ALPHA2
        self.batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.loss = []
        self.accuracy = []
        self.thres = 0.5

    def build_data_loader_eval(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = Dataloader_eval(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains

        self.dm = dm

    def build_data_loader_XU(self, pred=[], prob=[]):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = Dataloader_XU(self.cfg, pred=pred, prob=prob)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains

        self.dm = dm

    def check_cfg(self, cfg):
        assert cfg.TRAINER.JOAPR.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.JOAPR.PREC == "fp32" or cfg.TRAINER.JOAPR.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.JOAPR.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def train(self):
        """Generic training loops."""

        print("Start WarmUp")
        for self.epoch in range(0, self.warmup_epoch):
            self.warmup()

        self.before_train()
        for self.epoch in range(self.start_epoch + self.warmup_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # co-divide
        prob, pred = self.eval_train()
        print("divide data to clean and noisy label")
        self.build_data_loader_XU(pred, prob)

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        # if self.cfg.TRAIN.COUNT_ITER == "train_x":
        #    self.num_batches = len_train_loader_x
        # elif self.cfg.TRAIN.COUNT_ITER == "train_u":
        #    self.num_batches = len_train_loader_u
        # elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
        #   self.num_batches = min(len_train_loader_x, len_train_loader_u)
        # else:
        #   raise ValueError
        self.num_batches = max(len_train_loader_x, len_train_loader_u)

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x_non_onehot, label_x, w_x, input_u, org_labelu, impath_u = self.parse_batch_train(batch_x, batch_u)

        prec = self.cfg.TRAINER.JOAPR.PREC

        if prec == "amp":
            with autocast():
                with torch.no_grad():
                    # Generate pseudo-label for unlabeled data
                    output_u = 0
                    for input_ui in input_u:
                        output_ui = F.softmax(self.model(input_ui), 1)
                        output_u += output_ui
                    output_u /= len(input_u)
                    label_u = sharpen_prob(output_u, self.temp)
                    label_u = [label_u] * len(input_u)
                    label_u = torch.cat(label_u, 0)
                    input_u = torch.cat(input_u, 0)

                    # label refinement of labeled samples
                    output_x = 0
                    for i, input_xi in enumerate(input_x):
                        if i == 0:
                            output_img0 = self.model(input_xi)
                            output_xi = F.softmax(output_img0, 1)
                            output_x += output_xi
                        else:
                            output_xi = F.softmax(self.model(input_xi), 1)
                            output_x += output_xi
                    output_x /= len(input_x)
                    label_x = w_x * label_x + (1 - w_x) * output_x
                    label_x = sharpen_prob(label_x, self.temp)
                    label_x = [label_x] * len(input_x)
                    label_x = torch.cat(label_x, 0)
                    input_x = torch.cat(input_x, 0)

                # mixmatch
                l = np.random.beta(self.beta, self.beta)
                l = max(l, 1 - l)
                all_inputs = torch.cat([input_x, input_u], dim=0)
                all_labels = torch.cat([label_x, label_u], dim=0)
                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                label_a, label_b = all_labels, all_labels[idx]
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_label = l * label_a + (1 - l) * label_b

                mixed_output = self.model(mixed_input)

                Lx = -torch.mean(torch.sum(F.log_softmax(mixed_output, dim=1) * mixed_label, dim=1))

                # regularization
                prior = torch.ones(self.num_classes) / self.num_classes
                prior = prior.to(self.device)
                pred_mean = torch.softmax(mixed_output, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                loss = Lx + self.alpha2 * penalty
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            with torch.no_grad():
                # Generate pseudo-label for unlabeled data
                output_u = 0
                for input_ui in input_u:
                    output_ui = F.softmax(self.model(input_ui), 1)
                    output_u += output_ui
                output_u /= len(input_u)
                label_u = sharpen_prob(output_u, self.temp)
                label_u = [label_u] * len(input_u)
                label_u = torch.cat(label_u, 0)
                input_u = torch.cat(input_u, 0)

                # label refinement of labeled samples
                output_x = 0
                for i, input_xi in enumerate(input_x):
                    if i == 0:
                        output_img0 = self.model(input_xi)
                        output_xi = F.softmax(output_img0, 1)
                        output_x += output_xi
                    else:
                        output_xi = F.softmax(self.model(input_xi), 1)
                        output_x += output_xi
                output_x /= len(input_x)
                label_x = w_x * label_x + (1 - w_x) * output_x
                label_x = sharpen_prob(label_x, self.temp)
                label_x = [label_x] * len(input_x)
                label_x = torch.cat(label_x, 0)
                input_x = torch.cat(input_x, 0)

            # mixmatch
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1 - l)
            all_inputs = torch.cat([input_x, input_u], dim=0)
            all_labels = torch.cat([label_x, label_u], dim=0)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            label_a, label_b = all_labels, all_labels[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_label = l * label_a + (1 - l) * label_b

            mixed_output = self.model(mixed_input)

            Lx = -torch.mean(torch.sum(F.log_softmax(mixed_output, dim=1) * mixed_label, dim=1))

            # regularization
            prior = torch.ones(self.num_classes) / self.num_classes
            prior = prior.to(self.device)
            pred_mean = torch.softmax(mixed_output, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + self.alpha2 * penalty
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc_x": compute_accuracy(output_img0, label_x_non_onehot)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def forward_backward_x(self, batch):
        input, label_no_one_hot, label, _ = self.parse_batch_train_x(batch)
        negloss = NegEntropy()
        prec = self.cfg.TRAINER.JOAPR.PREC
        if prec == "amp":
            with autocast():
                output = self.model(input)
                loss = F.cross_entropy(output, label_no_one_hot)
                penalty = negloss(output)
                loss += self.alpha1 * penalty
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(input)
            loss = F.cross_entropy(output, label_no_one_hot)
            penalty = negloss(output)
            loss += self.alpha1 * penalty
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label_no_one_hot)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def warmup(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_x(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def eval_train(self):
        print("apply GaussianMixture")
        self.set_model_mode("eval")
        self.build_data_loader_eval()
        self.prob_output = []
        data_len = len(self.train_loader_x.dataset)
        losses = torch.zeros(data_len)
        with torch.no_grad():
            for self.batch_idx, batch_x in enumerate(self.train_loader_x):
                input, label, index = self.parse_batch_eval_train(batch_x)
                output = self.model(input)
                predict = F.softmax(output, dim=1)
                for i, lab in enumerate(label):
                    self.prob_output.append(predict[i][lab].cpu())

                loss = F.cross_entropy(output, label, reduction='none')
                if self.epoch >= self.warmup_epoch:
                    probs = torch.softmax(output, dim=1)
                    regular = -torch.sum(probs.log() * probs, dim=1)
                    loss = loss + regular
                for b in range(input.size(0)):
                    losses[index[b]] = loss[b]

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        self.loss.append(losses)

        if self.cfg.TRAINER.JOAPR.AVERAGE_LOSS:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(self.loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)


        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean = gmm.means_.reshape(-1)
        std = gmm.covariances_.reshape(-1)
        idx_clean = mean.argmin()
        idx_noise = mean.argmax()

        mean_clean = torch.tensor(mean[idx_clean]).cuda()
        mean_noise = torch.tensor(mean[idx_noise]).cuda()
        std_clean = torch.tensor(std[idx_clean]).cuda()
        std_noise = torch.tensor(std[idx_noise]).cuda()

        A = std_noise ** 2 - std_clean ** 2
        B = 2 * ((mean_noise * (std_clean ** 2)) - (mean_clean * (std_noise ** 2)))
        C = ((mean_clean * std_noise) ** 2) - ((mean_noise * std_clean) ** 2) + 2 * ((std_noise * std_clean) ** 2) * torch.log(
            (2 * std_clean) / std_noise + 1e-8)
        E = B ** 2 - 4 * A * C
        thres = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()

        pred_1 = (input_loss < thres)
        pred_1 = np.array(pred_1, dtype=bool).reshape(-1, 1)

        self.prob_output = np.array(self.prob_output)
        prob = prob[:,idx_clean]

        pred_2 = np.array(prob > self.thres)
        noise_rate = (self.num_classes * self.cfg.DATASET.NUM_SHOTS - np.count_nonzero(pred_2)) / (self.num_classes * self.cfg.DATASET.NUM_SHOTS)
        self.thres = noise_rate
        pred_2 = np.array((prob > self.thres), dtype=bool).reshape(-1, 1)
        pred_final = np.array(np.logical_or(pred_1, pred_2), dtype=int)

        if self.epoch <= self.warmup_epoch:
            return prob, pred_final
        else:
            return prob.T * self.prob_output, pred_final



    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        if not isinstance(input_x, list):
            input_x = [input_x]
        label_x_non_onehot = batch_x["label"]
        label_x = create_onehot(label_x_non_onehot, self.num_classes)
        prob_x = batch_x["prob_x"]
        prob_x = prob_x.view(-1, 1).type(torch.FloatTensor)
        input_u = batch_u["img"]
        impath_u = batch_u["impath"]
        label_u = batch_u["label"]
        #label_u = create_onehot(label_u_non_onehot, self.num_classes)
        if not isinstance(input_u, list):
            input_u = [input_u]

        input_x = [input_xi.to(self.device) for input_xi in input_x]
        label_x = label_x.to(self.device)
        label_x_non_onehot = label_x_non_onehot.to(self.device)
        prob_x = prob_x.to(self.device)
        input_u = [input_ui.to(self.device) for input_ui in input_u]
        label_u = label_u.to(self.device)

        return input_x, label_x_non_onehot, label_x, prob_x, input_u, label_u, impath_u


    def parse_batch_train_x(self, batch):
        input = batch["img"]
        label_x_non_onehot = batch["label"]
        label = create_onehot(label_x_non_onehot, self.num_classes)
        if isinstance(input, list):
            # label = [label] * len(input)
            # label_x_non_onehot = [label_x_non_onehot] * len(input)
            # input = torch.cat(input, 0)
            # label = torch.cat(label, 0)
            # label_x_non_onehot = torch.cat(label_x_non_onehot,0)
            input = input[0]
        index = batch["index"]
        input = input.to(self.device)
        label_x_non_onehot = label_x_non_onehot.to(self.device)
        label = label.to(self.device)
        return input, label_x_non_onehot, label, index

    def parse_batch_eval_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        index = batch["index"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, index

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


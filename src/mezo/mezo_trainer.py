"""Код был адаптирован отсюда https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py Нужна была поддержка transformers>4.5"""

import math
import time

import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput, speed_metrics


class MeZoTrainer(Trainer):
    """
    Zero-Order (MeZO) Trainer для fine-tuning LLM без обратного распространения.

    Адаптировано из https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py
    Добавлена поддержка transformers >= 4.5.

    MeZO оценивает градиенты методом конечных разностей с помощью двух forward-проходов,
    используя случайные возмущения параметров. Это позволяет обучать модели без
    сохранения графов вычислений и обратного распространения, что значительно
    экономит память GPU.

    Args:
        args (TrainingArguments): Аргументы тренировки. Должны содержать:
            - zo_eps (float): Масштаб возмущения для zeroth-order (по умолчанию 1e-3)
    """

    def _inner_training_loop(
        self,
        batch_size: int = None,
        args: TrainingArguments = None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):

        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(
            len_dataloader / args.gradient_accumulation_steps
        )

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.global_step = 0
        self.state.epoch = 0

        model = self._wrap_model(self.model_wrapped)
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        tr_loss = torch.tensor(0.0, device=args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0

        start_time = time.time()

        for epoch in range(num_train_epochs):
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            for step, inputs in enumerate(train_dataloader):
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                loss = self.zo_step(model, inputs)

                if args.logging_nan_inf_filter and (
                    torch.isnan(loss) or torch.isinf(loss)
                ):
                    loss = torch.zeros_like(loss)

                tr_loss += loss

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len_dataloader:
                    self.zo_update(model)

                    if self.optimizer is not None:
                        self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + ((step + 1) / len_dataloader)

                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    if step % args.logging_steps == 0:
                        self.log({"loss": loss.item(), "step": self.state.global_step})

                    self._maybe_log_save_evaluate(
                        grad_norm=None,
                        tr_loss=tr_loss,
                        model=model,
                        trial=trial,
                        epoch=epoch,
                        ignore_keys_for_eval=ignore_keys_for_eval,
                        start_time=start_time,
                    )

                    if self.state.global_step >= max_steps:
                        break

                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )

            if self.state.global_step >= max_steps:
                break

        self._total_loss_scalar += tr_loss.item()
        train_loss = (
            self._total_loss_scalar / self.state.global_step
            if self.state.global_step > 0
            else 0.0
        )

        metrics = speed_metrics(
            "train",
            start_time,
            num_steps=self.state.global_step,
        )
        metrics["train_loss"] = train_loss

        self.log(metrics, start_time=start_time)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(
            self.state.global_step,
            train_loss,
            metrics,
        )

    # ---------- MeZO core ----------
    def zo_step(self, model, inputs):
        """Оценка градиента методом конечных разностей."""

        if not hasattr(self, "named_parameters_to_optim"):
            self.named_parameters_to_optim = [
                (n, p) for n, p in model.named_parameters() if p.requires_grad
            ]

        self.zo_random_seed = np.random.randint(0, 2**31 - 1)

        # θ + εz
        self._perturb_parameters(scaling_factor=1)
        loss1 = self._forward_loss(model, inputs)

        # θ - εz
        self._perturb_parameters(scaling_factor=-2)
        loss2 = self._forward_loss(model, inputs)

        projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)

        # вернуть параметры
        self._perturb_parameters(scaling_factor=1)

        # gradient accumulation
        if not hasattr(self, "_zo_grad_accum"):
            self._zo_grad_accum = 0.0
            self._zo_accum_steps = 0

        self._zo_grad_accum += projected_grad.item()
        self._zo_accum_steps += 1

        return loss1.detach()

    def zo_update(self, model):
        """Обновление параметров по накопленному ZO-градиенту."""

        if self._zo_accum_steps == 0:
            return

        lr = self._get_learning_rate()

        projected_grad = self._zo_grad_accum / self._zo_accum_steps
        projected_grad = max(min(projected_grad, 10.0), -10.0)

        for name, param in self.named_parameters_to_optim:
            generator = torch.Generator(device=param.device)
            generator.manual_seed(self.zo_random_seed)

            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=param.shape,
                generator=generator,
                device=param.device,
                dtype=param.dtype,
            )

            grad_estimate = projected_grad * z

            with torch.no_grad():
                if any(nd in name.lower() for nd in ("bias", "layernorm")):
                    param -= lr * grad_estimate
                else:
                    param -= lr * (grad_estimate + self.args.weight_decay * param)

        # сброс accumulation
        self._zo_grad_accum = 0.0
        self._zo_accum_steps = 0

    def _perturb_parameters(self, scaling_factor=1):
        """Добавить εz к параметрам."""

        for _, param in self.named_parameters_to_optim:
            generator = torch.Generator(device=param.device)
            generator.manual_seed(self.zo_random_seed)

            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=param.shape,
                generator=generator,
                device=param.device,
                dtype=param.dtype,
            )

            with torch.no_grad():
                param.add_(z, alpha=scaling_factor * self.args.zo_eps)

    def _forward_loss(self, model, inputs):
        """
        Forward без градиентов.
        Dropout отключён намеренно для стабильной ZO-оценки.
        """

        was_training = model.training
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

        if was_training:
            model.train()

        return loss.detach()

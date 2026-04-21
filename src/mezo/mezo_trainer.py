"""Код был адаптирован отсюда https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py Нужна была поддержка transformers>4.5"""

import time

import numpy as np
import torch
from tqdm.auto import tqdm
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
            - zo_eps (float): Масштаб возмущения для zeroth-order (по умолчанию 1e-3).
            - weight_decay (float): Коэффициент регуляризации для не-bias/не-norm параметров.
        Другие параметры передаются родительскому классу Trainer.

    Note:
        В текущей реализации gradient_accumulation_steps не поддерживается
        (всегда равен 1). Оптимизация выполняется после каждого батча."""

    def _inner_training_loop(
        self,
        batch_size: int | None = None,
        args: TrainingArguments | None = None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ) -> TrainOutput:
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()
        if hasattr(train_dataloader, "__len__"):
            num_update_steps_per_epoch = (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            max_steps = (
                args.max_steps
                if args.max_steps > 0
                else args.num_train_epochs * num_update_steps_per_epoch
            )
        else:
            max_steps = args.max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state.max_steps = max_steps
        model = self._wrap_model(self.model_wrapped)
        model.zero_grad()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        start_time = time.time()

        for epoch in range(int(args.num_train_epochs)):
            epoch_iterator = (
                tqdm(train_dataloader, desc=f"Epoch {epoch}")
                if self.is_local_process_zero()
                else train_dataloader
            )
            for step, inputs in enumerate(epoch_iterator):
                # Оценка градиента и loss через MeZO
                loss = self.zo_step(model, inputs)
                tr_loss += loss

                # Обновление параметров после каждого шага (gradient_accumulation_steps == 1)
                self.zo_update(model)
                self.state.global_step += 1

                # Логирование
                if step % args.logging_steps == 0:
                    self.log({"loss": loss.item(), "step": self.state.global_step})

                if self.state.global_step >= max_steps:
                    break

        # Финальные метрики
        train_loss = tr_loss.item() / self.state.global_step
        metrics = speed_metrics("train", start_time, num_steps=self.state.global_step)
        metrics["train_loss"] = train_loss
        self.log(metrics)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    # ---------- MeZO core ----------
    def zo_step(self, model, inputs):
        """Оценить градиент методом конечных разностей, вернуть loss₁."""
        # Какие параметры оптимизировать
        self.named_parameters_to_optim = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        self.zo_random_seed = np.random.randint(1_000_000_000)

        # f(θ + ε·z)
        self._perturb_parameters(scaling_factor=1)
        loss1 = self._forward_loss(model, inputs)

        # f(θ - ε·z)
        self._perturb_parameters(scaling_factor=-2)
        loss2 = self._forward_loss(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # Вернуть модель в исходное состояние
        self._perturb_parameters(scaling_factor=1)
        return loss1

    def zo_update(self, model):
        """Обновить параметры, используя оценённый градиент."""
        lr = self._get_learning_rate()
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            # Ресемплируем тот же z
            z = torch.normal(
                0,
                1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            grad_estimate = self.projected_grad * z

            if any(nd in name for nd in ("bias", "layer_norm", "layernorm")):
                param.data -= lr * grad_estimate
            else:
                param.data -= lr * (grad_estimate + self.args.weight_decay * param.data)

        self.lr_scheduler.step()

    def _perturb_parameters(self, scaling_factor=1):
        """Прибавить к параметрам ε·z (или вычесть, если scaling_factor отрицательный)."""
        torch.manual_seed(self.zo_random_seed)
        for _, param in self.named_parameters_to_optim:
            z = torch.normal(
                0,
                1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data.add_(z, alpha=scaling_factor * self.args.zo_eps)

    def _forward_loss(self, model, inputs):
        """Вычислить loss без градиентов, с отключённым dropout."""
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

import os
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from groundingdino.util.train import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from ema_pytorch import EMA
from typing import Dict, NamedTuple
from groundingdino.util.model_utils import freeze_model_layers, print_frozen_status
from torch.optim.lr_scheduler import OneCycleLR
from groundingdino.util.matchers import build_matcher
from groundingdino.util.inference import GroundingDINOVisualizer,load_image
from groundingdino.util.lora import get_lora_optimizer_params, verify_only_lora_trainable
from datetime import datetime
import yaml
from typing import Dict, Optional, Any
from groundingdino.datasets.dataset import GroundingDINODataset
from groundingdino.util.losses import SetCriterion
from config import ConfigurationManager, DataConfig, ModelConfig
import matplotlib.pyplot as plt

# Ignore tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_model(model_config: ModelConfig, use_lora: bool=False) -> torch.nn.Module:
    md =  load_model(
        model_config.config_path,
        model_config.weights_path,
        use_lora=use_lora,
    )

    if use_lora:
        # 冻结所有原模型参数
        for param in md.parameters():
            param.requires_grad = False
        # 只解冻LORA参数
        for name, param in md.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    return md

def setup_data_loaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = GroundingDINODataset(
        config.train_dir,
        config.train_ann
    )
    
    val_dataset = GroundingDINODataset(
        config.val_dir,
        config.val_ann
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Keep batch size 1 for validation
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader


class GroundingDINOTrainer:
    def __init__(
        self,
        model,
        device="cuda",
        ema_decay=0.999,
        ema_update_after_step=150,
        ema_update_every=20,
        warmup_epochs=5,
        class_loss_coef=1.0,
        bbox_loss_coef=5.0,  
        giou_loss_coef=1.0,  
        learning_rate=2e-4,   
        use_ema=False,      
        num_epochs=500,
        num_steps_per_epoch=None,
        lr_scheduler="onecycle",
        eos_coef=0.1,
        max_txt_len=256,
        use_lora=False
    ):
        self.model = model.to(device)
        self.device = device
        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        if use_lora:
            lora_params = get_lora_optimizer_params(model)
            self.optimizer = torch.optim.AdamW(
                lora_params,
                lr=learning_rate,
                weight_decay=1e-4  # Removed for overfitting
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4  # Removed for overfitting
            )
        
        # Initialize scheduler with warmup
        if lr_scheduler == "onecycle":
            total_steps = num_steps_per_epoch * num_epochs
            # warmup_steps = num_steps_per_epoch * warmup_epochs  
            # self.scheduler = get_cosine_schedule_with_warmup(
            #    self.optimizer,
            #    num_warmup_steps=warmup_steps,
            #    num_training_steps=total_steps
            # )
            # One Cycle LR with warmup
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.1,  # 10% of training for warmup
                div_factor=25,
                final_div_factor=1e4,
                anneal_strategy='cos'
            )
        else:
            # Simple step scheduler
            total_steps = num_steps_per_epoch * num_epochs
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=total_steps // 20, 
                gamma=0.5
            )
        
        # Initialize EMA
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = EMA(
                model,
                beta=ema_decay,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every
            )

        self.matcher = build_matcher(set_cost_class=class_loss_coef * 2,
                                     set_cost_bbox=bbox_loss_coef,
                                     set_cost_giou=giou_loss_coef)
        
        losses = ['labels', 'boxes']
        self.weights_dict = {'loss_ce': class_loss_coef, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        # Give more weightage to bbox loss in loss calculation compared to matcher 
        self.weights_dict_loss = {'loss_ce': class_loss_coef, 'loss_bbox': bbox_loss_coef * 2, 'loss_giou': giou_loss_coef}
        self.criterion = SetCriterion(max_txt_len, self.matcher, eos_coef, losses)
        self.criterion.to(device)

    def prepare_batch(self, batch):
        images, targets = batch
        # Convert list of images to NestedTensor and move to device
        if isinstance(images, (list, tuple)):
            images = nested_tensor_from_tensor_list(images)  # Convert list to NestedTensor
        images = images.to(self.device)

        captions = []
        for target in targets:
            target['boxes'] = target['boxes'].to(self.device)
            target['size'] = target['size'].to(self.device)
            target['labels'] = target['labels'].to(self.device)
            captions.append(target['caption'])
            
        return images, targets, captions

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        # self.get_ema_model().train()
        self.optimizer.zero_grad() 
        # Prepare batch
        images, targets, captions = self.prepare_batch(batch)
        outputs = self.model(images, captions=captions)
        loss_dict = self.criterion(outputs, targets, captions=captions, tokenizer=self.model.tokenizer)
        total_loss = sum(loss_dict[k] * self.weights_dict_loss[k] for k in loss_dict.keys() if k in self.weights_dict_loss)
        # backward pass
        total_loss.backward()
        loss_dict['total_loss'] = total_loss
        # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)
        # print(f"Gradient norm: {total_norm:.4f}")
        self.optimizer.step()
        
        # Step scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update EMA model
        if self.use_ema:
            self.ema_model.update()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_losses = defaultdict(float)
        num_batches = 0
        
        for batch in val_loader:
            images, targets, captions = self.prepare_batch(batch)
            outputs = self.model(images, captions=captions)
            
            # Calculate losses
            loss_dict = self.criterion(outputs, targets, captions=captions, tokenizer=self.model.tokenizer)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                val_losses[k] += v.item()
                
            val_losses['total_loss'] += sum(loss_dict[k] * self.weights_dict[k] 
                                            for k in loss_dict.keys() if k in self.weights_dict_loss).item()
            num_batches += 1

        # Average losses
        return {k: v / num_batches for k, v in val_losses.items()}

    def get_ema_model(self):
        """Return EMA model for evaluation"""
        return self.ema_model.ema_model

    def save_checkpoint(self, path, epoch, losses, use_lora=False):
        """Save checkpoint with EMA and scheduler state""" 
        if use_lora:
            from peft import get_peft_model_state_dict  # 确保导入 get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(self.model)
            checkpoint = {
                'epoch': epoch,
                'model': lora_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'losses': losses,
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'ema_state_dict': self.ema_model.state_dict() if self.use_ema else None,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'losses': losses,
            }
        torch.save(checkpoint, path)

def train(config_path: str, save_dir: Optional[str] = None) -> None:
    """
    Main training function with configuration management
    
    Args:
        config_path: Path to the YAML configuration file
        save_dir: Optional override for save directory
    """

    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

    model = setup_model(model_config, training_config.use_lora)
    
    if save_dir:
        training_config.save_dir = save_dir
    
    # Setup save directory with timestamp
    save_dir = os.path.join(
        training_config.save_dir,
        datetime.now().strftime("%Y%m%d_%H%M")
    )
    os.makedirs(save_dir, exist_ok=True)
    
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'data': vars(data_config),
            'model': vars(model_config),
            'training': vars(training_config)
        }, f, default_flow_style=False)  # 保存config
    
    train_loader, val_loader = setup_data_loaders(data_config)

    # steps_per_epoch = len(train_loader.dataset) // data_config.batch_size  # 计算每一轮要训练几批
    steps_per_epoch = len(train_loader)
    visualizer = GroundingDINOVisualizer(save_dir=save_dir)
    
    if not training_config.use_lora:
        print("Freezing most of model except few layers!")
        freeze_model_layers(model)  # 自定义冻结组块，在module_utils
    
    else:
        print(f"Is only Lora trainable?  {verify_only_lora_trainable(model)} ")
    

    print_frozen_status(model)

    trainer = GroundingDINOTrainer(
        model,
        num_steps_per_epoch=steps_per_epoch,
        num_epochs=training_config.num_epochs,
        warmup_epochs=training_config.warmup_epochs,
        learning_rate=training_config.learning_rate,
        use_lora=training_config.use_lora
    )   
    # Training loop

    # 创建一个txt文件，用于保存val的loss, 仅创建文件
    val_loss_file_path = os.path.join(save_dir, 'val_loss.txt')
    # 创建并打开文件，写入空内容以确保文件存在
    with open(val_loss_file_path, 'w') as file:
        file.write('')

    train_losses_over_epochs = defaultdict(list)
    val_losses_over_epochs = defaultdict(list)
    val_epochs = []

    for epoch in range(training_config.num_epochs):
        valid_epoch_0 = dict()
        if epoch % training_config.visualization_frequency == 0:
            val_losses = trainer.validate(val_loader)  # 平均loss
            val_epochs.append(epoch)
            # 第0轮的结果，可以认为是原模型的验证结果，作为baseline
            if epoch == 0:
                valid_epoch_0 = val_losses
                # 在val_loss_file中追加第0轮的结果
                with open(val_loss_file_path, 'a') as file:
                    file.write(f"Baseline: {valid_epoch_0}\n")
            else: 
                # 输出loss_ce, loss_bbox, loss_giou, total_loss
                print(f"----------- Validation loss at epoch {epoch}: -----------------")
                print(f"Loss_ce: {val_losses['loss_ce']}, Loss_bbox: {val_losses['loss_bbox']}, Loss_giou: {val_losses['loss_giou']}, Total_loss: {val_losses['total_loss']}")
                print(f"-----------------------------------------------------------------")
                # 在val_loss_file中追加当前轮的结果
                with open(val_loss_file_path, 'a') as file:
                    file.write(f"Epoch {epoch}: {val_losses}\n")
            visualizer.visualize_epoch(model, val_loader, epoch, trainer.prepare_batch,box_th=0.3, txt_th= 0)
            # 记录验证损失
            for k, v in val_losses.items():
                val_losses_over_epochs[k].append(v)
        
        epoch_losses = defaultdict(list)
        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch)
            
            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            # if batch_idx % 5 == 0: 
            #     loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
            #     print(f"Epoch {epoch+1}/{training_config.num_epochs}, "
            #           f"Batch {batch_idx}/{len(train_loader)}, {loss_str}")
            #     print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")

        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1} complete. Average losses:",
              ", ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()))
        
        # 记录训练损失
        for k, v in avg_losses.items():
            train_losses_over_epochs[k].append(v)
        
        # Save checkpoint
        if (epoch + 1) % training_config.save_frequency == 0:
            trainer.save_checkpoint(
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch,
                avg_losses,
                use_lora=training_config.use_lora
            )

    # 绘制训练和验证损失的折线图
    def plot_losses(losses_dict, title, save_path, epochs):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses_dict['loss_ce'], label='Loss_ce')
        plt.plot(epochs, losses_dict['loss_bbox'], label='Loss_bbox')
        plt.plot(epochs, losses_dict['loss_giou'], label='Loss_giou')
        plt.plot(epochs, losses_dict['total_loss'], label='Total_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    plot_losses(train_losses_over_epochs, 'Training Losses', os.path.join(save_dir, 'training_losses.png'), range(1, training_config.num_epochs + 1))
    plot_losses(val_losses_over_epochs, 'Validation Losses', os.path.join(save_dir, 'validation_losses.png'), val_epochs)

if __name__ == "__main__":
    train('configs/train_config.yaml')
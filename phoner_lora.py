import torch
import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score
import time
import argparse
from mint.phoner_helpers import PhoNERDataModule

def fft_parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Vietnamese NER"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Select model: 1=PhoBERT-base-v2, 2=PhoBERT-large"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU indices to use, e.g., '0,1,2' (default: '0')"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()

class LoRA4NER(pl.LightningModule):
    def __init__(self, model_name, num_labels=1, lr=2e-5, 
                 lora_r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # initialize model
        base_model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # config LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value", "key"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.TOKEN_CLS
        )
        
        # Aplly LoRA to the model
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()  # print trainable parameters
        
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        # Tính F1-score
        f1 = self._compute_f1(preds, batch["labels"])
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        return {"loss": loss, "f1": f1}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _compute_f1(self, preds, labels):
        # Flatten and filter invalid labels (-100)
        preds = preds.view(-1)
        labels = labels.view(-1)
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]
        
        # move to CPU for sklearn
        return f1_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            average="weighted"
        )

if __name__ == "__main__":
    args = fft_parse_args()

    # Load the tokenizer
    model_name = "vinai/phobert-base-v2"
    if args.model == 2:
        model_name = "vinai/phobert-large"
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)

    data_module = PhoNERDataModule(
        model_name=model_name,
        batch_size=args.batch_size,
        max_length=128
    )

    data_module.prepare_data()

    # Setup model
    model = LoRA4NER(
        model_name=model_name,
        num_labels=len(data_module.label2id),
        lr=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Set up the trainer
    GPUs = [int(gpu) for gpu in args.gpus.split(',')]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=GPUs,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Training
    if trainer.is_global_zero:
        start_time = time.time()

    trainer.fit(model, data_module)
    
    if trainer.is_global_zero:
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")

    # Testing
    trainer.test(model, datamodule=data_module)

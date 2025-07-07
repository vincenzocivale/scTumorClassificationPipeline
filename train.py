# %%
import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, ClassLabel, load_from_disk
from transformers import Trainer, TrainingArguments, default_data_collator
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix

# %%
# Set device to use the second GPU (GPU 1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set environment variable to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
SUB_GROUP = "Lung"
TARGET_COLUMN = "disease"
PERCENTAGE = 1

# %%
dataset = load_from_disk(f"/equilibrium/datasets/TCGA-histological-data/huggingface/lung_embeddings_updated")

# %%
from datasets import Dataset, DatasetDict
from typing import Union, Dict
import random
from collections import Counter

def subsample_stratified(
    data: Union[Dataset, DatasetDict],
    label_column: str,
    sampling_percentage: float,
    seed: int = 42
) -> Union[Dataset, DatasetDict]:
    """
    Subsamples a Hugging Face Dataset or each split of a DatasetDict,
    maintaining the original label distribution (stratified sampling).

    Args:
        data (Union[Dataset, DatasetDict]): The input Dataset or DatasetDict.
        label_column (str): The name of the column containing the labels.
        sampling_percentage (float): The percentage (0.0 to 1.0) of the original
                                     size to sample for each split/dataset.
        seed (int): Random seed for reproducibility.

    Returns:
        Union[Dataset, DatasetDict]: A new Dataset or DatasetDict with subsampled data.
    """
    if not (0 < sampling_percentage <= 1.0):
        raise ValueError("sampling_percentage must be between 0.0 (exclusive) and 1.0 (inclusive).")

    random.seed(seed)

    def _subsample_single_dataset(ds: Dataset) -> Dataset:
        labels = ds[label_column]
        indices_by_label = Counter()
        for i, label in enumerate(labels):
            indices_by_label.setdefault(label, []).append(i)

        selected_indices = []
        for label, indices in indices_by_label.items():
            num_samples_for_label = max(1, int(len(indices) * sampling_percentage))
            num_samples_for_label = min(num_samples_for_label, len(indices)) # Don't oversample
            selected_indices.extend(random.sample(indices, num_samples_for_label))

        random.shuffle(selected_indices) # Shuffle to mix up order
        return ds.select(selected_indices)

    if isinstance(data, Dataset):
        print(f"Subsampling single dataset (size: {len(data)}) by {sampling_percentage*100:.2f}%...")
        return _subsample_single_dataset(data)
    elif isinstance(data, DatasetDict):
        subsampled_dict = DatasetDict()
        for split_name, ds in data.items():
            print(f"Subsampling split '{split_name}' (size: {len(ds)}) by {sampling_percentage*100:.2f}%...")
            subsampled_dict[split_name] = _subsample_single_dataset(ds)
        return subsampled_dict
    else:
        raise TypeError("Input 'data' must be a Hugging Face Dataset or DatasetDict.")

dataset = subsample_stratified(
        dataset,
        label_column="cell_type",
        sampling_percentage=PERCENTAGE,  
        seed=42
    )

# %%
import torch
import numpy as np
from torch import nn

def compute_class_weights_multiclass(labels, num_classes: int) -> torch.Tensor:
    """
    Calcola i pesi per una classificazione multi-class (CrossEntropy)

    Args:
        labels (array-like): array/list/tensor 1D con i target [0, 1, 2, ...]
        num_classes (int): numero totale di classi

    Returns:
        torch.Tensor: pesi normalizzati per ogni classe (shape: [num_classes])
    """
    # Converti in torch.Tensor per uniformità
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # Evita divisioni per zero

    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.min()  # Normalizza (peso minimo = 1.0)

    return weights


all_labels = dataset["train"][TARGET_COLUMN]  # Evita dataset["train"][:]

# Calcola numero classi automaticamente
num_classes = len(np.unique(all_labels))


# Mostra info base
print(f"Classi uniche: {np.unique(all_labels)}")
print(f"Numero classi: {num_classes}")

# Calcola pesi
class_weights = compute_class_weights_multiclass(all_labels, num_classes)

# Loss con pesi
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Per debug
print(f"Pesi per classe: {class_weights}")


# %%
len(class_weights)

# %%
dataset['train'].features['cell_type']

# %% [markdown]
# ### Model

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualMLPConv1D(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim,
        dropout_rate=0.2, use_residual=True,
        use_conv=True, conv_channels=64, kernel_size=3
    ):
        super().__init__()

        self.use_conv = use_conv

        # Normalizzazione iniziale
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Blocco convoluzionale opzionale
        if use_conv:
            self.conv_block = nn.Sequential(
                nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(conv_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(1),
                nn.ReLU()
            )

        # Primo layer MLP
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Blocchi nascosti
        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                hidden_layers.append(ResidualBlock(hidden_dims[i], dropout_rate))
            else:
                hidden_layers.extend([
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Bottleneck + output
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dims[-1] // 2, output_dim)

        self._initialize_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        x = self.input_bn(input_ids)

        if self.use_conv:
            x = x.unsqueeze(1)  # [B, 1, D]
            x = self.conv_block(x)
            x = x.squeeze(1)    # [B, D]

        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.bottleneck(x)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers.modeling_outputs import SequenceClassifierOutput


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual and (input_dim == output_dim)

        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity
        return x


class AdvancedMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.2,
        use_residual_in_hidden: bool = True,
        loss_fn: Optional[nn.Module] = None
    ):
        super().__init__()



        self.initial_bn = nn.BatchNorm1d(input_dim)

        all_dims = [input_dim] + hidden_dims
        mlp_layers = []
        for i in range(len(all_dims) - 1):
            mlp_layers.append(
                MLPBlock(
                    input_dim=all_dims[i],
                    output_dim=all_dims[i + 1],
                    dropout_rate=dropout_rate,
                    use_residual=use_residual_in_hidden and (all_dims[i] == all_dims[i + 1])
                )
            )
        self.hidden_network = nn.Sequential(*mlp_layers)
        self.output_projection = nn.Linear(all_dims[-1], output_dim)
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        self._initialize_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> SequenceClassifierOutput:

        if input_ids.ndim > 2:
            input_ids = input_ids.view(input_ids.size(0), -1)  # Flatten if necessary

        x = self.initial_bn(input_ids)
        x = self.hidden_network(x)
        logits = self.output_projection(x)

        loss = self.loss_fn(logits, labels) if labels is not None else None

        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# %%
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Connessione residuale
        return self.relu(out)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2, use_residual=True):
        super(ResidualMLP, self).__init__()

        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Prima layer con dimensione differente
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Blocchi nascosti
        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                hidden_layers.append(ResidualBlock(hidden_dims[i], dropout_rate))
            else:
                hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                hidden_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                hidden_layers.append(nn.ReLU())
                hidden_layers.append(nn.Dropout(dropout_rate))

        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Layer di output con una piccola bottleneck prima della classificazione
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(hidden_dims[-1] // 2, output_dim)

        # Inizializzazione dei pesi più efficace
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, labels=None):
        x = self.input_bn(input_ids)
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.bottleneck(x)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            # Utilizziamo focale loss per gestire meglio classi sbilanciate
            if hasattr(self, 'loss_fn'):
                loss = self.loss_fn(logits, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers.modeling_outputs import SequenceClassifierOutput

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class ImprovedMLPBlock3(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        # layer
        self.linear = nn.Linear(input_dim, output_dim)
        # layer scale
        self.ls = LayerScale(output_dim, init_value=0.1)
        # batch norm + activation + dropout
        self.bn = nn.BatchNorm1d(output_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)
        # identity proj if dims differ
        self.need_proj = (input_dim != output_dim)
        if self.need_proj:
            self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.linear(x)
        x = self.ls(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        if self.need_proj:
            identity = self.proj(identity)
        return x + identity

class ImprovedMLPClassifier3(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.3,
        weight_decay: float = 1e-4,
        label_smoothing: Optional[float] = None,
        loss_fn: Optional[nn.Module] = None
    ):
        super().__init__()
        self.initial_bn = nn.BatchNorm1d(input_dim)
        # build hidden blocks
        dims = [input_dim] + hidden_dims
        blocks = []
        for i in range(len(dims)-1):
            blocks.append(
                ImprovedMLPBlock3(
                    input_dim=dims[i],
                    output_dim=dims[i+1],
                    dropout_rate=dropout_rate
                )
            )
        self.hidden_net = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(dims[-1], output_dim)

        # loss with optional label smoothing
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            if label_smoothing:
                self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                self.loss_fn = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform works well with GELU
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> SequenceClassifierOutput:
        # flatten if needed
        if input_ids.ndim > 2:
            input_ids = input_ids.view(input_ids.size(0), -1)

        x = self.initial_bn(input_ids)
        x = self.hidden_net(x)
        logits = self.output_proj(x)

        loss = self.loss_fn(logits, labels) if labels is not None else None

        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


# %%
from transformers.modeling_outputs import SequenceClassifierOutput

# SIMPLE AND FAST MLP FOR QUICK TRAINING
class FastMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(FastMLP, self).__init__()
        
        # Simple architecture without complex blocks
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, input_ids, labels=None):
        logits = self.network(input_ids)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)

# ORIGINAL COMPLEX MODEL (commented out for comparison)
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(ImprovedMLP, self).__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim

        # Costruisci più livelli nascosti
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, labels=None):
        x = self.input_bn(input_ids)  # Normalizzazione del batch
        x = self.hidden_layers(x)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

input_dim = len(dataset["train"][0]["embedding"])
labels = np.unique(dataset['test'][TARGET_COLUMN])
output_dim = len(labels)

# FASTER TRAINING: Use smaller, simpler architecture
hidden_dims = [3072, 1536, 768] # Reduced from [3072, 1536, 768]

model = AdvancedMLPClassifier(input_dim, hidden_dims, output_dim, loss_fn=loss_fn)

hidden_str = "hdim_" + "x".join(map(str, hidden_dims))
print(f"Model architecture: {hidden_str}")
print(f"Input dim: {input_dim}, Output dim: {output_dim}")


# %% [markdown]
# ### Training

# %%
current_time = datetime.now()

run_name = f"AdvancedMLPClassifier_{hidden_str}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"

# %%
dataset = dataset.rename_columns({
    'embedding': 'input_ids',   # solo se è un array di interi!
    TARGET_COLUMN: 'labels'       # target
})

# %%
from transformers import EarlyStoppingCallback
output_dir=f"/equilibrium/datasets/TCGA-histological-data/lung/checkpoints/{run_name}"


wandb.init(
    project="scTumorClassification",  
    group=SUB_GROUP,                       
    name=run_name,
    tags=[TARGET_COLUMN, str(PERCENTAGE)]
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# OPTIMIZED TRAINING ARGUMENTS FOR FASTER TRAINING
training_args = TrainingArguments(
    output_dir=output_dir,
    
    # REDUCED EPOCHS - most important change
    num_train_epochs=150,  # Reduced from 400 to 50
    
    # EVALUATION STRATEGY - evaluate every N steps instead of every epoch
    eval_strategy="epoch",
    
    # SAVING STRATEGY - save less frequently
    save_strategy="epoch",  # Save every epoch
    save_total_limit=3,  # Keep only 3 best checkpoints
    
    # BATCH SIZE - you can try increasing this if you have GPU memory
    per_device_train_batch_size=1024,  #
    per_device_eval_batch_size=1024,  
    
    # LEARNING RATE - slightly higher since we have fewer epochs
    learning_rate=1e-3,  # Increased from 5e-4 to 1e-3
    
    # LOGGING - reduce logging frequency
    logging_steps=100,  # Log every 100 steps
    
    # OTHER OPTIMIZATIONS
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Use F1 score as primary metric
    greater_is_better=True,
    report_to="wandb",
    remove_unused_columns=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    dataloader_num_workers=12,
    run_name=run_name,
    
    # DISABLE SOME OVERHEAD
    dataloader_pin_memory=True,  # Faster data loading
    ignore_data_skip=True,  # Skip data loading optimizations
)

# EARLY STOPPING - more aggressive to prevent overfitting
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Reduced from 10 to 5
    early_stopping_threshold=0.001  # Stop if improvement < 0.1%
)


# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def log_confusion_matrix(y_true_ids, y_pred_ids, ds, column_name, title, save_path=None):
    # Usa ClassLabel se presente
    label_feature = ds.features[column_name]

    if hasattr(label_feature, "int2str"):
        num_classes = label_feature.num_classes
        class_names = [label_feature.int2str(i) for i in range(num_classes)]
    else:
        num_classes = len(np.unique(y_true_ids))
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(y_true_ids, y_pred_ids, labels=range(num_classes))
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_annot = np.empty_like(cm, dtype=object)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm.sum(axis=1)[i] > 0:
                cm_annot[i, j] = f"{cm_percent[i, j]:.1f}%\n({cm[i, j]})"
            else:
                cm_annot[i, j] = f"0.0%\n({cm[i, j]})"

    fig = plt.figure(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4)))

    font_size = max(6, min(12, 80 / len(class_names)))

    sns.heatmap(cm_percent, annot=cm_annot, fmt='', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False,
                annot_kws={"fontsize": font_size, "ha": "center", "va": "center"})
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Se richiesto, salva in PDF
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format='pdf')
        print(f"[Salvato] Confusion matrix salvata in: {save_path}")

    # Logga su W&B se disponibile
    try:
        import wandb
        wandb.log({title: wandb.Image(fig)})
    except ImportError:
        pass

    plt.close(fig)



# === Predizione e plot su validation ===
val_preds = trainer.predict(dataset["validation"])
val_y_true = val_preds.label_ids
val_y_pred = np.argmax(val_preds.predictions, axis=1)
log_confusion_matrix(
    val_y_true, val_y_pred,
    dataset["validation"],
    column_name="labels",
    title=f"{SUB_GROUP} - Validation",
    save_path=f"results/confusion_matrices/{SUB_GROUP}_validation.pdf"
)




# === Predizione e plot su test ===
test_preds = trainer.predict(dataset["test"])
test_y_true = test_preds.label_ids
test_y_pred = np.argmax(test_preds.predictions, axis=1)

log_confusion_matrix(
    test_y_true, test_y_pred,
    dataset["test"],
    column_name="labels",
    title=f"{SUB_GROUP} - Test",
    save_path=f"results/confusion_matrices/{SUB_GROUP}_test.pdf"
)

# %%
output_dir = f"saved_models/{run_name}"
trainer.save_model(output_dir)

artifact = wandb.Artifact(name=run_name, type="model")
artifact.add_dir(output_dir)
wandb.log_artifact(artifact)

# %%
# run_name = "AdvancedMLPClassifier_hdim_1024x512_2025-07-05_11-47-25"
# output_dir=f"/equilibrium/datasets/TCGA-histological-data/lung/checkpoints/{run_name}/checkpoint-106953/"

# input_dim = len(dataset["train"][0]["input_ids"])
# labels = np.unique(dataset['test']["labels"])
# output_dim = len(labels)
# hidden_dims = [1024, 512]

# # === Ricarica modello ===
# model = AdvancedMLPClassifier(input_dim, hidden_dims, output_dim, loss_fn=loss_fn)
# from safetensors.torch import load_file
# model.load_state_dict(load_file(f"{output_dir}/model.safetensors"))
# model.eval()

# # === Ricarica Trainer (senza training) ===
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_eval_batch_size=256,
#     dataloader_num_workers=16,
#     report_to="wandb",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args
# )

# # === Funzione per plottare matrice di confusione ===
# def log_confusion_matrix(y_true_ids, y_pred_ids, ds, column_name, title):
#     # Usa ClassLabel se presente
#     label_feature = ds.features[column_name]
   
#     if hasattr(label_feature, "int2str"):
#         num_classes = label_feature.num_classes
#         class_names = [label_feature.int2str(i) for i in range(num_classes)]
#     else:
#         # fallback: nomi generici
#         num_classes = len(np.unique(y_true_ids))
#         class_names = [str(i) for i in range(num_classes)]
    
#     cm = confusion_matrix(y_true_ids, y_pred_ids, labels=range(num_classes))
#     cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
#     cm_annot = np.empty_like(cm, dtype=object)
    
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             if cm.sum(axis=1)[i] > 0:
#                 # Changed format: percentage first, then count in parentheses
#                 cm_annot[i, j] = f"{cm_percent[i, j]:.1f}%\n({cm[i, j]})"
#             else:
#                 cm_annot[i, j] = f"0.0%\n({cm[i, j]})"
    
#     plt.figure(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4)))
    
#     # Calculate font size based on number of classes to prevent overlap
#     font_size = max(6, min(12, 80 / len(class_names)))
    
#     sns.heatmap(cm_percent, annot=cm_annot, fmt='', cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names,
#                 cbar=False,
#                 annot_kws={"fontsize": font_size, "ha": "center", "va": "center"})
#     plt.title(title)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.tight_layout()
#     wandb.log({title: wandb.Image(plt)})
#     plt.close()

# # === Inizializza wandb ===
# wandb.init(
#     project="scTumorClassification",
#     group=SUB_GROUP,
#     name=run_name,
#     tags=[TARGET_COLUMN, str(PERCENTAGE)],
#     resume="allow",
# )

# # === Predizione e plot su validation ===
# val_preds = trainer.predict(dataset["validation"])
# val_y_true = val_preds.label_ids
# val_y_pred = np.argmax(val_preds.predictions, axis=1)
# log_confusion_matrix(val_y_true, val_y_pred, dataset["validation"], column_name="labels", title=f"{SUB_GROUP} - Validation")

# # === Predizione e plot su test ===
# test_preds = trainer.predict(dataset["test"])
# test_y_true = test_preds.label_ids
# test_y_pred = np.argmax(test_preds.predictions, axis=1)
# log_confusion_matrix(test_y_true, test_y_pred, dataset["test"], column_name="labels", title=f"{SUB_GROUP} - Test")





from datasets import load_dataset, Audio
import numpy as np
from transformers import AutoModelForAudioClassification
import os
from transformers import TrainingArguments, get_linear_schedule_with_warmup, Trainer
import torch.optim as optim
import numpy as np
import evaluate
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"


### Part 1: Preparing the dataset ### 

# Loads the GTZAN dataset
gtzan = load_dataset("marsyas/gtzan", "all")
# Create validation set
gtzan = gtzan['train'].train_test_split(seed=42, shuffle=True, test_size=0.1)
# Use the int2str() method of the genre feature to map integers to human-readable names
id2label_fn = gtzan["train"].features["genre"].int2str
id2label_fn(gtzan["train"][0]["genre"])

### Part 2: Preparing the model ###
# Instantiating the feature extractor for DistilHuBERT from the pre-trained checkpoint
from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)

# Resample the audio fil to 16,000Hz
sampling_rate = feature_extractor.sampling_rate
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

# Normalize the audio data, by rescaling each sample to zero mean and unit variance,
# a process called feature scaling
sample = gtzan["train"][0]["audio"]
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])

# Define the max duration of the input audio
max_duration = 30.0

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs

# Use map() method to apply the whole preprocess into dataset
gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)

# Rename the genre column to label: enable the Trainer to process the class labels
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")

# Obtain label mappings from the dataset
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(gtzan_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

id2label["7"]

### Part 3: Training the model ###
# Load a model
num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

# Define the training arguments
model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-gtzan",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
    seed=42,
    lr_scheduler_type="linear",
)

# Define custom Adam optimizer with specific betas and epsilon values
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08
        )

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )

# Define the learning rate scheduler
total_steps = len(gtzan_encoded["train"]) * num_train_epochs // batch_size

# Define the metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# Instantiate the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,  # Assuming you've already defined this
    compute_metrics=compute_metrics
)

# Empty the cache
torch.cuda.empty_cache()

# Train the model
trainer.train()

kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}

trainer.push_to_hub(**kwargs)
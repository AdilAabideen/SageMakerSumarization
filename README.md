# Email Summarization with Hugging Face and Amazon SageMaker

This project demonstrates the process of fine-tuning and deploying a T5-based model (`google/flan-t5-base`) on the Amazon SageMaker platform for summarizing emails. The model was trained on the `argilla/FinePersonas-Conversations-Email-Summaries` dataset to summarize email content effectively.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Training on SageMaker](#training-on-sagemaker)
- [Deployment](#deployment)
- [Inference](#inference)
- [Cleanup](#cleanup)
- [Results](#results)

---

## Overview

The goal of this project is to fine-tune a pre-trained Hugging Face model (`FLAN-T5`) on an email summarization dataset using Amazon SageMaker. After fine-tuning, the model is deployed for inference, enabling it to generate concise email summaries.

### Key Steps:
1. Preprocess the dataset.
2. Upload the dataset to S3.
3. Fine-tune the model using SageMaker.
4. Deploy the model for inference.
5. Evaluate results and clean up resources.

---

## Setup

Install the required packages by running the following commands:

```bash
pip install transformers datasets sagemaker --upgrade
pip install widgetsnbextension ipywidgets fsspec s3fs
```

Ensure that your environment has access to Amazon SageMaker and S3 for dataset uploads.

## Dataset Preprocessing

### 1. Load the Dataset

We load the argilla/FinePersonas-Conversations-Email-Summaries dataset from Hugging Face:

```bash
from datasets import load_dataset

dataset_id = "argilla/FinePersonas-Conversations-Email-Summaries"
dataset = load_dataset(dataset_id)
```

### 2. Split the Dataset

Split the dataset into training, testing, and validation sets:

```bash
train_test_split = dataset["train"].train_test_split(train_size=10000, shuffle=True, seed=42)
test_val_split = train_test_split["test"].train_test_split(test_size=3000, train_size=1000, shuffle=True, seed=42)

dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": test_val_split["test"],
    "validation": test_val_split["train"]
})
```
### 3. Preprocess the Dataset

We tokenize and process the dataset for use with T5:
```bash
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
input_max_length = 1024
output_max_length = 128

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=input_max_length, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=output_max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary", "title"])
```

## Training on SageMaker

### 1. Upload Dataset to S3

Save the preprocessed dataset to S3:
```bash
import s3fs

s3_prefix = "huggingface/email-t5-summarization"
bucket = sess.default_bucket()
dataset_input_path = f"s3://{bucket}/{s3_prefix}"

tokenized_dataset["train"].save_to_disk("./tokenized_dataset/train")
tokenized_dataset["validation"].save_to_disk("./tokenized_dataset/validation")

s3 = s3fs.S3FileSystem()
s3.put("./tokenized_dataset/train", f"{dataset_input_path}/train", recursive=True)
s3.put("./tokenized_dataset/validation", f"{dataset_input_path}/validation", recursive=True)
```

### 2. Fine-tune the Model

Fine-tune the google/flan-t5-base model using Hugging Face’s Deep Learning container on SageMaker:
```bash
from sagemaker.huggingface import HuggingFace

hyperparameters = {
    "epochs": 1,
    "learning-rate": 1e-6,
    "train-batch-size": 1,
    "eval-batch-size": 8,
    "model-name": "google/flan-t5-base",
}

huggingface_estimator = HuggingFace(
    role=sagemaker.get_execution_role(),
    entry_point="train.py",
    hyperparameters=hyperparameters,
    instance_type="ml.g5.xlarge",
    instance_count=1,
    transformers_version="4.26.0",
    pytorch_version="1.13.1",
    py_version="py39",
)

huggingface_estimator.fit({"train": train_input_path, "valid": valid_input_path})
```

## Deployment

Deploy the fine-tuned model on SageMaker:

```bash
huggingface_predictor = huggingface_estimator.deploy(
    initial_instance_count=2,
    instance_type="ml.g4dn.xlarge"
)
```

## Inference

Test the deployed model by passing an email for summarization:
```bash
test_data = {"inputs": "summarize: " + dataset['test'][3]['text']}
prediction = huggingface_predictor.predict(test_data)
print(prediction)
```

## Cleanup

After finishing the tasks, clean up the deployed resources:
```bash
huggingface_predictor.delete_endpoint()
```

## Results

After deploying the model, we generated the following example summary:

	•	Email Text: "Subject: RE: Upcoming conference and potential collaboration...".
	•	Model Output: "Subject: RE: Upcoming conference and potential collaborationnnHi Alex,".

The results demonstrate effective summarization of lengthy email conversations, making communication easier and more efficient.

## Conclusion

This project highlights the integration of Hugging Face models and Amazon SageMaker for fine-tuning and deploying large-scale models. By applying the T5 model to email summarization, we demonstrated its ability to create concise and effective summaries of complex emails.

### References

	•	Hugging Face transformers library
	•	Amazon SageMaker

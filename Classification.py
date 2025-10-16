import json
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

class Classification():

    def __init__(self, model_path: str, device: torch.device, train_data_path: str, test_data_path: str) -> None:
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def get_train_data(self) -> Dataset:
        with open(self.train_data_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        train_data = Dataset.from_dict({
            "text": [data["text"] for data in data],
            "label": [data["label"] for data in data]
        })
        return train_data
    
    def get_test_data(self) -> Dataset:
        with open(self.test_data_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        test_data = Dataset.from_dict({
            "text": [data["text"] for data in data],
            "label": [data["label"] for data in data]
        })
        return test_data
    
    def tokenize_data(self, data: Dataset) -> Dataset:
        return self.tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)

    def train(self) -> None:
        # get train and test data
        train_data = self.get_train_data()
        test_data = self.get_test_data()
        # load BERT fine-tune resources
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2).to(self.device)
        # preprocess data
        tokenized_train_data = train_data.map(self.tokenize_data, batched=True)
        tokenized_test_data = test_data.map(self.tokenize_data, batched=True)
        # set training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_dir='./logs',
        )
        # build trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_test_data,
        )
        # start to train
        trainer.train()
        # store model and tokenizer
        self.model.save_pretrained("./classification_model")
        self.tokenizer.save_pretrained("./classification_model")
    
    # need to build model and tokenizer berfore predict, and do not want this process repeat so I seperate this into function
    def create_predict_env(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "種類" if prediction == 0 else "演員"

    

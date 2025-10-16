import torch
from AutomateTokenization import AutomateTokenization
from Ner import Ner
from Classification import Classification
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification

# detect which device to conduct training
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():

    # init settings
    jieba_extra_dic_path = "./resources/jieba_extra_dic.json"
    automate_tokenization = AutomateTokenization(jieba_extra_dic_path)

    # get training device
    device = get_device()
    
    # set path
    ner_model_path = "./ner_model"
    ner_train_data_path = "./resources/train_data.json"

    # create model
    ner = Ner(ner_model_path, device, ner_train_data_path)
    
    # train ner model
    ner.train()
    ner.create_predict_env()
    
    while True:
        user_input = input("請輸入今天的需求: ")
        actresses, categories = automate_tokenization.detect_key_word(user_input)
        print(actresses)
        print(categories)
    

if __name__ == "__main__":
    main()
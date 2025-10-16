import jieba
import json

class AutomateTokenization():

    def __init__(self, jieba_extra_dic_path: str):
        with open(jieba_extra_dic_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # load specific data into set due to search time complexity O(1)
        self.actresses = set(data["actress"])
        self.categories = set(data["category"])

        # add each actress name and category to jieba's dictionary
        for actress in self.actresses:
            jieba.add_word(actress, freq=10000, tag="nz")
        for category in self.categories:
            jieba.add_word(category, freq=10000, tag="nz")

    def detect_key_word(self, user_input: str):
        # tokenize user input
        seg_list = jieba.cut(user_input, cut_all=False)

        actresses = []
        categories = []

        # detect specific data in seg
        for seg in seg_list:
            if seg in self.actresses:
                actresses.append(seg)
            if seg in self.categories:
                categories.append(seg)

        return actresses, categories

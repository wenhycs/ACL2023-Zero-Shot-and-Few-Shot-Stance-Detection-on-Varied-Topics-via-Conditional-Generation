import json
import torch
import pickle
import transformers
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class StanceData(Dataset):

    def __init__(self, data_name, tokenizer, data_augment=None, wiki_path=None):
        self.data_file = pd.read_csv(data_name)
        self.data = dict()
        self.tokenizer = tokenizer
        self.max_length = 300

        if wiki_path:
            self.wiki_dict = pickle.load(open(wiki_path, "rb"))
            self.max_length = 512

        self.preprocess_data()
        if type(self.tokenizer) in [transformers.BartTokenizer, transformers.BartTokenizerFast]:
            self.gen_data = dict()
            self.preprocess_data_gen(data_augment)
            self.data = self.gen_data

    def preprocess_data(self):
        self.data["text"] = []
        self.data["topic"] = []
        self.data["new_topic"] = []
        self.data["input_ids"] = []
        self.data["attention_mask"] = []
        self.data["topic_type"] = []
        self.data["stance_label"] = []
        self.data["instance_id"] = []
        self.data["data_type_mark"] = []

        # max_len = 0
        for i in self.data_file.index:
            row = self.data_file.iloc[i]
            self.data["text"].append(row["post"])
            self.data["topic"].append(row["topic_str"])
            self.data["topic_type"].append(row["type_idx"])
            self.data["stance_label"].append(row["label"])
            self.data["instance_id"].append(row["new_id"])
            self.data["data_type_mark"].append(row["seen?"])
            self.data["new_topic"].append(row["new_topic"])

            if hasattr(self, "wiki_dict"):
                wiki_summary = self.wiki_dict[row["new_topic"]]
                topic_text = f"text: {row['post']}, topic: {row['topic_str']}"
                tokenized_text = self.tokenizer(text=topic_text, text_pair=wiki_summary,
                                                padding='max_length', truncation=True,
                                                max_length=self.max_length)
            else:
                tokenized_text = self.tokenizer(text=row["topic_str"], text_pair=row["post"],
                                                padding='max_length', max_length=self.max_length)
            self.data["input_ids"].append(torch.LongTensor(tokenized_text["input_ids"]))
            self.data["attention_mask"].append(torch.LongTensor(tokenized_text["attention_mask"]))

        # print(max_len)
        self.data["stance_label"] = torch.LongTensor(self.data["stance_label"])
        self.data["data_type_mark"] = torch.LongTensor(self.data["data_type_mark"])

    def preprocess_data_gen(self, data_augment=None):
        self.tokenizer.add_tokens(["<stance>", "<topic>"])
        self.gen_data["input_template"] = []
        self.gen_data["output_template"] = []
        self.gen_data["stance_text"] = []
        self.gen_data["context"] = []
        self.gen_data["input_ids"] = []
        self.gen_data["attention_mask"] = []
        self.gen_data["output_input_ids"] = []
        self.gen_data["output_attention_mask"] = []
        self.gen_data["data_type_mark"] = []
        self.gen_data["negative_loss_mask"] = []
        if data_augment == 'predict_topic_neg_1':
            offset = 1
            data_augment = 'predict_topic'
        elif data_augment == 'predict_topic_neg_2':
            offset = 2
            data_augment = 'predict_topic'
        elif data_augment == 'predict_stance_neg_1':
            offset = 1
            data_augment = 'predict_stance'
        elif data_augment == 'predict_stance_neg_2':
            offset = 2
            data_augment = 'predict_stance'
        else:
            offset = 0
        for i in range(len(self.data["input_ids"])):
            # template = "Topic is <topic>. Stance is <stance>."

            permuted_stance = (self.data['stance_label'][i] + offset) % 3
            stance_text = 'opposite' if permuted_stance == 0 else 'supportive' if permuted_stance == 1 else 'netural'
            if data_augment == "predict_topic":
                input_template = f"Stance is {stance_text}. Target is <topic>."
                output_template = f"Stance is {stance_text}. Target is "
                offset_begin = len(output_template)
                output_template += f"{self.data['topic'][i]}"
                offset_end = len(output_template)
                output_template += "."
            else:
                input_template = f"Target is {self.data['topic'][i]}. Stance is <stance>."
                output_template = f"Target is {self.data['topic'][i]}. Stance is "
                offset_begin = len(output_template)
                output_template += f"{stance_text}"
                offset_end = len(output_template)
                output_template += "."
            context = self.data["text"][i]

            if hasattr(self, "wiki_dict"):
                wiki_summary = self.wiki_dict[self.data["new_topic"][i]]
                context = context + "</s></s>" + wiki_summary

            self.gen_data["input_template"].append(input_template)
            self.gen_data["output_template"].append(output_template)
            self.gen_data["stance_text"].append(stance_text)
            self.gen_data["context"].append(context)

            tokenized_text = self.tokenizer(text=input_template, text_pair=context,
                                            padding='max_length', max_length=self.max_length,
                                            truncation=True)

            self.gen_data["input_ids"].append(torch.LongTensor(tokenized_text["input_ids"]))
            self.gen_data["attention_mask"].append(torch.Tensor(tokenized_text["attention_mask"]))

            output_tokenized_text = self.tokenizer(text=output_template,
                                                   padding='max_length', max_length=50,
                                                   return_offsets_mapping=True)
            self.gen_data["output_input_ids"].append(torch.LongTensor(output_tokenized_text["input_ids"]))

            output_attention_mask = torch.Tensor(output_tokenized_text["attention_mask"])
            self.gen_data["output_attention_mask"].append(output_attention_mask)
            self.gen_data["data_type_mark"].append(self.data["data_type_mark"][i])
            neg_loss_mask = torch.zeros_like(output_attention_mask)
            for i in range(len(output_tokenized_text["offset_mapping"])):
                if (output_tokenized_text["offset_mapping"][i][0] >= offset_begin) and \
                    (output_tokenized_text["offset_mapping"][i][1] <= offset_end):
                    neg_loss_mask[i] = 1.0
            self.gen_data["negative_loss_mask"].append(neg_loss_mask)

    def __len__(self):
        return len(self.data["input_ids"])
    
    def __getitem__(self, index):
        item = dict()
        for k in self.data:
            item[k] = self.data[k][index]
        return item

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/hywen/BERT_CACHE/bart-base/")
    dataset = StanceData("data/VAST/vast_train.csv", tokenizer, wiki_path="data/VAST/wiki_dict.pkl")
    for x in dataset:
        print(x)
        break

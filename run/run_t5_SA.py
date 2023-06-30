# -*- coding: utf-8 -*-
"""t5_sst_test_map.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mpBINHhabKBorAJLkC0gd-HLFUAB3Mtr
"""


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import re
from tqdm import tqdm
import openpyxl
model_name = "michelecafagna26/t5-base-finetuned-sst2-sentiment"
model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/t5_SA")
tokenizer = AutoTokenizer.from_pretrained("/mnt/t5_SA")
model = model.to(torch.device("cuda"))

with open('./sst_test_map_syn.txt', 'r') as file:
  lines = file.readlines()
  for line in tqdm(lines):
    if line.startswith('sent_id') or line.startswith('insert') or line == '\n':
      continue
    texts = line.split('|')
    for i in range(0, len(texts)):
      texts[i] = re.sub(r'\s+', ' ', texts[i]).strip()
    sentiments = []
    scores = []
    labels = ["p", "n"]
    for text in texts:
      class_ids = torch.LongTensor(tokenizer(labels, padding=True).input_ids)
      class_ids = class_ids.to(torch.device("cuda"))
      inputs = tokenizer("sentiment: " + text, max_length=128, truncation=True, return_tensors="pt").to(torch.device("cuda"))
      preds = model.generate(inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            output_scores=True,
                            return_dict_in_generate=True,
                            min_length=class_ids.shape[1] + 1,
                            max_length=class_ids.shape[1] + 1,
                            do_sample=False)
      scores_tensor = torch.stack(preds.scores, dim=1)
      score_of_labels = scores_tensor.gather(dim=2, index=class_ids.T.expand(1, -1, -1))
      probabilities = score_of_labels.nanmean(dim=2).softmax(1)
      probabilities_list = []
      probabilities_list.append(probabilities[0][0].item())
      probabilities_list.append(probabilities[0][1].item())
      scores.append(probabilities_list)
      decoded_preds = tokenizer.batch_decode(sequences=preds[0], skip_special_tokens=True)
      if decoded_preds[0] == 'p':
        sentiments.append(1)
      else:
        sentiments.append(0)
    workbook = openpyxl.load_workbook('sst_test_map_t5.xlsx')
    sheet = workbook.active
    data = []
    for i in range(0, len(texts)):
      data.append(texts[i])
      data.append(sentiments[i])
      data.append(scores[i][0])
      data.append(scores[i][1])
    sheet.append(data)
    workbook.save('sst_test_map_t5.xlsx')
    workbook.close()
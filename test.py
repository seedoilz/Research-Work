#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import spacy
from tqdm import tqdm


# In[2]:


def find_insert_text(str1, str2):
    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    i = 0
    j = 0
    res = ''
    for j in range(len(str2_list)):
        if str1_list[i] != str2_list[j]:
            res += str2_list[j] + ' '
        else:
            i += 1
        if i == len(str1_list):
            break
    return res


# In[5]:


file_path = '/Users/seedoilz/Downloads/data/句意理解/result.xlsx'
df = pd.read_excel(file_path)


# In[6]:


df_copy = df.copy()
df_copy = df_copy.iloc[0:0]
entity_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
for entity_kind in entity_list:
    df[entity_kind] = 0
    df[entity_kind + '_text'] = 'blank'
    df_copy[entity_kind] = 0
    df_copy[entity_kind + '_text'] = 'blank'
df['insert_text'] = ''
df_copy['insert_text'] = ''


# In[ ]:


nlp = spacy.load("en_core_web_lg")
for index, row in tqdm(df.iterrows()):
    judge_text = find_insert_text(df['q1'], df['q2'])
    row['insert_text'] = judge_text
    doc = nlp(judge_text)
    if len(doc.ents) > 0:
        for ent in doc.ents:
            row[ent.label_] += 1
            if row[ent.label_] == 1:
                row[ent.label_ + '_text'] = ent.text
            else:
                row[ent.label_ + '_text'] += ',' + ent.text
        df_copy = pd.concat([df_copy, row.to_frame().T], axis=0)
df_copy.to_csv('res' + file_path.replace('xlsx','csv'), index=False)


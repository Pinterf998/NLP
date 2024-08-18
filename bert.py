#%%
import transformers
import torch
from transformers import BertTokenizer
from IPython.display import clear_output


PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", clean_up_tokenization_spaces=True)
#clean_up_tokenization_spaces=False這句是要不要清除多餘的空格

#%%
clear_output()
print("hello")
print("PyTorch 版本：", torch.__version__)

# %%
vocab=tokenizer.vocab
#vocab是一個字典{字體:對應ID}
print("字典大小:",len(vocab))
# %%
import random
random_tokens = random.sample(list(vocab), 10)
#將vocab字典先轉換成list，此時印出vocab會是["這","是","舉","例",[....],]
#而random.sample的用法就是於這list中隨機抽出10個當作新輸出

random_ids = [vocab[t] for t in random_tokens]
#將這十個字放進vocab轉換成對應的ID

print("{0:20}{1:15}".format("token", "index"))
print("-" * 25)

for t, id in zip(random_tokens, random_ids):
    print("{0:15}{1:10}".format(t, id))
    
# %%
list1=["大","家","好","我","是","黃","品","綸"]
id = [vocab[i] for i in list1]
print("{0:10}{1:20}".format("token","對應ID"))
print("-"*20)
for x,y in zip(list1,id):
    print("{0:5}{1:10}".format(x,y))

# %%
#接著開始斷句
sentence="[CLS]你們好，我是你們的老師"
tokens = tokenizer.tokenize(sentence)
#tokenizer.tokenize(句子)將句子切為一個個token
tokens_id=tokenizer.convert_tokens_to_ids(tokens)
#轉換token成id
print(tokens[:20])
print(tokens_id[:20])
# %%
from transformers import BertForMaskedLM
#%%

# %%

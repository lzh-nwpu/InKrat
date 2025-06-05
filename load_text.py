from transformers import AutoTokenizer, AutoModel
import torch
import pickle

def load_llm(localpath: str):
    llm_tokenizer = AutoTokenizer.from_pretrained(localpath)
    llm_model = AutoModel.from_pretrained(localpath)
    return llm_model, llm_tokenizer

llm_model, llm_tokenizer = load_llm('./bioLongformer')
device = 'cuda:0'

text_root_path_48 = '/home/lzh/MultimodalMIMIC/Data/ihm/root_text.pkl'
with open(text_root_path_48,'rb') as f:
    text_root = pickle.load(f)

out_text_embedding_list = []
for key, value in text_root.items():
    inputs = llm_tokenizer(value, padding=True, max_length=1024, truncation=True, return_tensors='pt',
                           return_attention_mask=True)
    llm_model.to(device)
    with torch.no_grad():
        text_embedding = llm_model(input_ids=inputs['input_ids'].to(device),
                               attention_mask=inputs['attention_mask'].to(device))
    out_text_embedding_list.append(torch.mean(text_embedding.pooler_output, dim=0))

out_tensor = torch.cat(out_text_embedding_list, dim=0)
# text_list = data['text']
#     out_text_embedding_list = []
#     for text in text_list:
#         text = [item for sublist in text for item in sublist]
#         inputs = llm_tokenizer(text, padding=True, max_length=1024, truncation=True, return_tensors='pt',
#                                     return_attention_mask=True)
#         text_embedding = llm_model(input_ids=inputs['input_ids'].to(self.device),
#                                         attention_mask=inputs['attention_mask'].to(self.device))
#
#         out_text_embedding_list.append(torch.mean(text_embedding.pooler_output, dim=0))
#         torch.cat(out_text_embedding_list, dim=0)

# process_text()
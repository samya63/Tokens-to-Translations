import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from Tokenizer import BPETokenizer

class TranslationDataset(Dataset):

    def __init__(self,split):
        self.dataset=load_dataset("bentrevett/multi30k")[split]
        print(f"Dataset loaded: {len(self.dataset)} samples.")
        self.src_tokenizer=BPETokenizer.load("tokenizers/en_tokenizer.pkl")
        self.tgt_tokenizer=BPETokenizer.load("tokenizers/german_tokenizer.pkl")
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        row=self.dataset[idx]
        src=[self.src_tokenizer.stoi["<SOS>"]]+self.src_tokenizer.encode(row["en"])+[self.src_tokenizer.stoi["<EOS>"]]
        tgt=[self.tgt_tokenizer.stoi["<SOS>"]]+self.tgt_tokenizer.encode(row["de"])+[self.tgt_tokenizer.stoi["<EOS>"]]
        return src,tgt
    
    @property
    def src_vocab_size(self):
        return self.src_tokenizer.vocab_size
    
    @property
    def tgt_vocab_size(self):
        return self.tgt_tokenizer.vocab_size
    
    
def collate_fn(batch):
    src=[row[0] for row in batch]
    tgt=[row[1] for row in batch]
    src_mxln=len(max(src,key=lambda x:len(x)))
    tgt_mxln=len(max(tgt,key=lambda x:len(x)))
    out_src=[]
    out_tgt=[]
    for s in src:
        out_src.append(s+[0]*(src_mxln-len(s)))
    for t in tgt:
        out_tgt.append(t+[0]*(tgt_mxln-len(t)))
    return torch.tensor(out_src,dtype=torch.long),torch.tensor(out_tgt,dtype=torch.long)



    






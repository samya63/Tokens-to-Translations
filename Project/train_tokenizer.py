from datasets import load_dataset
from Tokenizer import BPETokenizer

#hyperparameters
src_vocab_size=5000
tgt_vocab_size=5000

dataset=load_dataset("bentrevett/multi30k")
tr_data=dataset["train"]
src_corpus=[]
tgt_corpus=[]

for row in tr_data:
    src_corpus+=row["en"].split()
    tgt_corpus+=row["de"].split()

src_tokenizer=BPETokenizer(src_vocab_size)
tgt_tokenizer=BPETokenizer(tgt_vocab_size)

src_tokenizer.train(src_corpus)
tgt_tokenizer.train(tgt_corpus)

src_tokenizer.save("tokenizers/en_tokenizer.pkl")
tgt_tokenizer.save("tokenizers/german_tokenizer.pkl")


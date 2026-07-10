import pickle

class BPETokenizer:
    def __init__(self,vocab_size):
        self.vocab_size=vocab_size
        self.special_tokens={
            "<PAD>":0,#for padding sentences to desired context length 
            "<SOS>":1,#start of sentence
            "<EOS>":2,#end of sentence
            "<UNK>":3,
        }
        self.stoi=self.special_tokens.copy()
        self.itos={v:k for k,v in self.special_tokens.items()}
        self.merges=[]
        self.next_id=len(self.stoi)
        #self.merge_ranks={}

    def build_initial_vocab(self,corpus):
        vocab=sorted(list(set("".join(corpus)+" ")))
        for ch in vocab:
            self.stoi[ch]=self.next_id
            self.itos[self.next_id]=ch
            self.next_id+=1

    def convert_corpus(self,corpus):
        out=[]
        for word in corpus:
            out.append(list(word))
        return out
    
    def get_pair_frequencies(self,ccorpus):
        d={}
        for word in ccorpus:
            for j in range(len(word)-1):
                x=word[j]
                y=word[j+1]
                if (x,y) in d:
                    d[(x,y)]+=1
                else:
                    d[(x,y)]=1
        return d
    
    def merge_pair(self,ccorpus,pair):
        for i in range(len(ccorpus)):
            word=ccorpus[i]
            if len(word)==1:
                continue
            nword=[]
            j=0
            while j < len(word)-1:
                x=word[j]
                y=word[j+1]
                if (x,y)==pair:
                    nword.append(x+y)
                    j+=2
                else:
                    nword.append(x)
                    j+=1
            if j==len(word)-1:
                nword.append(word[len(word)-1])
            ccorpus[i]=nword
    
    def add_new_token(self,pair):
        self.stoi[pair[0]+pair[1]]=self.next_id
        self.itos[self.next_id]=pair[0]+pair[1]
        self.next_id+=1

    def get_most_frequent_pair(self,d):
        return max(d.items(),key=lambda x: x[1])[0]
    
    def train(self,corpus):
        self.build_initial_vocab(corpus)
        ccorpus=self.convert_corpus(corpus)

        #To track progress
        print(f"Starting training. Target vocab size: {self.vocab_size}")

        while self.next_id<self.vocab_size:
            d=self.get_pair_frequencies(ccorpus)
            if len(d)==0:
                break
            pair=self.get_most_frequent_pair(d)
            self.merge_pair(ccorpus,pair)
            self.add_new_token(pair)
            self.merges.append(pair)
            #to track progress
            if self.next_id % 100 == 0:  # Prints every 100 merges
                print(f"Progress: {self.next_id}/{self.vocab_size} tokens learned.")
        
        print("Training complete!")

    def encode(self,text):
        dc=[list(text)]
        for pair in self.merges:
            self.merge_pair(dc,pair)
        out=[]
        for tok in dc[0]:
            if tok in self.stoi:
                out.append(self.stoi[tok])
            else:
                out.append(self.stoi["<UNK>"])
        return out
    
    def decode(self,seq):
        out=""
        for i in seq:
            if i==self.stoi["<SOS>"]:
                continue
            if i==self.stoi["<PAD>"]:
                continue
            if i==self.stoi["<EOS>"]:
                break
            out+=self.itos[i]
        return out
    
    def save(self,filename):
        with open(filename,"wb") as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,filename):
        with open(filename,"rb") as f:
            return pickle.load(f)


            



    


        
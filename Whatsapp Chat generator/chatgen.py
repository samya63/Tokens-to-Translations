import re
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1345679087)

#cleaning the data and creating the dataset
with open('Chat1.txt','r',encoding='utf-8') as f:
    text=f.read()
text=re.sub(r"[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9], [0-9][0-9]:[0-9][0-9] -","",text)
text = text.encode("ascii", errors="ignore").decode("ascii")

#encoding functions
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda x: [ stoi[c] for c in x]#converts string to list of integers
decode=lambda x: ''.join([itos[c] for c in x])#receives list of integers and converts to string

#splitting data
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

#hyperparameters
batch_size=64
block_size=100#maximum context length
n_embd=64
hidden_size=256
max_iters=20000
learning_rate=3e-4
eval_iters=200
eval_interval=500

#batch getter
def get_batch(split):
    data=train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,)) # randomly select starting indices for our examples where examples are of the form of (x,y) where x is  string of length block_size and y is another string of sixe block_size shifted by one
    x=torch.stack([data[i:i+block_size] for i in ix]) #torch.stack converts list of tensors to a tensor 
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

class Gate(nn.Module):
    
    def __init__(self,hidden_size,n_embd):
        super().__init__()
        self.Wh=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Wx=nn.Linear(n_embd,hidden_size)
        nn.init.zeros_(self.Wx.bias)

    def forward(self,x,ht):
        out=torch.sigmoid(self.Wh(ht)+self.Wx(x))
        return out

        
class LSTMcell(nn.Module):

    def __init__(self,hidden_size,n_embd):
        super().__init__()
        self.hidden_size=hidden_size
        self.forget_gate=Gate(hidden_size,n_embd)
        nn.init.ones_(self.forget_gate.Wx.bias)
        self.input_gate=Gate(hidden_size,n_embd)
        self.output_gate=Gate(hidden_size,n_embd)
        self.Whc=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Wxc=nn.Linear(n_embd,hidden_size)
        nn.init.zeros_(self.Wxc.bias)

    def forward(self,x,ht=None,ct=None):
        B,T,C=x.shape
        ht=torch.zeros((B,self.hidden_size)) if ht is None else ht
        ct=torch.zeros((B,self.hidden_size)) if ct is None else ct
        out=[]
        for t in range(T):
            xt=x[:,t,:]
            f=self.forget_gate(xt,ht)
            i=self.input_gate(xt,ht)
            o=self.output_gate(xt,ht)
            cnd=torch.tanh(self.Whc(ht)+self.Wxc(xt))
            ct=f*ct+i*cnd
            ht=torch.tanh(ct)*o
            out.append(ht)
        return(torch.stack(out,dim=1),(ht,ct))

        

        

        
class textLSTM(nn.Module):

    def __init__(self,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,n_embd)
        self.LSTMlayer=LSTMcell(hidden_size,n_embd)
        self.proj=nn.Linear(hidden_size,vocab_size)
        nn.init.zeros_(self.proj.bias)

    def forward(self,x,ht =None,ct=None):
        out=self.embedding(x)
        out,(ht,ct)=self.LSTMlayer(out,ht,ct)
        out=self.proj(out)
        return out,(ht,ct)
    
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval() #turns the model into evaluation mode
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for j in range(eval_iters):
            X,Y=get_batch(split)
            logits,_=model(X)
            loss=F.cross_entropy(logits.view(-1,vocab_size),Y.view(-1))
            losses[j]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

def save_checkpoints(model,optimizer,step,filename="checkpoint.pth"):
    torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step
    }, filename)

def load_checkpoint(model,optimizer=None ,filename="checkpoint.pth"):
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]

def train(model,optimizer=None,start_step=0):
    #the optimizer
    if optimizer is None:
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    best_val = float('inf')

    for iter in range(start_step,start_step+max_iters):
        if iter % eval_interval==0:
            losses=estimate_loss()
            print(f"At {iter} step ,The training loss is {losses['train']}, The validation loss is {losses['val']}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoints(
                    model,
                    optimizer,
                    iter,
                    filename="best.pth"
                )
        if iter % 1000 == 0 and iter > 0:
            save_checkpoints(
                model,
                optimizer,
                iter,
                filename="latest.pth"
            )
        xb,yb=get_batch('train')
        logits,_=model(xb)
        loss=F.cross_entropy(logits.view(-1,vocab_size),yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def generate(model,prompt,max_new_tokens=300,temperature=1.0):
    x = torch.tensor([[stoi[prompt[0]]]], dtype=torch.long)
    logits,(ht,ct)=model(x)
    for c in prompt[1:]:
        logits,(ht,ct)=model(torch.tensor([[stoi[c]]], dtype=torch.long),ht,ct)
    out=prompt
    xt=prompt[-1]

    for j in range(max_new_tokens):
        x=torch.tensor([[stoi[xt]]], dtype=torch.long)
        logits,(ht,ct)=model(x,ht,ct)
        logits=logits[:,-1,:]
        probs=F.softmax(logits/temperature,dim=1)
        ix=torch.multinomial(probs, num_samples=1).item()
        xt=decode([ix])
        
        out+=xt
    return out


#instantiate the model
model= textLSTM(hidden_size)


#printting the parameter count
print("Number of parameters:",sum(p.numel() for p in model.parameters()))

import os

print("="*55)
print("        WhatsApp Character-Level LSTM")
print("="*55)
print(f"Vocabulary Size : {vocab_size}")
print(f"Parameters      : {sum(p.numel() for p in model.parameters())}")
print("="*55)

while True:

    print("\nMain Menu")
    print("1. Train from scratch")
    print("2. Resume training")
    print("3. Generate text")
    print("4. Exit")

    choice = input("\nChoice > ")

    if choice == "1":

        confirm = input(
            "\nThis will overwrite the current checkpoints.\nContinue? (y/n): "
        )

        if confirm.lower() != 'y':
            continue

        train(model)

        print("\nTraining complete!")

    elif choice == "2":

        if not os.path.exists("latest.pth"):
            print("\nNo latest checkpoint found.")
            continue

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

        step = load_checkpoint(
            model,
            optimizer,
            filename="latest.pth"
        )

        print(f"\nCheckpoint loaded from step {step}.")
        print("Resuming training...\n")

        train(
            model,
            optimizer=optimizer,
            start_step=step+1
        )

    elif choice == "3":

        if not os.path.exists("best.pth"):
            print("\nNo trained model found.")
            continue

        load_checkpoint(
            model,
            filename="best.pth"
        )

        model.eval()

        prompt = input("\nPrompt > ")

        length = int(
            input("Characters to generate > ")
        )

        temp = input(
            "Temperature (Press Enter for 1.0) > "
        )

        if temp == "":
            temp = 1.0
        else:
            temp = float(temp)

        print("\nGenerating...\n")

        print(
            generate(
                model,
                prompt,
                max_new_tokens=length,
                temperature=temp
            )
        )

    elif choice == "4":

        print("\nGoodbye!")
        break

    else:

        print("\nInvalid choice.")
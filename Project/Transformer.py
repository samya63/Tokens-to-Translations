import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self,max_len,d_model):
        super().__init__()
        self.max_len=max_len
        self.d_model=d_model
        self.register_buffer("pe",self.generate_pe(max_len))

    def forward(self,x):
        T=x.shape[1]
        if T>self.max_len:
            self.pe=self.generate_pe(max(T,self.max_len))
            self.max_len=T

        return x+self.pe[:T,:]
    
    def generate_pe(self,length):
        out=torch.zeros(length,self.d_model)
        for pos in range(length):
            for i in range(self.d_model):
                if i%2==0:
                    out[pos,i]=math.sin(pos/(10000**(i/self.d_model)))
                else:
                    out[pos,i]=math.cos(pos/(10000**(i/self.d_model)))
        return out
    
class InputEmbedding(nn.Module):
    def __init__(self,vocab_size,max_len,d_model):
        super().__init__()
        self.posenc=PositionalEncoding(max_len,d_model)
        self.emb=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        out=self.posenc(self.emb(x))
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads ):
        super().__init__()
        self.d_head=d_model//n_heads
        self.d_model=d_model
        self.n_heads=n_heads
        self.WQ=nn.ModuleList([nn.Linear(d_model,self.d_head) for _ in range(n_heads)])
        self.WK=nn.ModuleList([nn.Linear(d_model,self.d_head) for _ in range(n_heads)])
        self.WV=nn.ModuleList([nn.Linear(d_model,self.d_head) for _ in range(n_heads)])
        self.WO=nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask=None):
        Q=[WQ(query) for WQ in self.WQ]
        K=[WK(key) for WK in self.WK]
        V=[WV(value) for WV in self.WV]
        out=[]
        for i in range(self.n_heads):
            scores=Q[i] @ K[i].transpose(-2,-1)
            scores=scores/math.sqrt(self.d_head)
            if mask is not None:
                scores=scores.masked_fill(mask==0,float("-inf"))
            weights=torch.softmax(scores,dim=-1)
            headout=weights @ V[i]
            out.append(headout)
        output=torch.cat(out,dim=-1)
        output=self.WO(output)
        return output
    
class FeedFwd(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.lin1=nn.Linear(d_model,4*d_model)
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(4*d_model,d_model)

    def forward(self,x):
        return self.lin2(self.relu(self.lin1(x)))

        
class EncoderBlock(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.mha=MultiHeadAttention(d_model,n_heads)
        self.ffw=FeedFwd(d_model)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

    def forward(self,x):
        out=self.mha(x,x,x)
        x=self.norm1(x+out)
        out=self.ffw(x)
        x=self.norm2(x+out)
        return x
    
class Encoder(nn.Module):
    def __init__(self,d_model,n_heads,n_blocks):
        super().__init__()
        self.blocks=nn.ModuleList([EncoderBlock(d_model,n_heads) for _ in range(n_blocks)] )
        self.n_blocks=n_blocks

    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_head,max_len):
        super().__init__()
        self.mskd_sfatn=MultiHeadAttention(d_model,n_head)
        self.cros_atn=MultiHeadAttention(d_model,n_head)
        self.ffw=FeedFwd(d_model)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.max_len=max_len
        self.register_buffer("mask",self.create_mask(max_len))

    def forward(self,d,e):
        T=d.shape[-2]
        if T>self.max_len:
            self.mask=self.create_mask(max(T,self.max_len))
            self.max_len=T
        out=self.mskd_sfatn(d,d,d,self.mask[:T,:T])
        d=self.norm1(d+out)
        out=self.cros_atn(d,e,e)
        d=self.norm2(d+out)
        out=self.ffw(d)
        d=self.norm3(d+out)
        return d
    
    def create_mask(self,length):
        out=torch.tril(torch.ones(length,length))
        return out
    
class Decoder(nn.Module):
    def __init__(self,d_model,n_head,max_len,n_blocks):
        super().__init__()
        self.blocks=nn.ModuleList([DecoderBlock(d_model,n_head,max_len) for _ in range(n_blocks)])

    def forward(self,d,e):
        for block in self.blocks:
            d=block(d,e)
        return d

class Transformer(nn.Module):

    def __init__(self,d_model,n_head,max_len,n_blocks,src_vocab_size,tgt_vocab_size):
        super().__init__()
        self.enc=Encoder(d_model,n_head,n_blocks)
        self.dec=Decoder(d_model,n_head,max_len,n_blocks)
        self.inemb=InputEmbedding(src_vocab_size,max_len,d_model)
        self.oemb=InputEmbedding(tgt_vocab_size,max_len,d_model)
        self.proj=nn.Linear(d_model,tgt_vocab_size)
        self.d_model=d_model
        self.n_head=n_head
        self.n_blocks=n_blocks
        self.src_vocab_size=src_vocab_size
        self.tgt_vocab_size=tgt_vocab_size
        self.max_len=max_len

    def forward(self,src,tgt):
        d=tgt
        e=src
        d=self.oemb(d)
        e=self.inemb(e)
        e=self.enc(e)
        d=self.dec(d,e)
        logits=self.proj(d)
        return logits
        
    def generate(self,src,sos_id,eos_id,sampler):
        gen_lim=max(self.max_len,5*src.shape[1])
        tgt=torch.tensor([[sos_id]])
        while len(tgt)< gen_lim and tgt[-1]!=eos_id:
            logit=self.forward(src,tgt)[:,-1,:]
            next_token=sampler.sample(logit)
            tgt=torch.cat([tgt,next_token.unsqueeze(0)],dim=1)
        return tgt
    
    






    

    

    
        

    



    

        











    
    

        

    




    


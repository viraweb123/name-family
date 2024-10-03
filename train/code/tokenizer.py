import regex as re

#raw string
mystring="هوشمند بیرانوند   فروغی مظاهری زینلی معینی دولت رضاییان قادمه حیدرزاده کوچک سیاح حکیمی غلامزاده"
#g=[ord(x)for x in mystring]


#encoding raw txt with utf-8 encoding
tokens=list(mystring.encode("utf-8"))

#gets the statistic of which pair appear togather more frequently
def get_stats(ids):
    counts={}
    for pair in zip(ids,ids[1:]):  # (ids,ids[1:]) a way to make a silding window to comp 2 elements
        counts[pair]=counts.get(pair,0)+1
    return counts
stats=get_stats(tokens)
#print(stats)
#print(sorted(((v,k) for k,v in stats.items()),reverse=True))
top_pair=max(stats,key=stats.get)
#print(top_pair)

#replaces the most common pair with a new id index or idx
def merge(ids,pair,idx):
    newids=[]
    i=0
    while i <len(ids):
        if i <len(ids) - 1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids
#print(merge([2,4,578,5,2,4,12,2,4,1,3,2,4,63,256,2453,24,2,4],(2,4),69))
"""tokens2=merge(tokens,top_pair,128)
print(tokens2)
print("length: ",len(tokens2)) """

vocabsize=276
num_merges=vocabsize-256
ids=list(tokens) #so we still have a copy of the og list
merges={} # (int,int) -> int or (child1,child2 ) turning into a new token
for i in range(num_merges):
    stats=get_stats(ids)
    pair=max(stats,key=stats.get)
    idx=256+i
    print(f"merging {pair} into a new token {idx}")
    ids=merge(ids,pair,idx)
    merges[pair]=idx


print("token length: ",len(tokens))
print("ids length:",len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}X")


#decoding 

# pre processing variable
vocab={idx:bytes([idx]) for idx in range(256)}
for (p0,p1),idx in merges.items():
    vocab[idx]=vocab[p0]+vocab[p1] #addition of two bytes object kinda of a concatination
def decoding(ids):
    #given ids (list of ints) ,return python string\
    tokens= b"".join(vocab[idx] for idx in ids)
    text=tokens.decode("utf-8",errors='replace')
    return text

 #encoding segment

def encoding(text):
    tokens=list(text.encode("utf-8"))
    while len(tokens)>=2:
        stats=get_stats(tokens)
        pair=min(stats, key=lambda p:merges.get(p,float("inf")))
        if pair not in merges:
            break #nothing else is mergable
        idx=merges[pair]
        tokens=merge(tokens,pair,idx)
    return tokens
f=encoding('حسینی زاده')
print(f)
print(decoding(f))


##print(re.findall(gpt2pat,"heyo 123 123 I've come to you with big MASSIvE     news "))

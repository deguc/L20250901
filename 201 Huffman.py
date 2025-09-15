#%%
from heapq import *
from collections import Counter

class Huffman:

    def __init__(self,data):

        tree = [[w,[c,'']] for c,w in Counter(data).items()]
        heapify(tree)

        while len(tree) > 1:

            lo = heappop(tree)
            hi = heappop(tree)

            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]

            heappush(tree,[lo[0]+hi[0]]+lo[1:]+hi[1:])
        
        self.dic = heappop(tree)[1:]
    
    def encode(self,data):

        dic = {k:v for k,v in self.dic}
        encoded = ''

        for c in data:
            encoded += dic[c]
        
        return encoded
    
    def decode(self,encoded):

        dic = {k:v for v,k in self.dic}
        decoded = ''
        code = ''

        for c in encoded:
            
            code += c

            if code in dic:

                decoded += dic[code]
                code = ''
        
        return decoded


data = 'ABACABA'
huff = Huffman(data)
print(huff.dic)
encoded = huff.encode(data)
print(encoded)
decoded = huff.decode(encoded)
print(decoded)

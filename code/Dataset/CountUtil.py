import numpy as np

class CountSet:

    def __init__(self,maxv):
        self.maxv=maxv
        self.used = set([])
        self.notUsed = set([])

    def reset(self):
        for i in range(self.maxv):
            self.notUsed.add(i)
        self.used.clear()

    def fetch(self, index):
        if index in self.used:
            return False
        else:
            self.notUsed.remove(index)
            self.used.add(index)
            return True

    def mustFetch(self, index=0):
        if self.fetch(index):
            return index
        if not self.isEmpty():
            skipind = np.random.randint(0,len(self.notUsed))
            i=0
            for ind in self.notUsed:
                if i==skipind:
                    index=ind
                    break
                else:
                    i+=1
                    continue
            self.fetch(index)
            return index
                
        self.reset()
        self.fetch(index)
        return index
    
    def isEmpty(self):
        return len(self.notUsed)==0
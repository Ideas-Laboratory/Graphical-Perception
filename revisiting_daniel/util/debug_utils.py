import torch
import logging

def check(t,name=""):
    if isinstance(t,list):
        a=0
        for i in t:
            check(i,"%s.%d"%(name,a))
            a+=1
    elif isinstance(t,dict):
        for k,v in t.items():
            check(v,"%s.%s"%(name,k))
    elif not torch.is_tensor(t):
        return
    elif torch.isnan(t).any():
        logging.info("Detect NaN at %s"%name)
        raise Exception("Detect NaN %s!"%name)
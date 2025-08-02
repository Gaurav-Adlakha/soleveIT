from fastcore.utils import L,first, last, groupby, mul, add, sub, truediv, patch, store_attr, BasicRepr
from aocd import get_data
import pandas as pd
import fastcore.all as fc, re, math, itertools, functools, numpy as np, types, typing, dataclasses, matplotlib.pyplot as plt,collections,regex 
from regex import search 
from collections import Counter 
from collections.abc import Iterable 
np.set_printoptions(linewidth=150)
plt.rcParams['figure.dpi'] = 50
np.set_printoptions(precision=2, suppress=True)

import torch, numpy as np, pandas as pd
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)

# import torch
# torch.set_printoptions(precision=2, sci_mode=False)

# Add your new functions/modifications here

# for plot
from ipywidgets import interact
from fastai.basics import *

plt.rc('figure', dpi=90)

def plot_function(f, title=None, min=-2.1, max=2.1, color='r', ylim=None):
    x = torch.linspace(min,max, 100)[:,None]
    if ylim: plt.ylim(ylim)
    plt.plot(x, f(x), color)
    if title is not None: plt.title(title)

def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)

##

def intro():
    print("Note to solveit: Hi, I am Gaurav, and I am working with you today! When writing code for me, please follow fast.ai coding style: avoid comments; place single line if/loop/etc constructs on the same line; use concise variable names where appropriate. Follow the style you will see in my messages. As we work on this dialogue together.")

def show(m):
    print('', '0123456789')
    for i,o in enumerate(m): print(str(i)+(''.join(o)))

def get_inp (year=None, day=None):
    inp = get_data (year=year, day=day)
    ilines = inp.splitlines()
    print (len(ilines),'lines')
    print(len(ilines [0]), 'chars', len(ilines [0].split()), 'tokens')
    print (inp[:50], 'inp') 
    return inp

def array(x): return x if isinstance(x, np.ndarray) else np.array(list(x) if isinstance(x, Iterable) else x)

def pretty_df(df, rows=10, precision=3):
    "Create a styled HTML table with better formatting"
    styles = [
        dict(selector="th", props=[("font-weight", "bold"), 
                                  ("background-color", "#4CAF50"),
                                  ("color", "white"),
                                  ("padding", "10px"),
                                  ("text-align", "left")]),
        dict(selector="td", props=[("padding", "8px"),
                                  ("border-bottom", "1px solid #ddd")]),
        dict(selector="tr:nth-child(even)", props=[("background-color", "#f2f2f2")]),
        dict(selector="tr:hover", props=[("background-color", "#ddd")])
    ]

    return (df.head(rows)
            .style.set_table_attributes('class="dataframe" style="width:100%; border-collapse: collapse"')
            .format(precision=precision)
            .set_table_styles(styles))


def enable_pretty_display():
    import pandas as pd
    from utils import pretty_df
    
    global original_repr_html
    original_repr_html = pd.DataFrame._repr_html_
    
    def custom_repr_html(self): return pretty_df(self)._repr_html_()
    
    pd.DataFrame._repr_html_ = custom_repr_html
    return "Pretty display enabled for all DataFrames"

def disable_pretty_display():
    import pandas as pd
    global original_repr_html
    
    if 'original_repr_html' in globals(): 
        pd.DataFrame._repr_html_ = original_repr_html
        return "Restored original DataFrame display"
    return "Original display not found"
    
enable_pretty_display()
import os
import sys
import pandas as pd
import numpy as np

from scripts.experiments import ModelEvaluator

m_lst = [2, 3, 4]
# = ["modelAvg", "modelSel", "Interval-based"] #
ev = ModelEvaluator('base', cluster=True)

for m_id in m_lst:
    for sub_id in ev.exp.sub_lst_all:
        ev.get_loss(m_id, sub_id, save=True)
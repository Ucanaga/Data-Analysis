# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:02:43 2026

@author:
"""

import pathlib as pl
import pandas as pd

readfilesdpq = list(pl.Path(".").glob("DPQ_*.xpt"))
readfilesdemo = list(pl.Path(".").glob("DEMO_*.xpt"))

for file in readfilesdpq:
    globals()[file.stem] = pd.read_sas(file)
    
for file in readfilesdemo:
    globals()[file.stem] = pd.read_sas(file)
    
for nameyear in readfilesdpq:
    year = nameyear.stem.replace("DPQ_", "")
    
    dpq_data = globals()[f"DPQ_{year}"]
    demo_data = globals()[f"DEMO_{year}"]
    
    globals()[f"MERGED_{year}"] = dpq_data.merge(
        demo_data,
        on="SEQN",
        how="left"
    )

dpq_cols = [
    "DPQ010", "DPQ020", "DPQ030",
    "DPQ040", "DPQ050", "DPQ060",
    "DPQ070", "DPQ080", "DPQ090"]

merged_names = [
    name for name in globals()
    if name.startswith("MERGED_")]

missing_counts = []

for name in merged_names:
    before = len(globals()[name])
    
    globals()[name] = globals()[name].dropna(subset=dpq_cols)
    
    after = len(globals()[name])
    
    missing_counts.append({
        "dataset": name,
        "before": before,
        "removed_missing_dpq": before - after,
        "after": after})

"""
data_DPQ = pd.read_sas("DPQ_D.XPT")
data_DPQ = data_DPQ.fillna(-10)
data_DEMO = pd.read_sas("DEMO_D.XPT")
data_DPQ = data_DPQ.replace(5.397605346934028e-79, 0)


merged_data = data_DPQ.merge(data_DEMO, on="SEQN")

merged_data["sum"] = merged_data[
    ["DPQ010",
     "DPQ020",
     "DPQ030",
     "DPQ040",
     "DPQ050",
     "DPQ060",
     "DPQ070",
     "DPQ080",
     "DPQ090"]].sum(axis=1)

calculated = merged_data.copy()
calculated["sv_band"] = pd.cut(calculated["sum"],
                               bins=[-1, 4, 9, 14, 19, 27],
                               labels=[1, 2, 3, 4, 5])


data_with_sv_bands = calculated.copy()


output = data_with_sv_bands.head()
"""

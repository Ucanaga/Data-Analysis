# -*- coding: utf-8 -*-

import pathlib as pl
from time import perf_counter

import numpy as np
import pandas as pd

dpq_cols = [
    "DPQ010", "DPQ020", "DPQ030",
    "DPQ040", "DPQ050", "DPQ060",
    "DPQ070", "DPQ080", "DPQ090"
]

severity_bins = [-1, 4, 9, 14, 19, 27]
severity_labels = [1, 2, 3, 4, 5]

candidate_sample_sizes = [10, 25, 50, 100, 250, 500]

repeat_count = 1000

random_seed = 42

start_total = perf_counter()

years = sorted([
    file.stem.replace("DPQ_", "")
    for file in pl.Path(".").glob("DPQ_*.xpt")
])

print("Cycles Found:")
print(years)
    
merged_datasets = {}
missing_counts = []

for year in years:
    start_year = perf_counter()

    dpq_path = f"DPQ_{year}.xpt"
    demo_path = f"DEMO_{year}.xpt"

    print(f"\nProccesing: {year}")

    dpq_data = pd.read_sas(dpq_path)
    demo_data = pd.read_sas(demo_path)

    merged_data = dpq_data.merge(
        demo_data,
        on="SEQN",
        how="left"
    )

    merged_data = merged_data.replace(5.39761e-79, 0)

    before_dropna = len(merged_data)

    merged_data = merged_data.dropna(subset=dpq_cols)

    after_dropna = len(merged_data)

    missing_counts.append({
        "year": year,
        "before_dropna": before_dropna,
        "removed_missing_dpq": before_dropna - after_dropna,
        "after_dropna": after_dropna
    })

    merged_data[dpq_cols] = merged_data[dpq_cols].astype(int)

    merged_data["dpq_sum"] = merged_data[dpq_cols].sum(axis=1)

    merged_data["severity_band"] = pd.cut(
        merged_data["dpq_sum"],
        bins=severity_bins,
        labels=severity_labels
    )

    merged_data = merged_data.dropna(subset=["severity_band"])

    merged_data["severity_band"] = merged_data["severity_band"].astype(int)

    merged_data["profile"] = (
        merged_data[dpq_cols]
        .astype(str)
        .agg("".join, axis=1)
    )

    merged_datasets[f"MERGED_{year}"] = merged_data

    end_year = perf_counter()
    print(f"{year} completed. Time: {end_year - start_year:.3f} second.")

missing_counts = pd.DataFrame(missing_counts)

print("\nNumber of participants removed due to missing DPQ responses:")
print(missing_counts)


all_data = pd.concat(
    merged_datasets.values(),
    ignore_index=True
)

print("\nMerged data dimention:")
print(all_data.shape)


entropy_results = []

for band, band_data in all_data.groupby("severity_band"):
    profile_counts = band_data["profile"].value_counts()

    probabilities = profile_counts / profile_counts.sum()

    shannon_h = -np.sum(probabilities * np.log(probabilities))

    effective_profiles = np.exp(shannon_h)

    entropy_results.append({
        "severity_band": band,
        "n_people": len(band_data),
        "observed_unique_profiles": profile_counts.size,
        "shannon_entropy": shannon_h,
        "effective_number_of_profiles": effective_profiles
    })
    
entropy_results = pd.DataFrame(entropy_results)

print("\nShannon entropy results:")
print(entropy_results)


band_sizes = all_data.groupby("severity_band").size()

print("\nNumber of participants in each band:")
print(band_sizes)

chosen_sample_size = None

for sample_size in candidate_sample_sizes:
    enough_band_count = (band_sizes >= sample_size).sum()

    if enough_band_count >= 4:
        chosen_sample_size = sample_size

print("\nSelected rarefaction sample size:")
print(chosen_sample_size)

if chosen_sample_size is None:
    raise ValueError("There are not enough participants in at least 4 bands. Rarefaction cannot be performed..")

rng = np.random.default_rng(random_seed)

rarefaction_results = []

for band, band_data in all_data.groupby("severity_band"):
    n_people = len(band_data)

    if n_people < chosen_sample_size:
        rarefaction_results.append({
            "severity_band": band,
            "n_people": n_people,
            "sample_size": chosen_sample_size,
            "mean_unique_profiles": np.nan,
            "ci_2_5": np.nan,
            "ci_97_5": np.nan,
            "status": "not_estimable"
        })

        continue

    profile_codes, _ = pd.factorize(band_data["profile"])

    m = len(profile_codes)

    random_scores = rng.random((repeat_count, m))

    sampled_indices = np.argpartition(
        random_scores,
        chosen_sample_size - 1,
        axis=1
    )[:, :chosen_sample_size]

    sampled_profiles = profile_codes[sampled_indices]

    sorted_profiles = np.sort(sampled_profiles, axis=1)

    unique_counts = 1 + np.sum(
        sorted_profiles[:, 1:] != sorted_profiles[:, :-1],
        axis=1
    )

    rarefaction_results.append({
        "severity_band": band,
        "n_people": n_people,
        "sample_size": chosen_sample_size,
        "mean_unique_profiles": unique_counts.mean(),
        "ci_2_5": np.percentile(unique_counts, 2.5),
        "ci_97_5": np.percentile(unique_counts, 97.5),
        "status": "ok"
    })

rarefaction_results = pd.DataFrame(rarefaction_results)

print("\nRarefaction results:")
print(rarefaction_results)


profile_frequency_tables = []

for band, band_data in all_data.groupby("severity_band"):
    counts = band_data["profile"].value_counts().reset_index()
    counts.columns = ["profile", "count"]
    counts["severity_band"] = band

    profile_frequency_tables.append(counts)

profile_frequencies = pd.concat(
    profile_frequency_tables,
    ignore_index=True
)

print("\nFirst 10 rows of profile frequencies:")
print(profile_frequencies.head(10))


missing_counts.to_csv("missing_counts.csv", index=False)
entropy_results.to_csv("entropy_results.csv", index=False)
rarefaction_results.to_csv("rarefaction_results.csv", index=False)
profile_frequencies.to_csv("profile_frequencies.csv", index=False)

all_data.to_csv("all_cleaned_merged_data.csv", index=False)

end_total = perf_counter()

print("\nAll analyses completed.")
print(f"Total time: {end_total - start_total:.3f} seconds")
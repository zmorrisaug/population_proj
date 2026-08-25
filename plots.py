import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
pd.options.mode.chained_assignment = None

split = 2010
COVID = 2018
extension_years = 0

model=pd.read_csv("model.csv")
abx_m=pd.read_csv("abx_m.csv")
abx_f=pd.read_csv("abx_f.csv")
kt_m=pd.read_csv("kt_m.csv")
kt_f=pd.read_csv("kt_f.csv")
kt_m_noise=pd.read_csv("kt_m_noise.csv")
kt_f_noise=pd.read_csv("kt_f_noise.csv")

model["E1x"] = (model["E1x_m"] + model["E1x_f"])/2
model["E2x"] = (model["E2x_m"] + model["E2x_f"])/2
model["Error_impr"] = np.abs(model["E1x"]) - np.abs(model["E2x"])

E1x_pivot = model.pivot(index='age', columns='year_start', values="E1x")
E2x_pivot = model.pivot(index='age', columns='year_start', values="E2x")
Eix_pivot = model.pivot(index='age', columns='year_start', values="Error_impr")

# errors by age
E1_mean = model[model["year_start"]>=split].groupby("age")[["age","E1x"]].transform("mean")[model["year_start"]==split]
E1_std = model[model["year_start"]>=split].groupby("age")[["age","E1x"]].transform("std")[model["year_start"]==split]
plt.figure(figsize=(6,4))
plt.plot(E1_mean["age"], E1_mean["E1x"], color="green")
plt.fill_between(
    E1_mean["age"],
    E1_mean["E1x"] - 2*E1_std["E1x"],
    E1_mean["E1x"] + 2*E1_std["E1x"],
    alpha=0.3,
    color="green"
)
plt.xlabel("age")
plt.ylabel("Error")
plt.title("Error distribution by age for 1st order approximation")
plt.legend(["mean","95% confidence interval"])
# errors by age
E2_mean = model[model["year_start"]>=split].groupby("age")[["age","E2x"]].transform("mean")[model["year_start"]==split]
E2_std = model[model["year_start"]>=split].groupby("age")[["age","E2x"]].transform("std")[model["year_start"]==split]
plt.figure(figsize=(6,4))
plt.plot(E2_mean["age"], E2_mean["E2x"], color="#00ff00")
plt.fill_between(
    E2_mean["age"],
    E2_mean["E2x"] - 2*E2_std["E2x"],
    E2_mean["E2x"] + 2*E2_std["E2x"],
    alpha=0.3,
    color="#00ff00"
)
plt.xlabel("age")
plt.ylabel("Error")
plt.title("Error distribution by age for 2nd order approximation")
plt.legend(["mean","95% confidence interval"])

# errors by year
E1_mean = model[model["year_start"]>=split].groupby("year_start")[["year_start","E1x"]].transform("mean")[model["age"]==0]
E1_std = model[model["year_start"]>=split].groupby("year_start")[["year_start","E1x"]].transform("std")[model["age"]==0]
plt.figure(figsize=(6,4))
plt.plot(E1_mean["year_start"], E1_mean["E1x"], color="green")
plt.fill_between(
    E1_mean["year_start"],
    E1_mean["E1x"] - 2*E1_std["E1x"],
    E1_mean["E1x"] + 2*E1_std["E1x"],
    alpha=0.3,
    color="green"
)
plt.xlabel("start year")
plt.ylabel("Error")
plt.title("Error distribution by start year for 1st order approximation")
plt.legend(["mean","95% confidence interval"])

# errors by year
E2_mean = model[model["year_start"]>=split].groupby("year_start")[["year_start","E2x"]].transform("mean")[model["age"]==0]
E2_std = model[model["year_start"]>=split].groupby("year_start")[["year_start","E2x"]].transform("std")[model["age"]==0]
plt.figure(figsize=(6,4))
plt.plot(E2_mean["year_start"], E2_mean["E2x"], color="#00ff00")
plt.fill_between(
    E2_mean["year_start"],
    E2_mean["E2x"] - 2*E2_std["E2x"],
    E2_mean["E2x"] + 2*E2_std["E2x"],
    alpha=0.3,
    color="#00ff00"
)
plt.xlabel("start year")
plt.ylabel("Error")
plt.title("Error distribution by start year for 2nd order approximation")
plt.legend(["mean","95% confidence interval"])

#ax b1x b2x m
plt.figure(figsize=(4,4))
plt.plot(abx_m["age"], abx_m["ax_m"], color="blue")
plt.fill_between(
    abx_m["age"],
    abx_m["ax_m"] + kt_m["k1t_m"].min()*abx_m["b1x_m"],
    abx_m["ax_m"] + kt_m["k1t_m"].max()*abx_m["b1x_m"],
    alpha=0.2,
    color="blue"
)
plt.fill_between(
    abx_m["age"],
    abx_m["ax_m"] + kt_m["k1t_m"].min()*abx_m["b1x_m"] + kt_m["k1t_m"].min()*abx_m["b1x_m"],
    abx_m["ax_m"] + kt_m["k2t_m"].max()*abx_m["b2x_m"] + kt_m["k2t_m"].max()*abx_m["b2x_m"],
    alpha=0.1,
    color="blue"
)
plt.xlabel("age")
plt.ylabel("ax")
plt.title("Age distribution of male mortality")
plt.legend(["average mortality","1st order fluctuations","2nd order fluctuations"])
#ax b1x b2x f
plt.figure(figsize=(4,4))
plt.plot(abx_f["age"], abx_f["ax_f"], color="red")
plt.fill_between(
    abx_f["age"],
    abx_f["ax_f"] + kt_f["k1t_f"].min()*abx_f["b1x_f"],
    abx_f["ax_f"] + kt_f["k1t_f"].max()*abx_f["b1x_f"],
    alpha=0.2,
    color="red"
)
plt.fill_between(
    abx_f["age"],
    abx_f["ax_f"] + kt_f["k1t_f"].min()*abx_f["b1x_f"] + kt_f["k1t_f"].min()*abx_f["b1x_f"],
    abx_f["ax_f"] + kt_f["k2t_f"].max()*abx_f["b2x_f"] + kt_f["k2t_f"].max()*abx_f["b2x_f"],
    alpha=0.1,
    color="red"
)
plt.xlabel("age")
plt.ylabel("ax")
plt.title("Age distribution of female mortality")
plt.legend(["average mortality","1st order fluctuations","2nd order fluctuations"])

#k1tm
plt.figure(figsize=(4,4))
plt.plot(kt_m_noise["year_start"],kt_m_noise["k1t_m"],color="blue")
plt.plot(kt_m["year_start"],kt_m["k1t_m"],color="blue",linestyle=":")
std = (kt_m_noise["k1t_m"] - kt_m["k1t_m"]).std()
plt.fill_between(
    kt_m[kt_m["year_start"]>=2010]["year_start"],
    kt_m[kt_m["year_start"]>=2010]["k1t_m"] - 2*std,
    kt_m[kt_m["year_start"]>=2010]["k1t_m"] + 2*std,
    alpha=0.3,
    color="blue"
)
plt.axvline(x=split-0.5, color='black', linestyle='--', linewidth=1)
plt.text(
    split-0.5+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID-0.5, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID-0.5+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
plt.xlabel("start year")
plt.ylabel("k1t_m")
plt.title("Predictions of k1t_m")
plt.legend(["fit values","modelled values","95% confidence interval"])
#k1tf
plt.figure(figsize=(4,4))
plt.plot(kt_f_noise["year_start"],kt_f_noise["k1t_f"],color="red")
plt.plot(kt_f["year_start"],kt_f["k1t_f"],color="red",linestyle=":")
std = (kt_f_noise["k1t_f"] - kt_f["k1t_f"]).std()
plt.fill_between(
    kt_f[kt_f["year_start"]>=2010]["year_start"],
    kt_f[kt_f["year_start"]>=2010]["k1t_f"] - 2*std,
    kt_f[kt_f["year_start"]>=2010]["k1t_f"] + 2*std,
    alpha=0.3,
    color="red"
)
plt.axvline(x=split-0.5, color='black', linestyle='--', linewidth=1)
plt.text(
    split-0.5+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID-0.5, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID-0.5+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
plt.xlabel("start year")
plt.ylabel("k1t_f")
plt.title("Predictions of k1t_f")
plt.legend(["fit values","modelled values","95% confidence interval"])

#k2tm
plt.figure(figsize=(4,4))
plt.plot(kt_m_noise["year_start"],kt_m_noise["k2t_m"],color="blue")
plt.plot(kt_m["year_start"],kt_m["k2t_m"],color="blue",linestyle=":")
std = (kt_m_noise["k2t_m"] - kt_m["k2t_m"]).std()
plt.fill_between(
    kt_m[kt_m["year_start"]>=2010]["year_start"],
    kt_m[kt_m["year_start"]>=2010]["k2t_m"] - 2*std,
    kt_m[kt_m["year_start"]>=2010]["k2t_m"] + 2*std,
    alpha=0.3,
    color="blue"
)
plt.axvline(x=split-0.5, color='black', linestyle='--', linewidth=1)
plt.text(
    split-0.5+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID-0.5, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID-0.5+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
plt.xlabel("start year")
plt.ylabel("k2t_m")
plt.title("Predictions of k2t_m")
plt.legend(["fit values","modelled values","95% confidence interval"])
#k2tf
plt.figure(figsize=(4,4))
plt.plot(kt_f_noise["year_start"],kt_f_noise["k2t_f"],color="red")
plt.plot(kt_f["year_start"],kt_f["k2t_f"],color="red",linestyle=":")
std = (kt_f_noise["k2t_f"] - kt_f["k2t_f"]).std()
plt.fill_between(
    kt_f[kt_f["year_start"]>=2010]["year_start"],
    kt_f[kt_f["year_start"]>=2010]["k2t_f"] - 2*std,
    kt_f[kt_f["year_start"]>=2010]["k2t_f"] + 2*std,
    alpha=0.3,
    color="red"
)
plt.axvline(x=split-0.5, color='black', linestyle='--', linewidth=1)
plt.text(
    split-0.5+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID-0.5, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID-0.5+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
plt.xlabel("start year")
plt.ylabel("k2t_f")
plt.title("Predictions of k2t_f")
plt.legend(["fit values","modelled values","95% confidence interval"])





#E1
plt.figure(figsize=(8,6))
plt.imshow(E1x_pivot, extent = [1980,2022+extension_years,0,100], aspect=0.5, cmap="RdBu", vmin=-0.5, vmax=0.5, origin="lower")
plt.colorbar(label="residual error of log mortality")
plt.xlabel("start year")
plt.ylabel("age")
plt.title("Heatmap of errors in fitting 1st order Lee-Carter")
plt.axvline(x=split, color='black', linestyle='--', linewidth=1)
plt.text(
    split+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
#E2
plt.figure(figsize=(8,6))
plt.imshow(E2x_pivot, extent = [1980,2022+extension_years,0,100], aspect=0.5, cmap="RdBu", vmin=-0.5, vmax=0.5, origin="lower")
plt.colorbar(label="residual error of log mortality")
plt.xlabel("start year")
plt.ylabel("age")
plt.title("Heatmap of errors in fitting 2nd order Lee-Carter")
plt.axvline(x=split, color='black', linestyle='--', linewidth=1)
plt.text(
    split+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
#Ei

rwg = LinearSegmentedColormap.from_list(
    "rwg",
    ["red", "white", "green"]
)

plt.figure(figsize=(8,6))
plt.imshow(Eix_pivot, extent = [1980,2022+extension_years,0,100], aspect=0.5, cmap=rwg, vmin=-0.5, vmax=0.5, origin="lower")
plt.colorbar(label="error difference")
plt.xlabel("start year")
plt.ylabel("age")
plt.title("Heatmap of error difference in Lee-Carter models")
plt.axvline(x=split, color='black', linestyle='--', linewidth=1)
plt.text(
    split+0.2,      # x position
    60,          # y position
    'Train/test split',
    rotation=90,
    va='top'
)
plt.axvline(x=COVID, color='black', linestyle=':', linewidth=2)
plt.text(
    COVID+0.2,      # x position
    30,          # y position
    'COVID',
    rotation=90,
    va='top'
)
#Difference

plt.show()

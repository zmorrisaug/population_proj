import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

data = pd.read_csv("clean.csv")

new_rows = []
split = 2010
extension_years = 0

for year in range(2023, 2023+extension_years):
    for age in range(0, 101):
        new_rows.append({
            "year_start": year,
            "age": age
        })
new_rows_df = pd.DataFrame(new_rows)
data = pd.concat([data, new_rows_df], ignore_index=True)

training_data = data[data["year_start"]<split]

training_data["log_mx_m"] = np.log(training_data["mx_m"])
training_data["log_mx_f"] = np.log(training_data["mx_f"])

training_data["ax_m"] = training_data.groupby("age")["log_mx_m"].transform("mean")
training_data["ax_f"] = training_data.groupby("age")["log_mx_f"].transform("mean")

training_data["Z1x_m"] = training_data["log_mx_m"] - training_data["ax_m"]
training_data["Z1x_f"] = training_data["log_mx_f"] - training_data["ax_f"]

#1m
Z1_m_pivot = training_data.pivot(index='age', columns='year_start', values='Z1x_m')
Z1_m = Z1_m_pivot.to_numpy()
b1x_m, s, k1t_m = np.linalg.svd(Z1_m, full_matrices=False)

b1x_m = pd.DataFrame({
    "age": Z1_m_pivot.index,
    "b1x_m": b1x_m[:, 0]
})

k1t_m = pd.DataFrame({
    "year_start": Z1_m_pivot.columns,
    "k1t_m": s[0] * k1t_m[0, :]
})

training_data = training_data.merge(b1x_m, on="age", how="left")
training_data = training_data.merge(k1t_m, on="year_start", how="left")

#1f
Z1_f_pivot = training_data.pivot(index='age', columns='year_start', values='Z1x_f')
Z1_f = Z1_f_pivot.to_numpy()
b1x_f, s, k1t_f = np.linalg.svd(Z1_f, full_matrices=False)

b1x_f = pd.DataFrame({
    "age": Z1_f_pivot.index,
    "b1x_f": b1x_f[:, 0]
})

k1t_f = pd.DataFrame({
    "year_start": Z1_f_pivot.columns,
    "k1t_f": s[0] * k1t_f[0, :]
})

training_data = training_data.merge(b1x_f, on="age", how="left")
training_data = training_data.merge(k1t_f, on="year_start", how="left")
#

training_data["Z2x_m"] = training_data["log_mx_m"] - training_data["ax_m"] - training_data["b1x_m"] * training_data["k1t_m"]
training_data["Z2x_f"] = training_data["log_mx_f"] - training_data["ax_f"] - training_data["b1x_f"] * training_data["k1t_f"]
#2m
Z2_m_pivot = training_data.pivot(index='age', columns='year_start', values='Z2x_m')
Z2_m = Z2_m_pivot.to_numpy()
b2x_m, s, k2t_m = np.linalg.svd(Z2_m, full_matrices=False)

b2x_m = pd.DataFrame({
    "age": Z2_m_pivot.index,
    "b2x_m": b2x_m[:, 0]
})

k2t_m = pd.DataFrame({
    "year_start": Z2_m_pivot.columns,
    "k2t_m": s[0] * k2t_m[0, :]
})

training_data = training_data.merge(b2x_m, on="age", how="left")
training_data = training_data.merge(k2t_m, on="year_start", how="left")

#2f
Z2_f_pivot = training_data.pivot(index='age', columns='year_start', values='Z2x_f')
Z2_f = Z2_f_pivot.to_numpy()
b2x_f, s, k2t_f = np.linalg.svd(Z2_f, full_matrices=False)

b2x_f = pd.DataFrame({
    "age": Z2_f_pivot.index,
    "b2x_f": b2x_f[:, 0]
})

k2t_f = pd.DataFrame({
    "year_start": Z2_f_pivot.columns,
    "k2t_f": s[0] * k2t_f[0, :]
})

training_data = training_data.merge(b2x_f, on="age", how="left")
training_data = training_data.merge(k2t_f, on="year_start", how="left")

# Extend k

n = split - 1980

#m
ax_m = training_data[training_data["year_start"]==1980][["age","ax_m"]]
b1x_m = training_data[training_data["year_start"]==1980][["age","b1x_m"]]
k1t_m = training_data[training_data["age"]==0][["year_start","k1t_m"]]
b2x_m = training_data[training_data["year_start"]==1980][["age","b2x_m"]]
k2t_m = training_data[training_data["age"]==0][["year_start","k2t_m"]]

k1t_m.merge(k2t_m, how = "left").to_csv("kt_m_noise.csv", index=False)

gradient = (n*(k1t_m["year_start"]*k1t_m["k1t_m"]).sum()
                - k1t_m["year_start"].sum()*k1t_m["k1t_m"].sum()) / (
                    (n*(k1t_m["year_start"]*k1t_m["year_start"]).sum()
                - k1t_m["year_start"].sum()*k1t_m["year_start"].sum()))
intercept = (k1t_m["k1t_m"].sum()-gradient*k1t_m["year_start"].sum())/n
rows = []
for t in range(1980,2023+extension_years):
    rows.append({
        "year_start": t,
        "k1t_m": gradient*t+intercept
    })
k1t_m = pd.DataFrame(rows)

k2t_m["year_squared"] = (k2t_m["year_start"]-(split-1+1980)/2)**2

gradient = (n*(k2t_m["year_squared"]*k2t_m["k2t_m"]).sum()
                - k2t_m["year_squared"].sum()*k2t_m["k2t_m"].sum()) / (
                    (n*(k2t_m["year_squared"]*k2t_m["year_squared"]).sum()
                - k2t_m["year_squared"].sum()*k2t_m["year_squared"].sum()))
intercept = (k2t_m["k2t_m"].sum()-gradient*k2t_m["year_squared"].sum())/n
rows = []
for t in range(1980,2023+extension_years):
    rows.append({
        "year_start": t,
        "k2t_m": gradient*(t-(split-1+1980)/2)**2+intercept
    })
k2t_m = pd.DataFrame(rows)

#f
ax_f = training_data[training_data["year_start"]==1980][["age","ax_f"]]
b1x_f = training_data[training_data["year_start"]==1980][["age","b1x_f"]]
k1t_f = training_data[training_data["age"]==0][["year_start","k1t_f"]]
b2x_f = training_data[training_data["year_start"]==1980][["age","b2x_f"]]
k2t_f = training_data[training_data["age"]==0][["year_start","k2t_f"]]

k1t_f.merge(k2t_f, how = "left").to_csv("kt_f_noise.csv", index=False)

gradient = (n*(k1t_f["year_start"]*k1t_f["k1t_f"]).sum()
                - k1t_f["year_start"].sum()*k1t_f["k1t_f"].sum()) / (
                    (n*(k1t_f["year_start"]*k1t_f["year_start"]).sum()
                - k1t_f["year_start"].sum()*k1t_f["year_start"].sum()))
intercept = (k1t_f["k1t_f"].sum()-gradient*k1t_f["year_start"].sum())/n
rows = []
for t in range(1980,2023+extension_years):
    rows.append({
        "year_start": t,
        "k1t_f": gradient*t+intercept
    })
k1t_f = pd.DataFrame(rows)

k2t_f["year_squared"] = (k2t_f["year_start"]-(split-1+1980)/2)**2

gradient = (n*(k2t_f["year_squared"]*k2t_f["k2t_f"]).sum()
                - k2t_f["year_squared"].sum()*k2t_f["k2t_f"].sum()) / (
                    (n*(k2t_f["year_squared"]*k2t_f["year_squared"]).sum()
                - k2t_f["year_squared"].sum()*k2t_f["year_squared"].sum()))
intercept = (k2t_f["k2t_f"].sum()-gradient*k2t_f["year_squared"].sum())/n
rows = []
for t in range(1980,2023+extension_years):
    rows.append({
        "year_start": t,
        "k2t_f": gradient*(t-(split-1+1980)/2)**2+intercept
    })
k2t_f = pd.DataFrame(rows)

#

abx_m = ax_m.merge(b1x_m, how="left").merge(b2x_m, how="left")
kt_m = k1t_m.merge(k2t_m, how="left")
abk_m = kt_m.merge(abx_m, how="cross")

abx_f = ax_f.merge(b1x_f, how="left").merge(b2x_f, how="left")
kt_f = k1t_f.merge(k2t_f, how="left")
abk_f = kt_f.merge(abx_f, how="cross")


model = data.merge(abk_m, how="left", on=["age", "year_start"]).merge(abk_f, how="left", on=["age", "year_start"])

model["log_mx_m"] = np.log(model["mx_m"])
model["log_M1x_m"] = model["ax_m"] + model["b1x_m"] * model["k1t_m"]
model["log_M2x_m"] = model["log_M1x_m"] + model["b2x_m"] * model["k2t_m"]

model["log_mx_f"] = np.log(model["mx_f"])
model["log_M1x_f"] = model["ax_f"] + model["b1x_f"] * model["k1t_f"]
model["log_M2x_f"] = model["log_M1x_f"] + model["b2x_f"] * model["k2t_f"]

model["E1x_m"] = model["log_M1x_m"] - model["log_mx_m"]
model["E2x_m"] = model["log_M2x_m"] - model["log_mx_m"]
model["E1x_f"] = model["log_M1x_f"] - model["log_mx_f"]
model["E2x_f"] = model["log_M2x_f"] - model["log_mx_f"]

model.to_csv("model.csv", index=False)
abx_m.to_csv("abx_m.csv", index=False)
abx_f.to_csv("abx_f.csv", index=False)
kt_m.to_csv("kt_m.csv", index=False)
kt_f.to_csv("kt_f.csv", index=False)





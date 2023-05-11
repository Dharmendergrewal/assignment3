import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wbgapi as wb
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Preprocessing data and returning original, processed data


def data(x, y, z):
    data = wb.data.DataFrame(x, mrv=y)
    data1 = pd.DataFrame(data.sum(), columns=z)
    return data, data1.rename_axis("year")

# Converting Year column to integers


def string(x):
    x["year"] = [int(i.split("YR")[1]) for i in x.index]
    return x


# GDP data
indicator = ["NY.GDP.PCAP.KD", "EN.ATM.CO2E.PC"]
data_GDP_O, data_GDP_R = data(indicator[0], 30, ["GDP"])
new_data_GDP = string(data_GDP_R)

# CO2_mt data
data_CO2_mt_O, data_CO2_mt_R = data(indicator[1], 30, ["CO2_mt"])
new_data_CO2_mt = string(data_CO2_mt_R)

# Computes exponential function with scale and growth free parameters


def exp_growth(t, scale, growth):
    return scale * np.exp(growth * (t - 1990))


# Curve fit for GDP
popr, pcov = curve_fit(
    exp_growth, new_data_GDP["year"], new_data_GDP["GDP"])

# Plotting graph between GDP data and curve_fit
plt.figure()
plt.plot(
    new_data_GDP["year"],
    new_data_GDP["GDP"],
    label="GDP")
plt.plot(
    new_data_GDP["year"],
    exp_growth(
        new_data_GDP["year"],
        *popr),
    label="fit")
plt.legend()
plt.title("Curve Fit and data line of GDP Per Capita")
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    This routine can be used in assignment programs.
    """
    import itertools as iter

    # initiate arrays for lower and upper limits

    lower = func(x, *param)
    upper = lower

    uplow = []  # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


# Calculating the upper and lower limits for the function, parameters and
# sigmas
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(new_data_GDP["year"], exp_growth, popr, sigma)

# Plotting graph between confidence ranges and fit data
plt.figure()
plt.title("exp_growth function")
plt.plot(
    new_data_GDP["year"],
    new_data_GDP["GDP"],
    label="GDP Per Capita")
plt.plot(
    new_data_GDP["year"],
    exp_growth(
        new_data_GDP["year"],
        *popr),
    label="fit")
plt.fill_between(new_data_GDP["year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()

# Curve fit for CO2_mt
popr2, pcov2 = curve_fit(
    exp_growth, new_data_CO2_mt["year"], new_data_CO2_mt['CO2_mt'])

# Plotting graph between CO2_mt data and curve_fit
plt.figure()
plt.plot(new_data_CO2_mt["year"], new_data_CO2_mt["CO2_mt"], label="CO2_mt")
plt.plot(
    new_data_CO2_mt["year"],
    exp_growth(
        new_data_CO2_mt["year"],
        *popr2),
    label="fit")
plt.legend()
plt.title("Curve fit and data line of CO2 Metric Ton Per Capita")
plt.xlabel("year")
plt.ylabel("CO2_mt")
plt.show()

# Calculating upper and lower limits for the function, parameters and sigmas
sigma2 = np.sqrt(np.diag(pcov2))
low2, up2 = err_ranges(new_data_CO2_mt["year"], exp_growth, popr2, sigma2)

# Plotting graph between confidence ranges and fit data
plt.figure()
plt.title("exp_growth function")
plt.plot(new_data_CO2_mt["year"], new_data_CO2_mt["CO2_mt"], label="CO2_mt")
plt.plot(
    new_data_CO2_mt["year"],
    exp_growth(
        new_data_CO2_mt["year"],
        *popr2),
    label="fit")
plt.fill_between(new_data_CO2_mt["year"], low2, up2, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("CO2_mt")
plt.show()

# prepossing data for clustering
data_GDP = pd.DataFrame(data_GDP_O.iloc[:, -1])
data_CO2_mt = pd.DataFrame(data_CO2_mt_O.iloc[:, -1])
data_GDP.columns = ["GDP"]
data_CO2_mt.columns = ["CO2_mt"]
data_GDP["CO2_mt"] = data_CO2_mt["CO2_mt"]
data_GDP_C = data_GDP.rename_axis("countries")
final_data = data_GDP_C.dropna()

# Visualizing data with Scatter plot
fig = plt.figure(figsize=(8, 6))
sns.scatterplot(data=final_data,
                x="CO2_mt", y="GDP",
                color='green')
plt.title("scatter plot before clustering")

# plotting scatter plot for kmeans clustering
X = final_data[['CO2_mt', 'GDP']].copy()
kmeanModel = KMeans(n_clusters=3)
identified = kmeanModel.fit_predict(final_data[['CO2_mt', 'GDP']])
cluster_centers = kmeanModel.cluster_centers_
u_labels = np.unique(identified)
clusters_with_data = final_data[['CO2_mt', 'GDP']].copy()
clusters_with_data['Clusters'] = identified
fig = plt.figure(figsize=(10, 8))

# ploting data points
plt.scatter(clusters_with_data['CO2_mt'], clusters_with_data['GDP'],
            c=clusters_with_data['Clusters'], cmap='viridis')

# ploting center points
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, alpha=0.8)
plt.title("Scatter plot after clusters")
plt.xlabel('CO2_mt')
plt.ylabel('GDP Per Capita')
plt.show()

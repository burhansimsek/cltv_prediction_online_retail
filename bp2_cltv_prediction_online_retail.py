####################################################################################################
# Dataset Info
# Invoice: transaction id, unique but multiple, if contains "C" this is canceled transaction.
# StockCode: product stock code, unique
# Description: product description
# Quantity: number of product sold
# InvoiceDate: invoice date
# Price: product price
# Customer ID: customer id, unique
# Country: customer country


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("M3_crm_analytics/my_codes/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.isnull().sum()
df.describe().T

# data preprocessing
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]
replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")
df.describe().T

df["TotalPrice"] = df["Price"] * df["Quantity"]

# bg/nbd model
# recency
# T
# frequency
# monetary

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

cltv = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                      lambda x: (today_date - x.min()).days],
                                      "Invoice": lambda x: x.nunique(),
                                      "TotalPrice": lambda x: x.sum()})

cltv.columns = ["recency", "T", "frequency", "monetary"]

cltv["recency"] = cltv["recency"] / 7
cltv["T"] = cltv["T"] / 7
cltv = cltv[cltv["frequency"] > 1]
cltv["monetary"] = cltv["monetary"] / cltv["frequency"]

cltv.describe().T

# bg/nbd model

bgf = BetaGeoFitter(penalizer_coef=0.01)

bgf.fit(cltv["frequency"],
        cltv["recency"],
        cltv["T"])

cltv["expected_purc_1_week"] = bgf.predict(1,
                                           cltv['frequency'],
                                           cltv['recency'],
                                           cltv['T'])

cltv["expected_purc_1_month"] = bgf.predict(4,
                                            cltv['frequency'],
                                            cltv['recency'],
                                            cltv['T'])

cltv["expected_purc_3_month"] = bgf.predict(12,
                                            cltv['frequency'],
                                            cltv['recency'],
                                            cltv['T'])

# Gamma Gamma

ggf = GammaGammaFitter(penalizer_coef=0.001)

ggf.fit(cltv["frequency"],
        cltv["monetary"])

cltv["expected_avg_profit"] = ggf.conditional_expected_average_profit(cltv["frequency"],
                                                                      cltv["monetary"])

cltv["CLTV"] = ggf.customer_lifetime_value(bgf,
                                           cltv["frequency"],
                                           cltv["recency"],
                                           cltv["T"],
                                           cltv["monetary"],
                                           time=6,
                                           discount_rate=0.01)

cltv.describe().T

cltv.sort_values("CLTV", ascending=False).head(20)

cltv["segment"] = pd.qcut(cltv["CLTV"], 4, labels=["D", "C", "B", "A"])

cltv.groupby("segment").agg("mean")

########################################################################

cltv["CLTV_1_Month"] = ggf.customer_lifetime_value(bgf,
                                                   cltv["frequency"],
                                                   cltv["recency"],
                                                   cltv["T"],
                                                   cltv["monetary"],
                                                   time=1,
                                                   discount_rate=0.01)

cltv["CLTV_12_Month"] = ggf.customer_lifetime_value(bgf,
                                                    cltv["frequency"],
                                                    cltv["recency"],
                                                    cltv["T"],
                                                    cltv["monetary"],
                                                    time=12,
                                                    discount_rate=0.01)


cltv.sort_values("CLTV_1_Month", ascending=False).head(10)
cltv.sort_values("CLTV_12_Month", ascending=False).head(10)
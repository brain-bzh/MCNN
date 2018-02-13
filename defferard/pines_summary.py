
import pandas as pd
import numpy as np
df = pd.read_csv("tests.csv",index_col=0)
grouped_mean = df.groupby(["type"]).aggregate([np.mean])
print(grouped_mean.sort_values([("last_acc","mean")],ascending=False)["last_acc"])

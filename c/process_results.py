import pandas as pd
import numpy as np

lines = open('exec.txt', 'r').readlines()

lines = [line.strip().split(" ") for line in lines]
# Read in the data
df = pd.DataFrame(lines, columns=['mag', 'temp'])
# Reindex by temp
df['mag'] = df['mag'].astype(float)
df['temp'] = df['temp'].astype(float)
df = df.groupby('temp').mean()
# df = df.set_index('temp')
print(df)
df.plot().get_figure().savefig('exec.png')
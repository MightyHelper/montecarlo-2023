import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lines = open('exec.txt', 'r').readlines()

line_parts = [line.strip().split(" ") for line in lines]
lines = [(mag, temp, size) for _, mag, temp, size in line_parts]
# Read in the data
df = pd.DataFrame(lines, columns=['mag', 'temp', 'size'])
# Reindex by temp
df['mag'] = df['mag'].astype(float, errors='ignore')
df['temp'] = df['temp'].astype(float, errors='ignore')
df['size'] = df['size'].astype(int, errors='ignore')

df = df.groupby(['size', 'temp']).mean()
# Sort by size
df = df.sort_index(level='size')
# df = df.set_index('temp')
print(df.unstack('size'))

# df.plot().get_figure().savefig('exec.png')
# Separate plot for each size
plt.clf()
sizes = df.index.get_level_values('size').unique()
print(sizes)
for size in sizes:
    # Drop size and plot
    # df.loc[df.index.get_level_values('size') == size].droplevel('size')
    mysize = df.loc[df.index.get_level_values('size') == size].droplevel('size')
    plt.plot(mysize.index, mysize['mag'])
    plt.scatter(mysize.index, mysize['mag'], marker='x', alpha=0.1)
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
leg = [(x,None) for x in sizes]
# flatten
leg = [item for sublist in leg for item in sublist]
plt.legend(leg)
plt.savefig(f"exec.png")

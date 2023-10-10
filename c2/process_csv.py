import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('multi_size.pbc.2.txt', header=None, names=['iter', 'size', 'temp', 'magnetization', 'energy'])

df['magnetization'] = df['magnetization'] / (df['size'] ** 2)
df['energy'] = df['energy'] / (df['size'] ** 2)
# print(df.groupby(['temp', 'size', 'iter']).agg(['std']).unstack('temp').to_string())

## 3-d plot by size, iter => (mean energy); color by temp

dfz = df.groupby(['temp', 'size', 'iter']).agg(['mean'])
## undo (magnetization, mean) to magnetization_mean
dfz = dfz.reset_index()
dfz.columns = dfz.columns.map('_'.join)
dfz = dfz.rename(columns={'temp_': 'temp', 'size_': 'size', 'iter_': 'iter', 'magnetization_mean': 'magnetization',
                          'energy_mean': 'energy'})

print(dfz.dtypes)

df_t1 = dfz[dfz['temp'] == 1.0]
df_t4 = dfz[dfz['temp'] == 4.0]


def plot_data(ax, my_df, temp_str, colors2use):
    ax.scatter(my_df['size'], np.log(my_df['iter'] + 1), my_df['energy'], label='energy T=' + temp_str, c=colors2use[0],
               s=1)
    ax.scatter(my_df['size'], np.log(my_df['iter'] + 1), my_df['magnetization'], label='magnetization T=' + temp_str,
               c=colors2use[1], s=1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('size')
ax.set_ylabel('iter')
ax.set_zlabel('energy')
ax.set_title('Energy + Mag by size, log(iter)')
plot_data(ax, df_t1, "1.0", ('r', 'b'))
plot_data(ax, df_t4, "4.0", ('g', 'y'))
ax.legend()
plt.savefig('energy_by_size_iter_temp.png', dpi=1000)

# Print max iter

print("Max iter for T=1.0", df_t1['iter'].max())

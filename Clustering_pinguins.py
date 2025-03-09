import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()
penguins_df.info()

#Preprocessing and T-SNE to evaluate the numbers of clusters
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
penguins_df = pd.get_dummies(penguins_df, columns=['sex'], drop_first=True)

samples = penguins_df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
scaler = StandardScaler()
samples_scaled = scaler.fit_transform(samples)
model = TSNE(learning_rate = 100)

transformed = model.fit_transform(samples_scaled)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs,ys)
plt.show()

inertie = {}
for k in range(1,10):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(samples_scaled)
    iner = kmeans.inertia_
    inertie[k] = iner
print(inertie)

penguins_df['label'] = kmeans.fit_predict(samples_scaled)

stat_penguins = penguins_df.groupby('label')[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']].mean()

print(stat_penguins)
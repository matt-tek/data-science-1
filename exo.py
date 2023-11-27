import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class DataAnalysis:
    def __init__(self, accident_file, meteo_file, other_files):
        self.accident_data = pd.read_excel(accident_file)
        self.meteo_data = pd.read_csv(meteo_file)

    def association_gravite_meteo(self):
        try:
            combined_data = pd.merge(self.accident_data, self.meteo_data, on='ColonneCommune')
            contingency_table = pd.crosstab(combined_data['GraviteAccident'], combined_data['ConditionMeteo'])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            return chi2, p
        except Exception as e:
            return f"Error: {e}"

    def correlation_dommages_age(self):
        try:
            accidents_legers = self.accident_data[self.accident_data['Gravite'] == 'Leger']
            return stats.pearsonr(accidents_legers['AgeConducteur'], accidents_legers['ValeurDommages'])
        except Exception as e:
            return f"Error: {e}"

    def difference_dommages_sexe(self):
        try:
            dommages_hommes = self.accident_data[self.accident_data['Sexe'] == 'Homme']['ValeurDommages']
            dommages_femmes = self.accident_data[self.accident_data['Sexe'] == 'Femme']['ValeurDommages']
            return ttest_ind(dommages_hommes, dommages_femmes)
        except Exception as e:
            return f"Error: {e}"

    def serie_temporelle(self):
        try:
            self.accident_data['Date'] = pd.to_datetime(self.accident_data['Date'])
            return self.accident_data.groupby('Date').size()
        except Exception as e:
            return f"Error: {e}"

    def stationnarite_serie(self, time_serie):
        try:
            return adfuller(time_serie)
        except Exception as e:
            return f"Error: {e}"

    def saisonnalite_serie(self, time_serie):
        try:
            decomposition = seasonal_decompose(time_serie, model='additive', period=1)
            return decomposition
        except Exception as e:
            return f"Error: {e}"

    def segmentation_mrcs(self, data_file, n_clusters=3):
        try:
            data = pd.read_csv(data_file)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(data_scaled)
            data['Cluster'] = kmeans.labels_
            return data
        except Exception as e:
            return f"Error: {e}"

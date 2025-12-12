import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load the full CSV (replace with your file path or use io.StringIO for string)
df = pd.read_csv('master_fourier.csv')

# Filter for Bok Choy
bok_df = df[df['Plant_Type'] == 'Bok Choy']

# Add Period
bok_df.loc[bok_df['Frequency'] > 0, 'Period'] = 1 / bok_df['Frequency']

# Filter periods >10 hours
bok_df = bok_df[bok_df['Period'] > 10]

# Scatter plot: Magnitude vs. Period by Treatment
plt.figure(figsize=(12, 8))
sns.scatterplot(data=bok_df, x='Period', y='Magnitude', hue='Treatment', style='Replicate_Number', s=100)
plt.title('Magnitude vs. Period for Bok Choy: Control vs. Drought')
plt.xlabel('Period (hours)')
plt.ylabel('Magnitude')
plt.axvline(x=24, color='red', linestyle='--', label='Diurnal (~24h)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('bok_choy_magnitude_vs_period.png')
plt.show()

# Bar plot for average ~24h magnitude
diurnal_df = bok_df[(bok_df['Period'] >= 20) & (bok_df['Period'] <= 28)]
avg_diurnal = diurnal_df.groupby(['Treatment', 'Replicate_Number'])['Magnitude'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_diurnal, x='Treatment', y='Magnitude', hue='Replicate_Number')
plt.title('Average Magnitude at ~24h Period: Control vs. Drought')
plt.ylabel('Average Magnitude')
plt.tight_layout()
plt.savefig('bok_choy_avg_24h_bar.png')
plt.show()

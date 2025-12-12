import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data (replace 'master_fourier.csv' with your file path if needed)
# If loading from string, use: df = pd.read_csv(io.StringIO(csv_text))
df = pd.read_csv(r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\Master\master_fourier.csv")

# Clean and filter data (e.g., focus on periods > 10 hours for diurnal analysis)
df = df[df['Period'] > 10]  # Filter out high-frequency noise
df.sort_values(by=['Plant_Type', 'Treatment', 'Period'], inplace=True)

# Print unique conditions for reference
print("Unique Plants:", df['Plant_Type'].unique())
print("Unique Treatments:", df['Treatment'].unique())
print("Unique Temp/Hum Controls:", df['Temp_Hum_Control'].unique())
print("Unique Light Cycles:", df['Light_Cycle'].unique())

# Group by conditions (e.g., Plant_Type and Treatment)
grouped = df.groupby(['Plant_Type', 'Treatment'])

# Plot 1: Scatter plot of Magnitude vs. Period, colored by Treatment, faceted by Plant_Type
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Period', y='Magnitude', hue='Treatment', style='Light_Cycle', s=100, alpha=0.7)
plt.title('Magnitude vs. Period Across Treatments and Light Cycles')
plt.xlabel('Period (hours)')
plt.ylabel('Magnitude')
plt.axvline(x=24, color='red', linestyle='--', label='Diurnal (24h)')  # Highlight 24h line
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('magnitude_vs_period_scatter.png')  # Save plot
plt.show()

# Plot 2: Bar plot for top peaks per group (e.g., average magnitude at ~24h period)
# Filter for periods around 24h (e.g., 20-28h)
diurnal_df = df[(df['Period'] >= 20) & (df['Period'] <= 28)]
avg_diurnal = diurnal_df.groupby(['Plant_Type', 'Treatment', 'Temp_Hum_Control', 'Light_Cycle'])['Magnitude'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(data=avg_diurnal, x='Treatment', y='Magnitude', hue='Plant_Type', dodge=True)
plt.title('Average Magnitude at ~24h Period by Treatment and Plant')
plt.xlabel('Treatment')
plt.ylabel('Average Magnitude')
plt.legend(title='Plant Type')
plt.tight_layout()
plt.savefig('avg_magnitude_24h_bar.png')  # Save plot
plt.show()

# Plot 3: Overlaid line plots for Magnitude vs. Period by Treatment (for a specific plant, e.g., Bok Choy)
bok_choy_df = df[df['Plant_Type'] == 'Bok Choy']
plt.figure(figsize=(12, 8))
for treatment in bok_choy_df['Treatment'].unique():
    subset = bok_choy_df[bok_choy_df['Treatment'] == treatment]
    plt.plot(subset['Period'], subset['Magnitude'], marker='o', label=treatment)
plt.title('Magnitude vs. Period for Bok Choy by Treatment')
plt.xlabel('Period (hours)')
plt.ylabel('Magnitude')
plt.axvline(x=24, color='red', linestyle='--', label='Diurnal (24h)')
plt.legend()
plt.tight_layout()
plt.savefig('bok_choy_overlay.png')  # Save plot
plt.show()

# Optional: Box plot for statistical comparison of magnitudes across replicates
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Treatment', y='Magnitude', hue='Light_Cycle')
plt.title('Distribution of Peak Magnitudes by Treatment and Light Cycle')
plt.xlabel('Treatment')
plt.ylabel('Magnitude')
plt.legend(title='Light Cycle')
plt.tight_layout()
plt.savefig('magnitude_boxplot.png')  # Save plot
plt.show()

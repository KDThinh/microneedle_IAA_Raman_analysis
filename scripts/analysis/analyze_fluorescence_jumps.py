import pandas as pd
import numpy as np

# Read both CSV files
batch_csv = r"IAA Nanosensor Experiment\Bok Choy\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 1\Raw data\test_outputs_batch_20251122_162854\batch_results_summary.csv"
main_csv = r"IAA Nanosensor Experiment\Bok Choy\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 1\Raw data\processed_results\raman_analysis_results_emission_nm_2025-11-21.csv"

print("Loading CSV files...")
batch_df = pd.read_csv(batch_csv, parse_dates=['Datetime'], index_col='Datetime')
main_df = pd.read_csv(main_csv, parse_dates=['Datetime'], index_col='Datetime')

# Sort by datetime to ensure correct order
batch_df = batch_df.sort_index()
main_df = main_df.sort_index()

print(f"Batch results: {len(batch_df)} scans")
print(f"Main results: {len(main_df)} scans")

# Calculate differences for fluorescence
print("\n=== Calculating differences ===")
batch_df['Fluorescence_M1_diff'] = batch_df['Fluorescence (Method 1)'].diff()
batch_df['Fluorescence_M2_diff'] = batch_df['Fluorescence (Method 2)'].diff()
batch_df['Background_diff'] = batch_df['Background Average (250-1250 cm^-1)'].diff()

main_df['Final_Fluorescence_diff'] = main_df['Final SWNT Fluorescence'].diff()
main_df['Background_diff'] = main_df['Average Background Intensity'].diff()

# Calculate statistics
batch_fluo_m1_std = batch_df['Fluorescence_M1_diff'].std()
batch_fluo_m2_std = batch_df['Fluorescence_M2_diff'].std()
batch_bg_std = batch_df['Background_diff'].std()

main_fluo_std = main_df['Final_Fluorescence_diff'].std()
main_bg_std = main_df['Background_diff'].std()

# Calculate mean absolute differences for comparison
batch_fluo_m1_mean_abs = batch_df['Fluorescence_M1_diff'].abs().mean()
batch_fluo_m2_mean_abs = batch_df['Fluorescence_M2_diff'].abs().mean()
main_fluo_mean_abs = main_df['Final_Fluorescence_diff'].abs().mean()

print("\n=== Statistics Summary ===")
print("\nBatch Results (main_test.py):")
print(f"  Method 1 (Lieberfit baseline AUC):")
print(f"    Mean absolute change: {batch_fluo_m1_mean_abs:.2f}")
print(f"    Std of changes: {batch_fluo_m1_std:.2f}")
print(f"  Method 2 (Raw - G-band - Background):")
print(f"    Mean absolute change: {batch_fluo_m2_mean_abs:.2f}")
print(f"    Std of changes: {batch_fluo_m2_std:.2f}")
print(f"  Background:")
print(f"    Std of changes: {batch_bg_std:.2f}")

print("\nMain Results (main.py):")
print(f"  Final Fluorescence:")
print(f"    Mean absolute change: {main_fluo_mean_abs:.2f}")
print(f"    Std of changes: {main_fluo_std:.2f}")
print(f"  Background:")
print(f"    Std of changes: {main_bg_std:.2f}")

# Find sudden jumps (more than 3 standard deviations)
threshold_batch_m1 = 3 * batch_fluo_m1_std
threshold_batch_m2 = 3 * batch_fluo_m2_std
threshold_batch_bg = 3 * batch_bg_std
threshold_main_fluo = 3 * main_fluo_std
threshold_main_bg = 3 * main_bg_std

batch_m1_jumps = batch_df[abs(batch_df['Fluorescence_M1_diff']) > threshold_batch_m1]
batch_m2_jumps = batch_df[abs(batch_df['Fluorescence_M2_diff']) > threshold_batch_m2]
batch_bg_jumps = batch_df[abs(batch_df['Background_diff']) > threshold_batch_bg]

main_fluo_jumps = main_df[abs(main_df['Final_Fluorescence_diff']) > threshold_main_fluo]
main_bg_jumps = main_df[abs(main_df['Background_diff']) > threshold_main_bg]

print("\n=== Sudden Jumps Detection (3 std dev threshold) ===")
print(f"\nBatch Results:")
print(f"  Method 1 jumps: {len(batch_m1_jumps)} ({len(batch_m1_jumps)/len(batch_df)*100:.2f}%)")
print(f"  Method 2 jumps: {len(batch_m2_jumps)} ({len(batch_m2_jumps)/len(batch_df)*100:.2f}%)")
print(f"  Background jumps: {len(batch_bg_jumps)} ({len(batch_bg_jumps)/len(batch_df)*100:.2f}%)")

print(f"\nMain Results:")
print(f"  Final Fluorescence jumps: {len(main_fluo_jumps)} ({len(main_fluo_jumps)/len(main_df)*100:.2f}%)")
print(f"  Background jumps: {len(main_bg_jumps)} ({len(main_bg_jumps)/len(main_df)*100:.2f}%)")

# Check correlation between background jumps and fluorescence jumps
print("\n=== Correlation Analysis ===")
# For batch results Method 1
batch_bg_jump_indices = set(batch_bg_jumps.index)
batch_m1_jump_indices = set(batch_m1_jumps.index)
batch_m2_jump_indices = set(batch_m2_jumps.index)
overlap_batch_m1 = len(batch_bg_jump_indices & batch_m1_jump_indices)
overlap_batch_m2 = len(batch_bg_jump_indices & batch_m2_jump_indices)

print(f"Batch Method 1: Background jumps coinciding with Fluorescence jumps: {overlap_batch_m1}")
print(f"Batch Method 2: Background jumps coinciding with Fluorescence jumps: {overlap_batch_m2}")

# For main results
main_bg_jump_indices = set(main_bg_jumps.index)
main_fluo_jump_indices = set(main_fluo_jumps.index)
overlap_main = len(main_bg_jump_indices & main_fluo_jump_indices)
print(f"Main: Background jumps coinciding with Fluorescence jumps: {overlap_main}")

# Print specific examples
print("\n=== Top 10 Largest Jumps ===")
if len(batch_df) > 0:
    print("\nBatch Method 1 - Top 10 fluorescence jumps:")
    top_m1 = batch_df.nlargest(10, 'Fluorescence_M1_diff', keep='all')[['Fluorescence (Method 1)', 'Fluorescence_M1_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']]
    print(top_m1[['Fluorescence (Method 1)', 'Fluorescence_M1_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']])
    
    print("\nBatch Method 1 - Top 10 fluorescence drops:")
    bottom_m1 = batch_df.nsmallest(10, 'Fluorescence_M1_diff', keep='all')[['Fluorescence (Method 1)', 'Fluorescence_M1_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']]
    print(bottom_m1[['Fluorescence (Method 1)', 'Fluorescence_M1_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']])
    
    print("\nBatch Method 2 - Top 10 fluorescence jumps:")
    top_m2 = batch_df.nlargest(10, 'Fluorescence_M2_diff', keep='all')[['Fluorescence (Method 2)', 'Fluorescence_M2_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']]
    print(top_m2[['Fluorescence (Method 2)', 'Fluorescence_M2_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']])
    
    print("\nBatch Method 2 - Top 10 fluorescence drops:")
    bottom_m2 = batch_df.nsmallest(10, 'Fluorescence_M2_diff', keep='all')[['Fluorescence (Method 2)', 'Fluorescence_M2_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']]
    print(bottom_m2[['Fluorescence (Method 2)', 'Fluorescence_M2_diff', 'Background Average (250-1250 cm^-1)', 'Background_diff']])

if len(main_df) > 0:
    print("\nMain - Top 10 fluorescence jumps:")
    top_main = main_df.nlargest(10, 'Final_Fluorescence_diff', keep='all')[['Final SWNT Fluorescence', 'Final_Fluorescence_diff', 'Average Background Intensity', 'Background_diff']]
    print(top_main[['Final SWNT Fluorescence', 'Final_Fluorescence_diff', 'Average Background Intensity', 'Background_diff']])
    
    print("\nMain - Top 10 fluorescence drops:")
    bottom_main = main_df.nsmallest(10, 'Final_Fluorescence_diff', keep='all')[['Final SWNT Fluorescence', 'Final_Fluorescence_diff', 'Average Background Intensity', 'Background_diff']]
    print(bottom_main[['Final SWNT Fluorescence', 'Final_Fluorescence_diff', 'Average Background Intensity', 'Background_diff']])

# Calculate coefficient of variation (CV) to measure variability
print("\n=== Variability Analysis (Coefficient of Variation) ===")
batch_m1_cv = (batch_df['Fluorescence_M1_diff'].std() / batch_df['Fluorescence_M1_diff'].abs().mean()) * 100
batch_m2_cv = (batch_df['Fluorescence_M2_diff'].std() / batch_df['Fluorescence_M2_diff'].abs().mean()) * 100
main_fluo_cv = (main_df['Final_Fluorescence_diff'].std() / main_df['Final_Fluorescence_diff'].abs().mean()) * 100

print(f"Batch Method 1 CV: {batch_m1_cv:.2f}%")
print(f"Batch Method 2 CV: {batch_m2_cv:.2f}%")
print(f"Main Method CV: {main_fluo_cv:.2f}%")
print("\n(Lower CV = more stable, fewer sudden changes)")

# Summary comparison
print("\n" + "="*60)
print("SUMMARY: Which method has fewer sudden changes?")
print("="*60)
print(f"\nMethod 1 (Lieberfit baseline AUC):")
print(f"  - Mean absolute change: {batch_fluo_m1_mean_abs:.2f}")
print(f"  - CV: {batch_m1_cv:.2f}%")
print(f"  - Number of jumps (>3 std dev): {len(batch_m1_jumps)}")

print(f"\nMethod 2 (Raw - G-band - Background):")
print(f"  - Mean absolute change: {batch_fluo_m2_mean_abs:.2f}")
print(f"  - CV: {batch_m2_cv:.2f}%")
print(f"  - Number of jumps (>3 std dev): {len(batch_m2_jumps)}")

print(f"\nMain.py Method:")
print(f"  - Mean absolute change: {main_fluo_mean_abs:.2f}")
print(f"  - CV: {main_fluo_cv:.2f}%")
print(f"  - Number of jumps (>3 std dev): {len(main_fluo_jumps)}")


import os
import pandas as pd
import numpy as np
from .utils import create_dir_if_needed

def update_master_diurnal(master_csv_path, diurnal_df, processed_dir, plant_type, treatment, temp_hum_control, light_cycle, replicate_number, run_date):
    """
    Append diurnal cycle averages to the master diurnal CSV.
    
    Parameters:
    master_csv_path : str
        Path to the master diurnal CSV file.
    diurnal_df : pandas.DataFrame
        DataFrame with diurnal cycle data (from compute_diurnal_average).
    processed_dir : str
        Directory containing the diurnal cycle CSV.
    plant_type, treatment, temp_hum_control, light_cycle, replicate_number : str
        Metadata for the experiment.
    run_date : str
        Date of the experiment (YYYY-MM-DD).
    """
    if os.path.exists(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
    else:
        columns = [
            'Plant_Type', 'Treatment', 'Temp_Hum_Control', 'Light_Cycle', 'Replicate_Number', 'Run_Date',
            'Binned_Hour', 'Mean_PLG_Ratio', 'Std_PLG_Ratio', 'SEM_PLG_Ratio', 'Count', 'Data_Path'
        ]
        master_df = pd.DataFrame(columns=columns)

    diurnal_csv = os.path.join(processed_dir, f'corrected_raman_results_gaussian_{run_date}.csv')
    new_data = []
    for _, row in diurnal_df.groupby('Binned Hour')['Smoothed Corrected (ALS)'].agg(['mean', 'std', 'count']).reset_index().iterrows():
        new_data.append({
            'Plant_Type': plant_type,
            'Treatment': treatment,
            'Temp_Hum_Control': temp_hum_control,
            'Light_Cycle': light_cycle,
            'Replicate_Number': replicate_number,
            'Run_Date': run_date,
            'Binned_Hour': row['Binned Hour'],
            'Mean_PLG_Ratio': row['mean'],
            'Std_PLG_Ratio': row['std'],
            'SEM_PLG_Ratio': row['std'] / np.sqrt(row['count']) if row['count'] > 0 else 0,
            'Count': row['count'],
            'Data_Path': diurnal_csv
        })

    new_df = pd.DataFrame(new_data)
    master_df = pd.concat([master_df, new_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['Plant_Type', 'Treatment', 'Temp_Hum_Control', 'Light_Cycle', 'Replicate_Number', 'Run_Date', 'Binned_Hour'], keep='last')
    create_dir_if_needed(os.path.dirname(master_csv_path))
    master_df.to_csv(master_csv_path, index=False)
    print(f"Updated master diurnal CSV at {master_csv_path}")

def update_master_fourier(master_csv_path, complete_ft_df, processed_dir, plant_type, treatment, temp_hum_control, light_cycle, replicate_number, run_date):
    """
    Append all Fourier transform data (frequencies 0.0 to 0.2 cycles/hour) to the master Fourier CSV.
    
    Parameters:
    master_csv_path : str
        Path to the master Fourier CSV file.
    complete_ft_df : pandas.DataFrame
        DataFrame with complete Fourier transform data for smoothed signal (from compute_fourier_transform).
    processed_dir : str
        Directory containing the Fourier CSV.
    plant_type, treatment, temp_hum_control, light_cycle, replicate_number : str
        Metadata for the experiment.
    run_date : str
        Date of the experiment (YYYY-MM-DD).
    """
    if os.path.exists(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
    else:
        columns = [
            'Plant_Type', 'Treatment', 'Temp_Hum_Control', 'Light_Cycle', 'Replicate_Number', 'Run_Date',
            'Frequency', 'Real', 'Imag', 'Magnitude', 'Phase', 'Data_Path'
        ]
        master_df = pd.DataFrame(columns=columns)

    fourier_csv = os.path.join(processed_dir, f'complete_fft_smoothed_{run_date}.csv')
    new_data = []
    filtered_fft = complete_ft_df[(complete_ft_df['Frequency (cycles/hour)'] >= 0.0) & (complete_ft_df['Frequency (cycles/hour)'] <= 0.2)]
    print("Filtered complete_ft_df (0.0-0.2 cycles/hour):\n", filtered_fft)
    if filtered_fft.empty:
        print("Warning: No frequencies found between 0.0 and 0.2 cycles/hour. No data will be appended.")
        return

    for _, row in filtered_fft.iterrows():
        new_data.append({
            'Plant_Type': plant_type,
            'Treatment': treatment,  # Fixed: Use treatment instead of temp_hum_control
            'Temp_Hum_Control': temp_hum_control,
            'Light_Cycle': light_cycle,
            'Replicate_Number': replicate_number,
            'Run_Date': run_date,
            'Frequency': row['Frequency (cycles/hour)'],
            'Real': row['Real'],
            'Imag': row['Imag'],
            'Magnitude': row['Magnitude'],
            'Phase': row['Phase'],
            'Data_Path': fourier_csv
        })

    new_df = pd.DataFrame(new_data)
    master_df = pd.concat([master_df, new_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['Plant_Type', 'Treatment', 'Temp_Hum_Control', 'Light_Cycle', 'Replicate_Number', 'Run_Date', 'Frequency'], keep='last')
    create_dir_if_needed(os.path.dirname(master_csv_path))
    master_df.to_csv(master_csv_path, index=False)
    print(f"Updated master Fourier CSV at {master_csv_path} with {len(new_df)} frequency points")

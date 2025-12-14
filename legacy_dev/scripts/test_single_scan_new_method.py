"""
Test script for new processing method:
1. Filter wavenumber to 250 cm-1 onward
2. Smooth with Savitzky-Golay
4. Lieberfit applied to smoothed spectrum
5. Detect Raman peak (800-950 cm^-1) and G-band peak (1500-1750 cm^-1) on the Lieberfit-corrected spectrum, perform Lorentzian fitting
6. Calculate the peak area of the Raman peaks and G-band peak
7. For Lieberfit baseline, subtract the average values between 250 cm-1 and 1250 cm-1
8. Calculate fluorescence using three methods:
   - Method 1: AUC of background-subtracted Lieberfit baseline (>= 1250 cm-1)
   - Method 2: Raw AUC - G-band area - Background AUC (>= 1250 cm-1)
   - Method 3: Normalize spectrum first (by average intensity 250-1250 cm-1), then apply Method 1 & 2 workflows on normalized spectrum

Processing Steps:
-----------------
- Step 1: Filter wavenumber to 250 cm-1 onward
- Step 2: Smooth with Savitzky-Golay filter
- Step 4: Apply Lieberfit to smoothed spectrum (removes fluorescence baseline)
- Step 5: Detect Raman peak (800-950 cm^-1) and G-band peak (1500-1750 cm^-1) on Lieberfit-corrected spectrum using Lorentzian fitting
- Step 6: Calculate peak area using analytical formula (π × amplitude × HWHM for Lorentzian)
- Step 7: Subtract average of Lieberfit baseline (250-1250 cm-1) from the baseline
- Step 8: Calculate fluorescence using three different methods

Peak Fitting and Area Calculation:
-----------------------------------
- Lorentzian fitting is used for both Raman and G-band peaks (more appropriate than Gaussian)
- Lorentzian better describes natural line broadening in Raman spectroscopy
- Peaks are detected and fitted on the Lieberfit-corrected spectrum
- Peak area is calculated using analytical formula: Area = π × amplitude × HWHM
- This gives accurate peak area measurement for both peaks

Fluorescence Calculation:
--------------------------
- Method 1: The Lieberfit baseline represents the fluorescence signal. Average of baseline from 250-1250 cm-1 is subtracted to remove background. Fluorescence is calculated as AUC of the background-subtracted baseline (>= 1250 cm-1)
- Method 2: Raw AUC of smoothed spectrum (>= 1250 cm-1) minus G-band area minus background AUC
- Method 3: First normalize smoothed spectrum by dividing by average intensity in 250-1250 cm-1 window. Then apply Lieberfit to normalized spectrum, detect G-band peak, and calculate fluorescence using both Method 1 and Method 2 workflows on the normalized spectrum. Uses Method 1 equivalent as primary value.

Lieberfit Parameter Optimization:
---------------------------------
The Lieberfit algorithm has two key parameters:
- poly_order (polynomial order): Controls baseline flexibility
  * Lower order (3-5): Smoother baseline, may underfit complex backgrounds
  * Higher order (6-8): More flexible baseline, may overfit and remove signal
  
- tot_iter (iterations): Controls convergence
  * Fewer iterations (50-100): Faster, may not fully converge
  * More iterations (200-300): Better convergence, but diminishing returns

To optimize parameters:
  python test_single_scan_new_method.py --optimize

To use custom parameters:
  python test_single_scan_new_method.py --order 6 --iter 150

The optimization function tests multiple combinations and evaluates:
- Residual sum of squares (RSS): Lower is better
- Mean absolute error (MAE): Lower is better  
- Baseline smoothness: Lower is better
- Peak preservation: Higher is better
- Negative fraction: Lower is better (corrected spectrum should be positive)
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path (go up two levels from scripts/ to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.ingestion import load_raman_dataset
from pipeline.config_loader import load_profile_config
from pipeline.utils import create_dir_if_needed, lieberfit
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

def gaussian(x, amplitude, center, width, offset):
    """
    Gaussian function for peak fitting.
    
    Parameters:
    -----------
    width : float
        Standard deviation (sigma) of the Gaussian
    """
    return amplitude * np.exp(-0.5 * ((x - center) / width)**2) + offset

def lorentzian(x, amplitude, center, width, offset):
    """
    Lorentzian function for peak fitting.
    This is the preferred function for Raman peaks as it better describes natural line broadening.
    
    Parameters:
    -----------
    width : float
        Half-width at half-maximum (HWHM) of the Lorentzian
    """
    return amplitude / (1 + ((x - center) / width)**2) + offset

def calculate_peak_area_analytical(amplitude, width, peak_type='lorentzian'):
    """
    Calculate peak area using analytical formulas.
    
    For Lorentzian: Area = π * amplitude * width (where width is HWHM)
    For Gaussian: Area = amplitude * width * sqrt(2π) (where width is sigma)
    
    Parameters:
    -----------
    amplitude : float
        Peak amplitude
    width : float
        Peak width parameter (HWHM for Lorentzian, sigma for Gaussian)
    peak_type : str
        'lorentzian' or 'gaussian'
    
    Returns:
    --------
    area : float
        Analytical peak area
    """
    if peak_type == 'lorentzian':
        # Lorentzian: Area = π * amplitude * HWHM
        return np.pi * amplitude * width
    elif peak_type == 'gaussian':
        # Gaussian: Area = amplitude * sigma * sqrt(2π)
        return amplitude * width * np.sqrt(2 * np.pi)
    else:
        raise ValueError(f"Unknown peak_type: {peak_type}")

def find_peak_accurate(wavenumbers, intensities, wavenumber_min, wavenumber_max, 
                       peak_type='gaussian', prominence_factor=0.1, width_min=5, width_max=50,
                       fluorescence_baseline=None):
    """
    Accurately find peak position using peak detection and curve fitting.
    Uses multiple strategies to find the most reliable peak.
    
    When a peak is on top of a fluorescence baseline, the area is calculated
    as the area above the local baseline value at the peak location.
    
    Parameters:
    -----------
    wavenumbers : array
        Wavenumber values
    intensities : array
        Intensity values (should be background-subtracted spectrum for peaks on fluorescence)
    wavenumber_min : float
        Minimum wavenumber for search range
    wavenumber_max : float
        Maximum wavenumber for search range
    peak_type : str
        'gaussian' or 'lorentzian' for peak fitting
    prominence_factor : float
        Minimum peak prominence as fraction of max intensity in range
    width_min : float
        Minimum peak width in cm^-1
    width_max : float
        Maximum peak width in cm^-1
    fluorescence_baseline : array, optional
        Fluorescence baseline (e.g., Lieberfit baseline) for calculating peak area above baseline.
        If provided, peak area will be calculated above the local baseline value at the peak center.
        If None, peak area is calculated above the fitted offset (local minimum in fitting window).
    
    Returns:
    --------
    peak_info : dict or None
        Dictionary with peak information or None if not found
        - 'area': Peak area above the local baseline (fluorescence if provided, or fitted offset)
        - 'area_above_baseline': Peak area specifically above fluorescence baseline (if provided)
        - 'local_baseline': Local baseline value at peak center
    """
    # Create mask for the search range
    mask = (wavenumbers >= wavenumber_min) & (wavenumbers <= wavenumber_max)
    
    if not np.any(mask):
        return None
    
    wavenumbers_range = wavenumbers[mask]
    intensities_range = intensities[mask]
    
    if len(intensities_range) < 3:
        return None
    
    # Calculate statistics for better peak selection
    max_intensity = np.max(intensities_range)
    min_intensity = np.min(intensities_range)
    mean_intensity = np.mean(intensities_range)
    std_intensity = np.std(intensities_range)
    
    # Use a more robust prominence threshold
    # Consider both relative and absolute thresholds
    prominence_threshold_relative = (max_intensity - min_intensity) * prominence_factor
    prominence_threshold_absolute = mean_intensity + 2 * std_intensity  # Statistical threshold
    prominence_threshold = max(prominence_threshold_relative, prominence_threshold_absolute)
    
    # Find peaks using scipy.signal.find_peaks
    # Convert width from cm^-1 to indices
    wavenumber_spacing = np.mean(np.diff(wavenumbers_range))
    width_min_idx = max(1, int(width_min / wavenumber_spacing))
    width_max_idx = min(len(intensities_range) // 2, int(width_max / wavenumber_spacing))
    
    # Try multiple strategies to find the best peak
    candidate_peaks = []
    
    try:
        # Strategy 1: Use find_peaks with prominence
        peaks, properties = find_peaks(
            intensities_range,
            prominence=prominence_threshold,
            width=(width_min_idx, width_max_idx),
            height=min_intensity + prominence_threshold * 0.5  # Lower height threshold
        )
        
        if len(peaks) > 0:
            for i, peak_idx in enumerate(peaks):
                peak_wavenumber = wavenumbers_range[peak_idx]
                peak_intensity = intensities_range[peak_idx]
                prominence = properties['prominences'][i] if 'prominences' in properties else peak_intensity - min_intensity
                width = properties['widths'][i] * wavenumber_spacing if 'widths' in properties else None
                
                # Calculate a quality score (higher is better)
                # Prefer peaks with good prominence, reasonable width, and high intensity
                quality_score = prominence * (peak_intensity / max_intensity)
                if width is not None:
                    # Penalize peaks that are too narrow or too wide
                    if width_min <= width <= width_max:
                        quality_score *= 1.0
                    else:
                        quality_score *= 0.5
                
                candidate_peaks.append({
                    'idx': peak_idx,
                    'wavenumber': peak_wavenumber,
                    'intensity': peak_intensity,
                    'prominence': prominence,
                    'width': width,
                    'quality': quality_score
                })
        
        # Strategy 2: If no peaks found or only weak peaks, try with lower threshold
        if len(candidate_peaks) == 0 or max([p['quality'] for p in candidate_peaks]) < prominence_threshold * 0.5:
            peaks2, properties2 = find_peaks(
                intensities_range,
                prominence=prominence_threshold * 0.3,  # Lower threshold
                width=(max(1, width_min_idx // 2), width_max_idx * 2),
                height=mean_intensity
            )
            
            for i, peak_idx in enumerate(peaks2):
                # Skip if we already have this peak
                if any(abs(p['wavenumber'] - wavenumbers_range[peak_idx]) < 5 for p in candidate_peaks):
                    continue
                    
                peak_wavenumber = wavenumbers_range[peak_idx]
                peak_intensity = intensities_range[peak_idx]
                prominence = properties2['prominences'][i] if 'prominences' in properties2 else peak_intensity - min_intensity
                width = properties2['widths'][i] * wavenumber_spacing if 'widths' in properties2 else None
                
                quality_score = prominence * (peak_intensity / max_intensity)
                if width is not None and width_min <= width <= width_max:
                    quality_score *= 1.0
                else:
                    quality_score *= 0.3  # Lower weight for peaks outside expected width
                
                candidate_peaks.append({
                    'idx': peak_idx,
                    'wavenumber': peak_wavenumber,
                    'intensity': peak_intensity,
                    'prominence': prominence,
                    'width': width,
                    'quality': quality_score
                })
        
        # Strategy 3: If still no good peaks, use argmax as fallback
        if len(candidate_peaks) == 0:
            peak_idx = np.argmax(intensities_range)
            peak_wavenumber = wavenumbers_range[peak_idx]
            peak_intensity = intensities_range[peak_idx]
            prominence = peak_intensity - min_intensity
            
            candidate_peaks.append({
                'idx': peak_idx,
                'wavenumber': peak_wavenumber,
                'intensity': peak_intensity,
                'prominence': prominence,
                'width': None,
                'quality': prominence * (peak_intensity / max_intensity) * 0.5  # Lower quality for argmax
            })
        
        # Select the best candidate peak based on quality score
        best_candidate = max(candidate_peaks, key=lambda x: x['quality'])
        best_peak_idx = best_candidate['idx']
        peak_wavenumber = best_candidate['wavenumber']
        peak_intensity = best_candidate['intensity']
        
        # Fit curve around the best peak for sub-pixel accuracy
        # Use adaptive window size based on estimated peak width
        estimated_width = best_candidate['width'] if best_candidate['width'] is not None else 15.0
        fit_window = max(5, int(2.5 * estimated_width / wavenumber_spacing))  # ~2.5x peak width
        fit_window = min(fit_window, int(40 / wavenumber_spacing))  # Cap at 40 cm^-1
        start_idx = max(0, best_peak_idx - fit_window)
        end_idx = min(len(intensities_range), best_peak_idx + fit_window + 1)
        
        fit_wavenumbers = wavenumbers_range[start_idx:end_idx]
        fit_intensities = intensities_range[start_idx:end_idx]
        
        # Initial guess for fitting
        initial_amplitude = peak_intensity - np.min(fit_intensities)
        initial_center = peak_wavenumber
        initial_width = 10.0  # cm^-1
        initial_offset = np.min(fit_intensities)
        
        try:
            if peak_type == 'gaussian':
                popt, _ = curve_fit(
                    gaussian, fit_wavenumbers, fit_intensities,
                    p0=[initial_amplitude, initial_center, initial_width, initial_offset],
                    bounds=([0, wavenumber_min, 2, -np.inf], 
                           [np.inf, wavenumber_max, 50, np.inf])
                )
                fitted_center = popt[1]
                fitted_amplitude = popt[0]
                fitted_width = abs(popt[2])
            else:  # lorentzian
                popt, _ = curve_fit(
                    lorentzian, fit_wavenumbers, fit_intensities,
                    p0=[initial_amplitude, initial_center, initial_width, initial_offset],
                    bounds=([0, wavenumber_min, 2, -np.inf], 
                           [np.inf, wavenumber_max, 50, np.inf])
                )
                fitted_center = popt[1]
                fitted_amplitude = popt[0]
                fitted_width = abs(popt[2])
            
            # Calculate peak area above the local baseline
            # If fluorescence_baseline is provided, calculate area above it; otherwise use fitted offset
            if fluorescence_baseline is not None:
                # Interpolate the fluorescence baseline at the peak center
                from scipy.interpolate import interp1d
                baseline_interp = interp1d(wavenumbers, fluorescence_baseline, 
                                          kind='linear', fill_value='extrapolate')
                local_baseline_value = baseline_interp(fitted_center)
                
                # Calculate area above fluorescence baseline using numerical integration
                # This is more accurate when the fitted offset differs from fluorescence baseline
                fit_x = np.linspace(fit_wavenumbers[0], fit_wavenumbers[-1], 1000)
                if peak_type == 'gaussian':
                    fit_y = gaussian(fit_x, *popt)
                else:
                    fit_y = lorentzian(fit_x, *popt)
                
                # Area above fluorescence baseline: integrate (peak - fluorescence_baseline) where positive
                # Interpolate fluorescence baseline over the fit range
                baseline_at_fit = baseline_interp(fit_x)
                peak_above_baseline = np.maximum(fit_y - baseline_at_fit, 0)
                peak_area_above_baseline = np.trapz(peak_above_baseline, x=fit_x)
                
                # Also calculate analytical area above fitted offset for reference
                peak_area_analytical = calculate_peak_area_analytical(fitted_amplitude, fitted_width, peak_type)
                
                # Use area above fluorescence baseline as the primary result
                peak_area = peak_area_above_baseline
                local_baseline = local_baseline_value
            else:
                # No fluorescence baseline provided - use fitted offset as baseline
                peak_area_analytical = calculate_peak_area_analytical(fitted_amplitude, fitted_width, peak_type)
                
                # Also calculate numerically for verification/comparison
                fit_x = np.linspace(fit_wavenumbers[0], fit_wavenumbers[-1], 1000)
                if peak_type == 'gaussian':
                    fit_y = gaussian(fit_x, *popt)
                else:
                    fit_y = lorentzian(fit_x, *popt)
                peak_area_numerical = np.trapz(fit_y - popt[3], x=fit_x)  # Subtract offset
                
                peak_area = peak_area_analytical
                local_baseline = popt[3]  # Fitted offset
                peak_area_above_baseline = None
            
            return {
                'wavenumber': fitted_center,
                'intensity': fitted_amplitude + popt[3],  # Peak height from fitted offset
                'amplitude': fitted_amplitude,
                'width': fitted_width,
                'area': peak_area,  # Area above local baseline (fluorescence or fitted offset)
                'area_above_baseline': peak_area_above_baseline if fluorescence_baseline is not None else peak_area,
                'local_baseline': local_baseline,
                'fit_type': peak_type,
                'fit_params': popt,
                'fit_wavenumbers': fit_wavenumbers,
                'fit_intensities': fit_intensities
            }
        except Exception as e:
            # If fitting fails, return simple peak
            return {
                'wavenumber': peak_wavenumber,
                'intensity': peak_intensity,
                'amplitude': peak_intensity - initial_offset,
                'width': None,
                'area': None,
                'fit_type': None,
                'fit_params': None,
                'fit_wavenumbers': None,
                'fit_intensities': None
            }
            
    except Exception as e:
        # Fallback to simple argmax
        peak_idx = np.argmax(intensities_range)
        return {
            'wavenumber': wavenumbers_range[peak_idx],
            'intensity': intensities_range[peak_idx],
            'amplitude': intensities_range[peak_idx] - np.min(intensities_range),
            'width': None,
            'area': None,
            'fit_type': None,
            'fit_params': None,
            'fit_wavenumbers': None,
            'fit_intensities': None
        }

def optimize_lieberfit_parameters(wavenumbers, intensities_smooth, 
                                   order_range=(3, 4, 5, 6, 7, 8), 
                                   iter_range=(50, 100, 150, 200, 300),
                                   output_dir=None):
    """
    Test different Lieberfit parameter combinations and visualize results.
    
    Parameters:
    -----------
    wavenumbers : array
        Wavenumber values
    intensities_smooth : array
        Smoothed intensity spectrum
    order_range : tuple
        Range of polynomial orders to test
    iter_range : tuple
        Range of iteration counts to test
    output_dir : Path, optional
        Directory to save optimization plots
    
    Returns:
    --------
    best_params : dict
        Dictionary with best parameters found
    results : dict
        Dictionary with all tested parameter combinations and their metrics
    """
    print("\n=== Lieberfit Parameter Optimization ===")
    print(f"Testing polynomial orders: {order_range}")
    print(f"Testing iteration counts: {iter_range}")
    
    results = []
    
    # Test all combinations
    for order in order_range:
        for tot_iter in iter_range:
            try:
                corrected, baseline = lieberfit(intensities_smooth, order=order, tot_iter=tot_iter)
                
                # Calculate metrics to evaluate fit quality
                # 1. Residual sum of squares (RSS) - lower is better
                residuals = intensities_smooth - baseline
                rss = np.sum(residuals**2)
                
                # 2. Mean absolute error (MAE) - lower is better
                mae = np.mean(np.abs(residuals))
                
                # 3. Baseline smoothness (second derivative) - lower is better (smoother)
                baseline_diff2 = np.diff(baseline, n=2)
                smoothness = np.mean(baseline_diff2**2)
                
                # 4. Check if baseline is below spectrum (should be)
                baseline_below_spectrum = np.all(baseline <= intensities_smooth + 1e-6)
                
                # 5. Peak preservation - check if corrected spectrum preserves peaks
                # Look at G-band region (1500-1750 cm^-1)
                gband_mask = (wavenumbers >= 1500) & (wavenumbers <= 1750)
                if np.any(gband_mask):
                    gband_peak_corrected = np.max(corrected[gband_mask])
                    gband_peak_original = np.max(intensities_smooth[gband_mask])
                    peak_preservation = gband_peak_corrected / (gband_peak_original - np.min(intensities_smooth[gband_mask]) + 1e-6)
                else:
                    peak_preservation = 0
                
                # 6. Negative values in corrected spectrum (should be minimal)
                negative_fraction = np.sum(corrected < 0) / len(corrected)
                
                results.append({
                    'order': order,
                    'tot_iter': tot_iter,
                    'rss': rss,
                    'mae': mae,
                    'smoothness': smoothness,
                    'baseline_below_spectrum': baseline_below_spectrum,
                    'peak_preservation': peak_preservation,
                    'negative_fraction': negative_fraction,
                    'baseline': baseline,
                    'corrected': corrected
                })
                
            except Exception as e:
                print(f"  Warning: Failed for order={order}, tot_iter={tot_iter}: {e}")
                continue
    
    if not results:
        print("ERROR: No valid parameter combinations found!")
        return None, None
    
    # Find best parameters based on composite score
    # Lower RSS, lower MAE, lower smoothness, higher peak preservation, lower negative fraction
    for r in results:
        # Normalize metrics (0-1 scale, higher is better)
        rss_norm = 1.0 / (1.0 + r['rss'] / np.max([res['rss'] for res in results]))
        mae_norm = 1.0 / (1.0 + r['mae'] / np.max([res['mae'] for res in results]))
        smoothness_norm = 1.0 / (1.0 + r['smoothness'] / (np.max([res['smoothness'] for res in results]) + 1e-10))
        peak_norm = r['peak_preservation'] / (np.max([res['peak_preservation'] for res in results]) + 1e-10)
        neg_norm = 1.0 - r['negative_fraction']  # Lower negative fraction is better
        
        # Composite score (weighted average)
        r['score'] = (0.25 * rss_norm + 
                      0.25 * mae_norm + 
                      0.15 * smoothness_norm + 
                      0.20 * peak_norm + 
                      0.15 * neg_norm)
        
        if not r['baseline_below_spectrum']:
            r['score'] *= 0.5  # Penalize if baseline goes above spectrum
    
    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    best_result = results_sorted[0]
    best_params = {'order': best_result['order'], 'tot_iter': best_result['tot_iter']}
    
    print(f"\nBest parameters found:")
    print(f"  Polynomial order: {best_params['order']}")
    print(f"  Iterations: {best_params['tot_iter']}")
    print(f"  Score: {best_result['score']:.4f}")
    print(f"  RSS: {best_result['rss']:.2f}")
    print(f"  MAE: {best_result['mae']:.2f}")
    print(f"  Peak preservation: {best_result['peak_preservation']:.4f}")
    print(f"  Negative fraction: {best_result['negative_fraction']:.4f}")
    
    # Create visualization
    if output_dir is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Lieberfit Parameter Optimization', fontsize=14, fontweight='bold')
        
        # Plot 1: Compare top 5 parameter combinations
        ax1 = axes[0, 0]
        ax1.plot(wavenumbers, intensities_smooth, 'k-', label='Smoothed Spectrum', linewidth=2, alpha=0.7)
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(results_sorted))))
        for i, res in enumerate(results_sorted[:5]):
            label = f"order={res['order']}, iter={res['tot_iter']} (score={res['score']:.3f})"
            ax1.plot(wavenumbers, res['baseline'], '--', color=colors[i], 
                   linewidth=1.5, alpha=0.8, label=label)
        ax1.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax1.set_ylabel('Intensity', fontsize=12)
        ax1.set_title('Top 5 Baseline Fits', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter space heatmap (score)
        ax2 = axes[0, 1]
        orders = sorted(set([r['order'] for r in results]))
        iters = sorted(set([r['tot_iter'] for r in results]))
        score_matrix = np.zeros((len(orders), len(iters)))
        for r in results:
            i = orders.index(r['order'])
            j = iters.index(r['tot_iter'])
            score_matrix[i, j] = r['score']
        im = ax2.imshow(score_matrix, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_xticks(range(len(iters)))
        ax2.set_xticklabels(iters)
        ax2.set_yticks(range(len(orders)))
        ax2.set_yticklabels(orders)
        ax2.set_xlabel('Iterations', fontsize=12)
        ax2.set_ylabel('Polynomial Order', fontsize=12)
        ax2.set_title('Parameter Space (Score)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Score')
        
        # Plot 3: Best fit detailed view
        ax3 = axes[1, 0]
        ax3.plot(wavenumbers, intensities_smooth, 'gray', label='Smoothed Spectrum', 
                linewidth=1.5, alpha=0.6)
        ax3.plot(wavenumbers, best_result['baseline'], 'r--', 
                label=f'Best Baseline (order={best_params["order"]}, iter={best_params["tot_iter"]})', 
                linewidth=2)
        ax3.plot(wavenumbers, best_result['corrected'], 'g-', 
                label='Corrected Spectrum', linewidth=1.5)
        ax3.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax3.set_ylabel('Intensity', fontsize=12)
        ax3.set_title('Best Fit Result', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Metrics comparison
        ax4 = axes[1, 1]
        top_5 = results_sorted[:5]
        x_pos = np.arange(len(top_5))
        scores = [r['score'] for r in top_5]
        bars = ax4.bar(x_pos, scores, color=colors[:len(top_5)], alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"O{r['order']}I{r['tot_iter']}" for r in top_5], rotation=45, ha='right')
        ax4.set_ylabel('Composite Score', fontsize=12)
        ax4.set_title('Top 5 Parameter Combinations', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if output_dir:
            output_path = output_dir / "lieberfit_optimization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Optimization plot saved to: {output_path}")
        plt.close()
    
    return best_params, results

DEFAULT_TEST_PROFILE = "bok_choy_control_6to22_run1"


def test_new_method(
    scan_number=None,
    optimize_params=False,
    custom_order=None,
    custom_tot_iter=None,
    raman_df=None,
    config=None,
    output_dir=None,
    profile_name=DEFAULT_TEST_PROFILE,
    suppress_plot=False,
    verbose=True,
    skip_plot_creation=False,
    cached_wavenumbers_full=None,
    cached_wavenumber_filter=None,
    cached_wavenumbers=None,
):
    """
    Test new processing method with modified workflow.
    
    Parameters
    ----------
    suppress_plot : bool, optional
        If True, skip saving and showing plots (useful for batch processing).
        Default is False.
    verbose : bool, optional
        If False, suppress detailed console output (useful for batch processing).
        Default is True.
    skip_plot_creation : bool, optional
        If True, skip creating matplotlib figures entirely (faster for batch processing).
        Saves 50-200 ms per scan. Default is False.
    cached_wavenumbers_full : array, optional
        Pre-computed full wavenumber array. If provided, skips computation.
    cached_wavenumber_filter : array, optional
        Pre-computed wavenumber filter mask (>= 250 cm^-1). If provided, skips computation.
    cached_wavenumbers : array, optional
        Pre-computed filtered wavenumber array. If provided, skips computation.
    """
    
    # Load configuration if not provided
    if config is None:
        config_path = PROJECT_ROOT / "config" / "pipeline.yml"
        config = load_profile_config(str(config_path), profile_name)
        config_label = profile_name
    else:
        config_label = profile_name or config.get("profile_name", "custom_config")
    
    # Load dataset if not provided (needed for both raman_df and output_dir determination)
    if raman_df is None:
        if verbose:
            print("Loading dataset...")
        dataset = load_raman_dataset(config)
        raman_df = dataset.spectra
    else:
        dataset = None
    
    # Create output directory for test results
    if output_dir is None:
        # If we have dataset info, save next to the data file (like main_in_vitro.py)
        # Otherwise, save to scripts/test_outputs
        if dataset is not None:
            file_path = dataset.source_path
            import os
            from datetime import datetime
            base_path = os.path.dirname(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(base_path, f"processed_results_new_method_{timestamp}")
        else:
            # Fallback to default location if dataset not available
            output_dir = PROJECT_ROOT / "scripts" / "test_outputs"
    output_dir = Path(output_dir)
    create_dir_if_needed(str(output_dir))
    
    if verbose:
        print("=== New Processing Method Test ===\n")
        print(f"Config loaded: {config_label}")
        print(f"Excitation wavelength: {config.get('excitation_nm', 830)} nm")
        print(f"Window size: {config.get('window_size', 25)}")
        print(f"Poly order: {config.get('poly_order', 5)}")
        print(f"Lieberfit iterations: {config.get('tot_iter', 100)}\n")
    
    if verbose:
        print(f"Loaded {len(raman_df)} scans")
        print(f"Wavenumber range: {raman_df.columns[2]} to {raman_df.columns[-1]}")
    
    # Select a scan number (try scan 100, or first available)
    available_scans = sorted(raman_df['Scan Number'].unique())
    if not available_scans:
        print("ERROR: No scans available in dataset.")
        return None
    
    if scan_number is None:
        scan_number = 100 if 100 in available_scans else available_scans[len(available_scans) // 2]
    
    if scan_number not in available_scans:
        print(f"ERROR: Scan {scan_number} not found! Available scans: {available_scans[:10]}{'...' if len(available_scans) > 10 else ''}")
        return None
    
    if verbose:
        print(f"\nProcessing Scan Number: {scan_number}")
    
    # Extract scan data
    scan_data = raman_df[raman_df['Scan Number'] == scan_number]
    if scan_data.empty:
        print(f"ERROR: Scan {scan_number} not found!")
        return
    
    # Extract time and intensities
    datetime_val = scan_data.index[0]
    seconds = scan_data['Seconds'].iloc[0]
    
    # Use cached wavenumber arrays if provided, otherwise compute them
    if cached_wavenumbers_full is not None and cached_wavenumber_filter is not None and cached_wavenumbers is not None:
        wavenumbers_full = cached_wavenumbers_full
        wavenumber_filter = cached_wavenumber_filter
        wavenumbers = cached_wavenumbers
    else:
        wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
        wavenumber_filter = wavenumbers_full >= 250
        wavenumbers = wavenumbers_full[wavenumber_filter]
    
    intensities_raw_full = scan_data.iloc[0, 2:].values
    intensities_raw = intensities_raw_full[wavenumber_filter]
    
    # Step 1: Filter wavenumber to 250 cm-1 onward
    if verbose:
        print("\nStep 1: Filtering wavenumber to 250 cm-1 onward...")
    if verbose:
        print(f"Filtered spectrum: {len(wavenumbers)} points (from {wavenumbers[0]:.1f} to {wavenumbers[-1]:.1f} cm^-1)")
    
    # Step 2: Smooth with Savitzky-Golay
    if verbose:
        print("\nStep 2: Applying Savitzky-Golay smoothing...")
    window_size = config.get('window_size', 25)
    poly_order_sg = config.get('poly_order_sg', 2)
    # Ensure window_size is odd and less than data length
    if window_size >= len(intensities_raw):
        window_size = len(intensities_raw) - 1 if len(intensities_raw) % 2 == 0 else len(intensities_raw) - 2
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        window_size = 3
    intensities_smooth = savgol_filter(intensities_raw, window_size, poly_order_sg)
    if verbose:
        print(f"Smoothed with window_size={window_size}, poly_order={poly_order_sg}")
    
    # Step 4: Lieberfit applied to smoothed spectrum
    if verbose:
        print("\nStep 4: Applying Lieberfit to smoothed spectrum...")
    
    # Use custom parameters if provided, otherwise use config or optimize
    if custom_order is not None:
        poly_order = custom_order
    else:
        poly_order = config.get('poly_order', 5)
    
    if custom_tot_iter is not None:
        tot_iter = custom_tot_iter
    else:
        tot_iter = config.get('tot_iter', 100)
    
    # Optimize parameters if requested
    if optimize_params:
        best_params, opt_results = optimize_lieberfit_parameters(
            wavenumbers, intensities_smooth, 
            order_range=(3, 4, 5, 6, 7, 8),
            iter_range=(50, 100, 150, 200, 300),
            output_dir=output_dir
        )
        if best_params:
            poly_order = best_params['order']
            tot_iter = best_params['tot_iter']
            if verbose:
                print(f"\nUsing optimized parameters: order={poly_order}, iterations={tot_iter}")
    
    # Apply Lieberfit to smoothed spectrum
    corrected_spectrum, baseline_lieberfit = lieberfit(intensities_smooth, order=poly_order, tot_iter=tot_iter)
    if verbose:
        print(f"Lieberfit applied with order={poly_order}, iterations={tot_iter}")
    
    # Step 5: Detect Raman peak on the Lieberfit-corrected spectrum, perform Lorentzian fitting
    if verbose:
        print("\nStep 5: Detecting Raman peak on Lieberfit-corrected spectrum with Lorentzian fitting...")
    
    # Narrowed ranges for more accurate peak detection
    # Raman peak: typically around 800-900 cm^-1 for some materials, but user specified 750-1000
    # G-band: typically around 1580-1600 cm^-1 for carbon nanotubes, user specified 1500-1750
    # Note: If peak at 939 cm^-1 seems misplaced, try adjusting these ranges
    raman_peak_min = 800  # Narrowed from 750 - adjust if needed
    raman_peak_max = 950  # Narrowed from 1000 - adjust if needed
    gband_peak_min = 1560  # Narrowed from 1500
    gband_peak_max = 1620  # Narrowed from 1750
    
    # Detect Raman peak on Lieberfit-corrected spectrum
    # Use Lorentzian fitting as specified
    raman_peak_info = find_peak_accurate(
        wavenumbers, corrected_spectrum,  # Use Lieberfit-corrected spectrum
        raman_peak_min, raman_peak_max,
        peak_type='lorentzian',  # Lorentzian fitting as specified
        prominence_factor=0.1,
        width_min=8,
        width_max=40,
        fluorescence_baseline=None  # No fluorescence baseline needed for corrected spectrum
    )
    
    if raman_peak_info is None:
        # Try with lower threshold if no peak found
        raman_peak_info = find_peak_accurate(
            wavenumbers, corrected_spectrum,
            raman_peak_min, raman_peak_max,
            peak_type='lorentzian',
            prominence_factor=0.05,
            width_min=5,
            width_max=40,
            fluorescence_baseline=None
        )
    
    if raman_peak_info:
        raman_peak_wavenumber = raman_peak_info['wavenumber']
        raman_peak_intensity = raman_peak_info['intensity']
        raman_peak_amplitude = raman_peak_info['amplitude']
        raman_peak_width = raman_peak_info['width']
        raman_peak_area = raman_peak_info['area']
        if verbose:
            print(f"Raman peak found at {raman_peak_wavenumber:.2f} cm^-1")
            print(f"  Intensity: {raman_peak_intensity:.2f}")
            print(f"  Amplitude: {raman_peak_amplitude:.2f}")
            if raman_peak_width:
                print(f"  Width (FWHM): {raman_peak_width:.2f} cm^-1")
            if raman_peak_area:
                print(f"  Area (analytical): {raman_peak_area:.2f}")
            if raman_peak_info['fit_type']:
                print(f"  Fit type: {raman_peak_info['fit_type']} (Lorentzian)")
            
            # Diagnostic: Check if peak position seems reasonable
            if raman_peak_wavenumber < raman_peak_min + 10 or raman_peak_wavenumber > raman_peak_max - 10:
                print(f"  WARNING: Peak is near the edge of search range ({raman_peak_min}-{raman_peak_max} cm^-1)")
                print(f"  Consider adjusting the search range if this seems incorrect")
    else:
        if verbose:
            print("Warning: No Raman peak found in region (800-950 cm^-1)")
            print("  Try adjusting raman_peak_min and raman_peak_max in the code if peak exists elsewhere")
        raman_peak_wavenumber = None
        raman_peak_intensity = None
        raman_peak_amplitude = None
        raman_peak_width = None
        raman_peak_area = None
        raman_peak_info = None
    
    # Step 6: Calculate the peak area of the Raman peaks
    # (Already calculated in find_peak_accurate, but we'll print it here)
    if verbose:
        print("\nStep 6: Peak area calculated from Lorentzian fit")
    
    # Also detect G-band peak (1500-1750 cm^-1) on Lieberfit-corrected spectrum
    if verbose:
        print("\nStep 6 (continued): Detecting G-band peak on Lieberfit-corrected spectrum...")
    gband_peak_min = 1500
    gband_peak_max = 1750
    
    gband_peak_info = find_peak_accurate(
        wavenumbers, corrected_spectrum,  # Use Lieberfit-corrected spectrum
        gband_peak_min, gband_peak_max,
        peak_type='lorentzian',  # Lorentzian fitting as specified
        prominence_factor=0.1,
        width_min=8,
        width_max=50,
        fluorescence_baseline=None  # No fluorescence baseline needed for corrected spectrum
    )
    
    if gband_peak_info is None:
        # Try with lower threshold if no peak found
        gband_peak_info = find_peak_accurate(
            wavenumbers, corrected_spectrum,
            gband_peak_min, gband_peak_max,
            peak_type='lorentzian',
            prominence_factor=0.05,
            width_min=5,
            width_max=50,
            fluorescence_baseline=None
        )
    
    if gband_peak_info:
        gband_peak_wavenumber = gband_peak_info['wavenumber']
        gband_peak_intensity = gband_peak_info['intensity']
        gband_peak_amplitude = gband_peak_info['amplitude']
        gband_peak_width = gband_peak_info['width']
        gband_area = gband_peak_info['area']
        if verbose:
            print(f"G-band peak found at {gband_peak_wavenumber:.2f} cm^-1")
            print(f"  Intensity: {gband_peak_intensity:.2f}")
            print(f"  Amplitude: {gband_peak_amplitude:.2f}")
            if gband_peak_width:
                print(f"  Width (FWHM): {gband_peak_width:.2f} cm^-1")
            if gband_area:
                print(f"  Area (analytical): {gband_area:.2f}")
            if gband_peak_info['fit_type']:
                print(f"  Fit type: {gband_peak_info['fit_type']} (Lorentzian)")
        
        # Also calculate area using trapezoidal integration for comparison
        gband_mask = (wavenumbers >= gband_peak_min) & (wavenumbers <= gband_peak_max)
        gband_area_trapz = np.trapz(corrected_spectrum[gband_mask], x=wavenumbers[gband_mask])
        if verbose:
            print(f"  Area (trapezoidal): {gband_area_trapz:.2f}")
    else:
        if verbose:
            print("Warning: No G-band peak found in region (1500-1750 cm^-1)")
        gband_peak_wavenumber = None
        gband_peak_intensity = None
        gband_peak_amplitude = None
        gband_peak_width = None
        gband_area = None
        gband_peak_info = None
        gband_mask = (wavenumbers >= gband_peak_min) & (wavenumbers <= gband_peak_max)
        gband_area_trapz = np.trapz(corrected_spectrum[gband_mask], x=wavenumbers[gband_mask]) if np.any(gband_mask) else None
    
    if gband_peak_info:
        gband_peak_wavenumber = gband_peak_info['wavenumber']
        gband_peak_intensity = gband_peak_info['intensity']
        gband_peak_amplitude = gband_peak_info['amplitude']
        gband_peak_width = gband_peak_info['width']
        gband_area = gband_peak_info['area']
        if verbose:
            print(f"\nG-band peak found at {gband_peak_wavenumber:.2f} cm^-1")
            print(f"  Intensity: {gband_peak_intensity:.2f}")
            print(f"  Amplitude: {gband_peak_amplitude:.2f}")
            if gband_peak_width:
                print(f"  Width (FWHM): {gband_peak_width:.2f} cm^-1")
    # Step 7: Subtract average background (from smoothed spectrum, 250-1250 cm^-1) from Lieberfit baseline
    if verbose:
        print("\nStep 7: Subtracting average background (smoothed spectrum, 250-1250 cm^-1) from Lieberfit baseline...")
    baseline_bg_mask = (wavenumbers >= 250) & (wavenumbers <= 1250)
    if np.any(baseline_bg_mask):
        bg_intensity_avg_250_1250 = np.mean(intensities_smooth[baseline_bg_mask])
    else:
        bg_intensity_avg_250_1250 = np.mean(intensities_smooth)
    if verbose:
        print(f"Average background (smoothed spectrum, 250-1250 cm^-1): {bg_intensity_avg_250_1250:.2f}")
    baseline_bg_intensity = bg_intensity_avg_250_1250  # For backward-compatible variable naming
    
    # Subtract this average from the Lieberfit baseline
    baseline_lieberfit_bg_subtracted = baseline_lieberfit - bg_intensity_avg_250_1250
    if verbose:
        print(f"Background-subtracted Lieberfit baseline range: [{baseline_lieberfit_bg_subtracted.min():.2f}, {baseline_lieberfit_bg_subtracted.max():.2f}]")
    
    # Step 8: Calculate fluorescence using two different methods
    if verbose:
        print("\nStep 8: Calculating fluorescence using two methods...")
    fluo_mask = wavenumbers >= 1250
    
    # Method 1: Current method - AUC of background-subtracted Lieberfit baseline
    if verbose:
        print("\nMethod 1: AUC of background-subtracted Lieberfit baseline...")
    # Use background-subtracted baseline in fluorescence window
    baseline_fluo = np.maximum(baseline_lieberfit_bg_subtracted[fluo_mask], 0)  # Ensure non-negative
    
    # Calculate fluorescence as AUC of background-subtracted Lieberfit baseline
    fluorescence_method1 = np.trapz(baseline_fluo, x=wavenumbers[fluo_mask])
    
    if verbose:
        print(f"  Fluorescence window: >= 1250 cm^-1")
        print(f"  Background-subtracted baseline range in fluo window: [{baseline_fluo.min():.2f}, {baseline_fluo.max():.2f}]")
        print(f"  Fluorescence (Method 1): {fluorescence_method1:.2f}")
    
    # Method 2: Raw AUC - G-band area - Background AUC
    if verbose:
        print("\nMethod 2: Raw AUC - G-band area - Background AUC...")
    # Calculate raw AUC of fluorescence region from smoothed spectrum
    raw_fluo_auc = np.trapz(intensities_smooth[fluo_mask], x=wavenumbers[fluo_mask])
    if verbose:
        print(f"  Raw AUC (>= 1250 cm^-1): {raw_fluo_auc:.2f}")
    
    # Subtract G-band area (if available)
    gband_area_to_subtract = gband_area if gband_area is not None else 0.0
    if verbose:
        if gband_area is not None:
            print(f"  G-band area to subtract: {gband_area_to_subtract:.2f}")
        else:
            print(f"  G-band area: Not available (using 0)")
    
    # Background average already computed (same source as Step 7)
    if verbose:
        print(f"  Background (average 250-1250 cm^-1): {bg_intensity_avg_250_1250:.2f}")
    
    # Calculate background AUC over the fluorescence window (>= 1250 cm^-1)
    if np.any(fluo_mask):
        fluo_start = wavenumbers[fluo_mask][0]
        fluo_end = wavenumbers[fluo_mask][-1]
    else:
        fluo_start = 1250.0
        fluo_end = wavenumbers[-1]
    fluo_width = max(fluo_end - fluo_start, 0)
    bg_auc_fluo_window = bg_intensity_avg_250_1250 * fluo_width
    if verbose:
        print(f"  Background AUC in fluorescence window (>= {fluo_start:.0f} cm^-1, width={fluo_width:.1f}): {bg_auc_fluo_window:.2f}")
    
    # Method 2: Raw AUC - G-band - Background AUC
    fluorescence_method2 = raw_fluo_auc - gband_area_to_subtract - bg_auc_fluo_window
    fluorescence_method2 = max(fluorescence_method2, 0)  # Ensure non-negative
    
    if verbose:
        print(f"  Fluorescence (Method 2): {fluorescence_method2:.2f}")
        print(f"    = Raw AUC ({raw_fluo_auc:.2f}) - G-band ({gband_area_to_subtract:.2f}) - Background AUC ({bg_auc_fluo_window:.2f})")
    
    # Method 3: Apply same processes as Method 1 and 2, but on normalized spectrum
    if verbose:
        print("\nMethod 3: Normalized spectrum processing (normalize first, then apply Method 1 & 2 workflows)...")
    
    # Step 1: Normalize the smoothed spectrum by dividing by the average intensity in 250-1250 cm^-1 window
    # bg_intensity_avg_250_1250 is already calculated above
    if bg_intensity_avg_250_1250 > 0:
        intensities_normalized = intensities_smooth / bg_intensity_avg_250_1250
        if verbose:
            print(f"  Step 1: Normalization factor (average 250-1250 cm^-1): {bg_intensity_avg_250_1250:.2f}")
            print(f"  Normalized intensity range: [{intensities_normalized.min():.2f}, {intensities_normalized.max():.2f}]")
    else:
        # Avoid division by zero
        intensities_normalized = intensities_smooth
        if verbose:
            print(f"  Warning: Average intensity is zero, skipping normalization")
    
    # Step 2: Apply Lieberfit to normalized spectrum (same as Method 1 workflow)
    if verbose:
        print(f"  Step 2: Applying Lieberfit to normalized spectrum (order={poly_order}, iterations={tot_iter})...")
    corrected_spectrum_normalized, baseline_lieberfit_normalized = lieberfit(intensities_normalized, order=poly_order, tot_iter=tot_iter)
    
    # Step 3: Detect G-band peak on normalized corrected spectrum (needed for Method 2 equivalent)
    # Use same G-band peak ranges as original detection (1500-1750 cm^-1)
    gband_peak_min_normalized = 1500
    gband_peak_max_normalized = 1750
    if verbose:
        print(f"  Step 3: Detecting G-band peak on normalized corrected spectrum...")
    gband_peak_info_normalized = find_peak_accurate(
        wavenumbers, corrected_spectrum_normalized,
        gband_peak_min_normalized, gband_peak_max_normalized,
        peak_type='lorentzian',
        prominence_factor=0.1,
        width_min=8,
        width_max=50,
        fluorescence_baseline=None
    )
    
    if gband_peak_info_normalized is None:
        # Try with lower threshold if no peak found
        gband_peak_info_normalized = find_peak_accurate(
            wavenumbers, corrected_spectrum_normalized,
            gband_peak_min_normalized, gband_peak_max_normalized,
            peak_type='lorentzian',
            prominence_factor=0.05,
            width_min=5,
            width_max=50,
            fluorescence_baseline=None
        )
    
    gband_area_normalized = gband_peak_info_normalized['area'] if gband_peak_info_normalized else None
    
    # Step 4: Calculate average background on normalized spectrum (250-1250 cm^-1)
    baseline_bg_mask = (wavenumbers >= 250) & (wavenumbers <= 1250)
    if np.any(baseline_bg_mask):
        bg_intensity_avg_normalized = np.mean(intensities_normalized[baseline_bg_mask])
    else:
        bg_intensity_avg_normalized = np.mean(intensities_normalized)
    
    if verbose:
        print(f"  Average background on normalized spectrum (250-1250 cm^-1): {bg_intensity_avg_normalized:.4f}")
        if gband_area_normalized is not None:
            print(f"  G-band area on normalized spectrum: {gband_area_normalized:.4f}")
    
    # Step 5: Normalized Method 1 - Background-subtracted Lieberfit baseline AUC
    baseline_lieberfit_bg_subtracted_normalized = baseline_lieberfit_normalized - bg_intensity_avg_normalized
    baseline_fluo_normalized = np.maximum(baseline_lieberfit_bg_subtracted_normalized[fluo_mask], 0)  # Ensure non-negative
    fluorescence_method1_normalized = np.trapz(baseline_fluo_normalized, x=wavenumbers[fluo_mask])
    
    if verbose:
        print(f"  Normalized Method 1: {fluorescence_method1_normalized:.4f}")
    
    # Step 6: Method 3 - Method 2 equivalent: Raw normalized AUC - G-band area - Background AUC
    raw_fluo_auc_normalized = np.trapz(intensities_normalized[fluo_mask], x=wavenumbers[fluo_mask])
    gband_area_to_subtract_normalized = gband_area_normalized if gband_area_normalized is not None else 0.0
    
    # Calculate background AUC over the fluorescence window (>= 1250 cm^-1) on normalized spectrum
    if np.any(fluo_mask):
        fluo_start = wavenumbers[fluo_mask][0]
        fluo_end = wavenumbers[fluo_mask][-1]
    else:
        fluo_start = 1250.0
        fluo_end = wavenumbers[-1]
    fluo_width = max(fluo_end - fluo_start, 0)
    bg_auc_fluo_window_normalized = bg_intensity_avg_normalized * fluo_width
    
    fluorescence_method2_normalized = raw_fluo_auc_normalized - gband_area_to_subtract_normalized - bg_auc_fluo_window_normalized
    fluorescence_method2_normalized = max(fluorescence_method2_normalized, 0)  # Ensure non-negative
    
    if verbose:
        print(f"  Normalized Method 2: {fluorescence_method2_normalized:.4f}")
        print(f"    = Raw normalized AUC ({raw_fluo_auc_normalized:.4f}) - G-band ({gband_area_to_subtract_normalized:.4f}) - Background AUC ({bg_auc_fluo_window_normalized:.4f})")
    
    # Use Method 1 as the primary fluorescence value (for backward compatibility)
    fluorescence = fluorescence_method1
    
    # Print comparison
    if verbose:
        print(f"\n=== Fluorescence Comparison ===")
        print(f"\nRaw (Original) Fluorescent Intensity Analysis:")
        print(f"  Method 1 (Lieberfit baseline AUC): {fluorescence_method1:.2f}")
        print(f"  Method 2 (Raw - G-band - Background): {fluorescence_method2:.2f}")
        print(f"  Raw AUC (no correction): {raw_fluo_auc:.2f}")
        print(f"\nNormalized Fluorescent Intensity Analysis:")
        print(f"  Method 1 (Lieberfit baseline AUC): {fluorescence_method1_normalized:.4f}")
        print(f"  Method 2 (Raw - G-band - Background): {fluorescence_method2_normalized:.4f}")
        print(f"  Raw AUC (no correction): {raw_fluo_auc_normalized:.4f}")
        print(f"\nDifference (Raw Method 1 vs Method 2): {abs(fluorescence_method1 - fluorescence_method2):.2f} ({abs(fluorescence_method1 - fluorescence_method2) / max(fluorescence_method1, fluorescence_method2) * 100 if max(fluorescence_method1, fluorescence_method2) > 0 else 0:.1f}%)")
    
    # Initialize output paths
    output_path = None
    output_path_norm = None
    
    # Create comprehensive visualization (skip if skip_plot_creation is True)
    if skip_plot_creation:
        fig = None
        axes = None
    else:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Scan {scan_number} - New Processing Method\n'
                     f'DateTime: {datetime_val}, Seconds: {seconds:.1f}', 
                     fontsize=14, fontweight='bold')
        
        # Calculate common y-axis range for all plots (for easier comparison)
        # Collect all data values that will be plotted
        all_y_values = []
        all_y_values.extend(intensities_raw)
        all_y_values.extend(intensities_smooth)
        all_y_values.extend(baseline_lieberfit)
        all_y_values.extend(baseline_lieberfit_bg_subtracted)
        all_y_values.extend(corrected_spectrum)
        all_y_values.append(baseline_bg_intensity)
        all_y_values.append(0)  # Include zero for reference
        
        # Add fitted peak values if available
        if raman_peak_info and raman_peak_info['fit_params'] is not None:
            fit_x = np.linspace(raman_peak_info['fit_wavenumbers'][0], 
                               raman_peak_info['fit_wavenumbers'][-1], 200)
            if raman_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *raman_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *raman_peak_info['fit_params'])
            all_y_values.extend(fit_y)
        
        if gband_peak_info and gband_peak_info['fit_params'] is not None:
            fit_x = np.linspace(gband_peak_info['fit_wavenumbers'][0], 
                               gband_peak_info['fit_wavenumbers'][-1], 200)
            if gband_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *gband_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *gband_peak_info['fit_params'])
            all_y_values.extend(fit_y)
        
        # Calculate y-range with padding
        y_min = np.min(all_y_values)
        y_max = np.max(all_y_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.05  # 5% padding on each side
        y_lim_shared = (y_min - y_padding, y_max + y_padding)
        
        # Plot 1: Raw vs Smoothed
        ax1 = axes[0, 0]
        ax1.plot(wavenumbers, intensities_raw, 'b-', label='Raw Spectrum', alpha=0.6, linewidth=1)
        ax1.plot(wavenumbers, intensities_smooth, 'r-', label='Smoothed (Savitzky-Golay)', alpha=0.8, linewidth=1.5)
        ax1.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax1.set_ylabel('Intensity', fontsize=12)
        ax1.set_title('Step 1-2: Raw vs Smoothed', fontsize=12, fontweight='bold')
        ax1.set_ylim(y_lim_shared)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(250, color='gray', linestyle='--', alpha=0.5, label='Filter cutoff')
        
        # Plot 2: Lieberfit Baseline
        ax2 = axes[0, 1]
        ax2.plot(wavenumbers, intensities_smooth, 'gray', label='Smoothed Spectrum', alpha=0.6, linewidth=1)
        ax2.plot(wavenumbers, baseline_lieberfit, 'r--', label=f'Lieberfit Baseline (order={poly_order}, iter={tot_iter})', linewidth=2)
        ax2.plot(wavenumbers, corrected_spectrum, 'g-', label='Lieberfit-Corrected Spectrum', linewidth=1.5)
        ax2.axhline(y=baseline_bg_intensity, color='orange', linestyle='--', linewidth=2, label=f'Baseline Avg (250-1250 cm^-1): {baseline_bg_intensity:.1f}')
        ax2.fill_between(wavenumbers[baseline_bg_mask], baseline_lieberfit[baseline_bg_mask],
                         alpha=0.2, color='orange', label='Baseline Background Region (250-1250 cm^-1)')
        ax2.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax2.set_ylabel('Intensity', fontsize=12)
        ax2.set_title('Step 4: Lieberfit Baseline Correction', fontsize=12, fontweight='bold')
        ax2.set_ylim(y_lim_shared)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Peak Detection on Corrected Spectrum
        ax3 = axes[1, 0]
        ax3.plot(wavenumbers, corrected_spectrum, 'g-', label='Lieberfit-Corrected Spectrum', linewidth=1.5)
        if raman_peak_wavenumber is not None:
            ax3.axvline(raman_peak_wavenumber, color='blue', linestyle=':', alpha=0.7, 
                       label=f'Raman Peak ({raman_peak_wavenumber:.1f} cm^-1)')
            raman_peak_mask = (wavenumbers >= raman_peak_min) & (wavenumbers <= raman_peak_max)
            ax3.fill_between(wavenumbers[raman_peak_mask], corrected_spectrum[raman_peak_mask],
                             alpha=0.15, color='blue', label=f'Raman Region ({raman_peak_min}-{raman_peak_max} cm^-1)')
        if gband_peak_wavenumber is not None:
            ax3.axvline(gband_peak_wavenumber, color='green', linestyle=':', alpha=0.7, 
                       label=f'G-band Peak ({gband_peak_wavenumber:.1f} cm^-1)')
            gband_mask = (wavenumbers >= gband_peak_min) & (wavenumbers <= gband_peak_max)
            ax3.fill_between(wavenumbers[gband_mask], corrected_spectrum[gband_mask],
                             alpha=0.15, color='green', label=f'G-band Region ({gband_peak_min}-{gband_peak_max} cm^-1)')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax3.set_ylabel('Intensity', fontsize=12)
        ax3.set_title('Step 5: Peak Detection on Corrected Spectrum', fontsize=12, fontweight='bold')
        ax3.set_ylim(y_lim_shared)
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Corrected Spectrum with Fitted Peaks
        ax4 = axes[1, 1]
        ax4.plot(wavenumbers, corrected_spectrum, 'g-', label='Corrected Spectrum', linewidth=1.5, alpha=0.7)
        
        # Plot fitted peaks if available
        # Note: Peaks were fitted on Lieberfit-corrected spectrum, so they're already on the correct scale
        if raman_peak_info and raman_peak_info['fit_params'] is not None:
            # Extend the fit range to cover the whole spectrum for better visualization
            # Use the full wavenumber range to show how the peak extends across the spectrum
            fit_x = np.linspace(wavenumbers[0], wavenumbers[-1], 1000)  # Full spectrum range
            
            if raman_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *raman_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *raman_peak_info['fit_params'])
            
            # Fitted peak is already on corrected_spectrum scale (since it was fitted on corrected_spectrum)
            ax4.plot(fit_x, fit_y, 'b--', linewidth=2, alpha=0.8, label='Raman Peak Fit (Lorentzian)')
            
            # Shade Lorentzian area (between fitted curve and fitted offset) within Raman window
            baseline_offset = raman_peak_info['fit_params'][3] if len(raman_peak_info['fit_params']) > 3 else 0.0
            raman_fit_mask = (fit_x >= raman_peak_min) & (fit_x <= raman_peak_max)
            ax4.fill_between(
                fit_x[raman_fit_mask],
                baseline_offset,
                fit_y[raman_fit_mask],
                alpha=0.2,
                color='blue',
                label='Raman Peak Area (Lorentzian fit)'
            )
            
            # Peak intensity is already on corrected spectrum scale
            raman_peak_intensity_corrected = raman_peak_intensity
            ax4.plot(raman_peak_wavenumber, raman_peak_intensity_corrected, 'bo', markersize=10, 
                    label=f'Raman Peak ({raman_peak_wavenumber:.1f} cm⁻¹)')
        elif raman_peak_wavenumber is not None:
            # No fit available, just plot the peak position from corrected spectrum
            raman_peak_mask = (wavenumbers >= raman_peak_min) & (wavenumbers <= raman_peak_max)
            raman_peak_idx = np.argmin(np.abs(wavenumbers - raman_peak_wavenumber))
            raman_peak_intensity_corrected = corrected_spectrum[raman_peak_idx]
            ax4.plot(raman_peak_wavenumber, raman_peak_intensity_corrected, 'bo', markersize=10, 
                    label=f'Raman Peak ({raman_peak_wavenumber:.1f} cm⁻¹)')
            ax4.fill_between(wavenumbers[raman_peak_mask], corrected_spectrum[raman_peak_mask],
                             alpha=0.15, color='blue', label='Raman Peak Area (corrected)')
        
        # Plot G-band peak if available
        if gband_peak_info and gband_peak_info['fit_params'] is not None:
            # Extend the fit range to cover the whole spectrum for better visualization
            fit_x = np.linspace(wavenumbers[0], wavenumbers[-1], 1000)  # Full spectrum range
            
            if gband_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *gband_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *gband_peak_info['fit_params'])
            
            # Fitted peak is already on corrected_spectrum scale (since it was fitted on corrected_spectrum)
            ax4.plot(fit_x, fit_y, 'g--', linewidth=2, alpha=0.8, label='G-band Peak Fit (Lorentzian)')
            
            # Shade Lorentzian area (between fitted curve and fitted offset) within G-band window
            baseline_offset = gband_peak_info['fit_params'][3] if len(gband_peak_info['fit_params']) > 3 else 0.0
            gband_fit_mask = (fit_x >= gband_peak_min) & (fit_x <= gband_peak_max)
            ax4.fill_between(
                fit_x[gband_fit_mask],
                baseline_offset,
                fit_y[gband_fit_mask],
                alpha=0.2,
                color='green',
                label='G-band Peak Area (Lorentzian fit)'
            )
            
            # Peak intensity is already on corrected spectrum scale
            gband_peak_intensity_corrected = gband_peak_intensity
            ax4.plot(gband_peak_wavenumber, gband_peak_intensity_corrected, 'go', markersize=10, 
                    label=f'G-band Peak ({gband_peak_wavenumber:.1f} cm⁻¹)')
        elif gband_peak_wavenumber is not None:
            # No fit available, just plot the peak position from corrected spectrum
            gband_mask = (wavenumbers >= gband_peak_min) & (wavenumbers <= gband_peak_max)
            gband_peak_idx = np.argmin(np.abs(wavenumbers - gband_peak_wavenumber))
            gband_peak_intensity_corrected = corrected_spectrum[gband_peak_idx]
            ax4.plot(gband_peak_wavenumber, gband_peak_intensity_corrected, 'go', markersize=10, 
                    label=f'G-band Peak ({gband_peak_wavenumber:.1f} cm⁻¹)')
            ax4.fill_between(wavenumbers[gband_mask], corrected_spectrum[gband_mask],
                             alpha=0.15, color='green', label='G-band Peak Area')
        
        ax4.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax4.set_ylabel('Intensity', fontsize=12)
        ax4.set_title('Step 5-6: Raman and G-band Peak Detection and Area Calculation', fontsize=12, fontweight='bold')
        ax4.set_ylim(y_lim_shared)
        ax4.legend(fontsize=9, loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.text(0.02, 0.02, 'Note: Peak detected and fitted on Lieberfit-corrected spectrum\nLorentzian fitting used for peak area calculation', 
                transform=ax4.transAxes, fontsize=8, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 5: Fluorescence Window - Comparison of Two Methods
        ax5 = axes[2, 0]
        ax5.plot(wavenumbers, intensities_smooth, 'gray', label='Smoothed Spectrum', alpha=0.6, linewidth=1)
        ax5.plot(wavenumbers, baseline_lieberfit, 'r--', label='Lieberfit Baseline (original)', linewidth=2, alpha=0.8)
        ax5.axhline(y=baseline_bg_intensity, color='orange', linestyle='--', linewidth=2, 
                    label=f'Baseline Avg (250-1250 cm^-1): {baseline_bg_intensity:.1f}')
        ax5.plot(wavenumbers, baseline_lieberfit_bg_subtracted, 'purple', label='Baseline (bg-subtracted)', linewidth=2, alpha=0.8)
        ax5.axhline(y=bg_intensity_avg_250_1250, color='cyan', linestyle='--', linewidth=2, 
                    label=f'Background Avg (250-1250 cm^-1): {bg_intensity_avg_250_1250:.1f}')
        
        # Method 1: Fill area under background-subtracted Lieberfit baseline
        ax5.fill_between(wavenumbers[fluo_mask], 0, baseline_fluo,
                         alpha=0.3, color='purple', label=f'Method 1 AUC = {fluorescence_method1:.1f}')
        
        # Method 2: Show raw fluorescence region, G-band, and background
        # Fill raw fluorescence region
        ax5.fill_between(wavenumbers[fluo_mask], 0, intensities_smooth[fluo_mask],
                         alpha=0.2, color='blue', label=f'Raw AUC = {raw_fluo_auc:.1f}')
        
        # Mark G-band area if available
        if gband_area is not None and gband_area > 0:
            # Show G-band region
            gband_fluo_overlap = (wavenumbers >= max(1250, gband_peak_min)) & (wavenumbers <= min(wavenumbers[fluo_mask][-1], gband_peak_max))
            if np.any(gband_fluo_overlap):
                ax5.fill_between(wavenumbers[gband_fluo_overlap], 0, intensities_smooth[gband_fluo_overlap],
                                 alpha=0.15, color='green', label=f'G-band area = {gband_area:.1f}')
        
        # Show background level in fluorescence region
        ax5.fill_between(wavenumbers[fluo_mask], 0, bg_intensity_avg_250_1250,
                         alpha=0.1, color='cyan', label=f'Background AUC (>=1250) = {bg_auc_fluo_window:.1f}')
        
        ax5.axvline(1250, color='purple', linestyle=':', alpha=0.7, label='Fluorescence cutoff (1250 cm^-1)')
        ax5.fill_between(wavenumbers[baseline_bg_mask], baseline_lieberfit[baseline_bg_mask],
                         alpha=0.2, color='orange', label='Baseline Background Region (250-1250 cm^-1)')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax5.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax5.set_ylabel('Intensity', fontsize=12)
        ax5.set_title('Step 8: Fluorescence Calculation - Two Methods Comparison', fontsize=12, fontweight='bold')
        ax5.set_ylim(y_lim_shared)
        ax5.legend(fontsize=8, loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.text(0.02, 0.02, f'Raw M1: {fluorescence_method1:.1f}\nRaw M2: {fluorescence_method2:.1f}\nNorm M1: {fluorescence_method1_normalized:.3f}\nNorm M2: {fluorescence_method2_normalized:.3f}', 
                transform=ax5.transAxes, fontsize=8, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 6: Summary - All Steps Overlaid
        ax6 = axes[2, 1]
        ax6.plot(wavenumbers, intensities_raw, 'b-', label='Raw', alpha=0.4, linewidth=1)
        ax6.plot(wavenumbers, intensities_smooth, 'gray', label='Smoothed', alpha=0.6, linewidth=1)
        ax6.plot(wavenumbers, baseline_lieberfit, 'r--', label='Lieberfit Baseline (original)', linewidth=2)
        ax6.axhline(y=baseline_bg_intensity, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
                    label=f'Baseline Avg (250-1250): {baseline_bg_intensity:.1f}')
        ax6.plot(wavenumbers, baseline_lieberfit_bg_subtracted, 'purple', label='Baseline (bg-subtracted)', linewidth=1.5, linestyle=':')
        ax6.plot(wavenumbers, corrected_spectrum, 'g-', label='Corrected Spectrum', linewidth=1.5)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Plot fitted peaks if available
        # Peaks were fitted on corrected_spectrum, so they're already on the correct scale
        if raman_peak_info and raman_peak_info['fit_params'] is not None:
            # Extend the fit range to cover the whole spectrum for better visualization
            fit_x = np.linspace(wavenumbers[0], wavenumbers[-1], 1000)  # Full spectrum range
            
            if raman_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *raman_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *raman_peak_info['fit_params'])
            
            # Fitted peak is already on corrected_spectrum scale
            ax6.plot(fit_x, fit_y, 'b--', linewidth=1.5, alpha=0.6, label='Raman Peak Fit (Lorentzian)')
            
            # Shade Lorentzian area in summary plot
            baseline_offset = raman_peak_info['fit_params'][3] if len(raman_peak_info['fit_params']) > 3 else 0.0
            raman_fit_mask = (fit_x >= raman_peak_min) & (fit_x <= raman_peak_max)
            ax6.fill_between(
                fit_x[raman_fit_mask],
                baseline_offset,
                fit_y[raman_fit_mask],
                alpha=0.15,
                color='blue',
                label='Raman Area (Lorentzian fit)'
            )
            
            # Peak intensity is already on corrected spectrum scale
            raman_peak_intensity_corrected = raman_peak_intensity
            ax6.plot(raman_peak_wavenumber, raman_peak_intensity_corrected, 'bo', markersize=8, 
                    label=f'Raman ({raman_peak_wavenumber:.1f})')
        elif raman_peak_wavenumber is not None:
            # No fit available, use corrected spectrum value
            raman_peak_idx = np.argmin(np.abs(wavenumbers - raman_peak_wavenumber))
            raman_peak_intensity_corrected = corrected_spectrum[raman_peak_idx]
            ax6.plot(raman_peak_wavenumber, raman_peak_intensity_corrected, 'bo', markersize=8, 
                    label=f'Raman ({raman_peak_wavenumber:.1f})')
        
        # Plot G-band peak if available
        if gband_peak_info and gband_peak_info['fit_params'] is not None:
            # Extend the fit range to cover the whole spectrum for better visualization
            fit_x = np.linspace(wavenumbers[0], wavenumbers[-1], 1000)  # Full spectrum range
            
            if gband_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *gband_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *gband_peak_info['fit_params'])
            
            # Fitted peak is already on corrected_spectrum scale
            ax6.plot(fit_x, fit_y, 'g--', linewidth=1.5, alpha=0.6, label='G-band Peak Fit (Lorentzian)')
            
            # Shade Lorentzian area in summary plot
            baseline_offset = gband_peak_info['fit_params'][3] if len(gband_peak_info['fit_params']) > 3 else 0.0
            gband_fit_mask = (fit_x >= gband_peak_min) & (fit_x <= gband_peak_max)
            ax6.fill_between(
                fit_x[gband_fit_mask],
                baseline_offset,
                fit_y[gband_fit_mask],
                alpha=0.15,
                color='green',
                label='G-band Area (Lorentzian fit)'
            )
            
            # Peak intensity is already on corrected spectrum scale
            gband_peak_intensity_corrected = gband_peak_intensity
            ax6.plot(gband_peak_wavenumber, gband_peak_intensity_corrected, 'go', markersize=8, 
                    label=f'G-band ({gband_peak_wavenumber:.1f})')
        elif gband_peak_wavenumber is not None:
            # No fit available, use corrected spectrum value
            gband_peak_idx = np.argmin(np.abs(wavenumbers - gband_peak_wavenumber))
            gband_peak_intensity_corrected = corrected_spectrum[gband_peak_idx]
            ax6.plot(gband_peak_wavenumber, gband_peak_intensity_corrected, 'go', markersize=8, 
                    label=f'G-band ({gband_peak_wavenumber:.1f})')
        
        ax6.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax6.set_ylabel('Intensity', fontsize=12)
        ax6.set_title('Summary: All Processing Steps', fontsize=12, fontweight='bold')
        ax6.set_ylim(y_lim_shared)
        ax6.legend(fontsize=9, loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save raw intensity plot (unless suppressed)
        output_path = output_dir / f"test_scan_{scan_number}_raw_intensity.png"
        if not suppress_plot:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure without showing
            if verbose:
                print(f"\n[OK] Raw intensity plot saved to: {output_path}")
        else:
            plt.close()  # Close figure without saving
            output_path = None  # Set to None to indicate no plot was saved
    
    # Print comprehensive statistics
    if verbose:
        print("\n=== Statistics ===")
        print(f"Raw spectrum range: [{intensities_raw.min():.1f}, {intensities_raw.max():.1f}]")
        print(f"Smoothed spectrum range: [{intensities_smooth.min():.1f}, {intensities_smooth.max():.1f}]")
        print(f"Lieberfit baseline range: [{baseline_lieberfit.min():.1f}, {baseline_lieberfit.max():.1f}]")
        print(f"Baseline average (250-1250 cm^-1): {baseline_bg_intensity:.1f}")
        print(f"Background-subtracted baseline range: [{baseline_lieberfit_bg_subtracted.min():.1f}, {baseline_lieberfit_bg_subtracted.max():.1f}]")
        print(f"Corrected spectrum range: [{corrected_spectrum.min():.1f}, {corrected_spectrum.max():.1f}]")
        
        if raman_peak_wavenumber is not None:
            print(f"\nRaman Peak ({raman_peak_min}-{raman_peak_max} cm^-1):")
            print(f"  Position: {raman_peak_wavenumber:.2f} cm^-1")
            print(f"  Intensity: {raman_peak_intensity:.2f}")
            if raman_peak_amplitude is not None:
                print(f"  Amplitude: {raman_peak_amplitude:.2f}")
            if raman_peak_width is not None:
                print(f"  Width (FWHM): {raman_peak_width:.2f} cm^-1")
            if raman_peak_area is not None:
                print(f"  Area above baseline (analytical): {raman_peak_area:.2f}")
                if raman_peak_info and 'local_baseline' in raman_peak_info and raman_peak_info['local_baseline'] is not None:
                    print(f"  Local baseline at peak: {raman_peak_info['local_baseline']:.2f}")
                if raman_peak_info and 'area_above_baseline' in raman_peak_info and raman_peak_info['area_above_baseline'] is not None:
                    print(f"  Area above fluorescence baseline: {raman_peak_info['area_above_baseline']:.2f}")
            if raman_peak_info and raman_peak_info['fit_type']:
                fit_type_name = raman_peak_info['fit_type'].capitalize()
                print(f"  Fit method: {fit_type_name} (Lorentzian)")
        
        if gband_peak_wavenumber is not None:
            print(f"\nG-band Peak ({gband_peak_min}-{gband_peak_max} cm^-1):")
            print(f"  Position: {gband_peak_wavenumber:.2f} cm^-1")
            print(f"  Intensity: {gband_peak_intensity:.2f}")
            if gband_peak_amplitude is not None:
                print(f"  Amplitude: {gband_peak_amplitude:.2f}")
            if gband_peak_width is not None:
                print(f"  Width (FWHM): {gband_peak_width:.2f} cm^-1")
            if gband_area is not None:
                print(f"  Area (analytical): {gband_area:.2f}")
            if 'gband_area_trapz' in locals() and gband_area_trapz is not None:
                print(f"  Area (trapezoidal): {gband_area_trapz:.2f}")
            if gband_peak_info and gband_peak_info['fit_type']:
                fit_type_name = gband_peak_info['fit_type'].capitalize()
                print(f"  Fit method: {fit_type_name} (Lorentzian)")
        
        print(f"\nFluorescence (>= 1250 cm^-1) - Two Methods Comparison:")
        print(f"\n  Method 1: AUC of background-subtracted Lieberfit baseline")
        print(f"    Baseline average subtracted (250-1250 cm^-1): {baseline_bg_intensity:.2f}")
        print(f"    Fluorescence (Method 1): {fluorescence_method1:.2f}")
        print(f"\n  Method 2: Raw AUC - G-band area - Background AUC")
        print(f"    Raw AUC (>= 1250 cm^-1): {raw_fluo_auc:.2f}")
        print(f"    G-band area subtracted: {gband_area_to_subtract:.2f}")
        print(f"    Background average (250-1250 cm^-1): {bg_intensity_avg_250_1250:.2f}")
        print(f"    Background AUC in fluorescence window: {bg_auc_fluo_window:.2f}")
        print(f"    Fluorescence (Method 2): {fluorescence_method2:.2f}")
        print(f"\n  Normalized Fluorescent Intensity Analysis:")
        print(f"    Normalization factor (average 250-1250 cm^-1): {bg_intensity_avg_250_1250:.2f}")
        print(f"    Method 1 (Lieberfit baseline AUC): {fluorescence_method1_normalized:.4f}")
        print(f"    Method 2 (Raw - G-band - Background): {fluorescence_method2_normalized:.4f}")
        print(f"    Raw AUC (no correction): {raw_fluo_auc_normalized:.4f}")
        print(f"\n  Comparison:")
        print(f"    Difference (Method 1 vs Method 2): {abs(fluorescence_method1 - fluorescence_method2):.2f}")
        if max(fluorescence_method1, fluorescence_method2) > 0:
            percent_diff = abs(fluorescence_method1 - fluorescence_method2) / max(fluorescence_method1, fluorescence_method2) * 100
            print(f"    Relative difference (Method 1 vs Method 2): {percent_diff:.1f}%")
        print(f"\n  Note: Method 1 is used as primary fluorescence value (fluorescence = {fluorescence:.2f})")
        
        # Print optimization recommendations
        print("\n=== Lieberfit Optimization Tips ===")
        print("To improve the baseline fit, consider:")
        print("1. Run with --optimize flag to automatically find best parameters")
        print("2. Check if baseline follows the fluorescence background smoothly")
        print("3. Ensure baseline doesn't go above the spectrum (should be below)")
        print("4. Verify corrected spectrum has minimal negative values")
        print("5. Check that peaks (Raman, G-band) are preserved after correction")
        print("\nParameter guidelines:")
        print(f"  Current: order={poly_order}, iterations={tot_iter}")
        if poly_order < 5:
            print("  → Consider increasing order (5-7) for more complex backgrounds")
        elif poly_order > 7:
            print("  → Consider decreasing order (5-6) to avoid overfitting")
        if tot_iter < 100:
            print("  → Consider increasing iterations (100-200) for better convergence")
        elif tot_iter > 200:
            print("  → Current iterations may be sufficient; more may not help")
    
    # Create normalized intensity plot (similar to raw plot but using normalized data)
    if not skip_plot_creation and not suppress_plot:
        # Create normalized intensity plot
        fig_norm, axes_norm = plt.subplots(3, 2, figsize=(16, 12))
        fig_norm.suptitle(f'Scan {scan_number} - Normalized Intensity Processing\n'
                         f'DateTime: {datetime_val}, Seconds: {seconds:.1f}', 
                         fontsize=14, fontweight='bold')
        
        # Calculate y-axis range for normalized plots
        all_y_values_norm = []
        all_y_values_norm.extend(intensities_normalized)
        all_y_values_norm.extend(corrected_spectrum_normalized)
        all_y_values_norm.extend(baseline_lieberfit_normalized)
        all_y_values_norm.extend(baseline_lieberfit_bg_subtracted_normalized)
        all_y_values_norm.append(bg_intensity_avg_normalized)
        all_y_values_norm.append(0)
        
        # Add fitted peak values if available
        if raman_peak_info and raman_peak_info['fit_params'] is not None:
            fit_x = np.linspace(raman_peak_info['fit_wavenumbers'][0], 
                               raman_peak_info['fit_wavenumbers'][-1], 200)
            if raman_peak_info['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *raman_peak_info['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *raman_peak_info['fit_params'])
            # Normalize fitted peak values
            fit_y_norm = fit_y / bg_intensity_avg_250_1250 if bg_intensity_avg_250_1250 > 0 else fit_y
            all_y_values_norm.extend(fit_y_norm)
        
        if gband_peak_info_normalized and gband_peak_info_normalized.get('fit_params') is not None:
            fit_x = np.linspace(gband_peak_info_normalized['fit_wavenumbers'][0], 
                               gband_peak_info_normalized['fit_wavenumbers'][-1], 200)
            if gband_peak_info_normalized['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *gband_peak_info_normalized['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *gband_peak_info_normalized['fit_params'])
            all_y_values_norm.extend(fit_y)
        
        y_min_norm = np.min(all_y_values_norm)
        y_max_norm = np.max(all_y_values_norm)
        y_range_norm = y_max_norm - y_min_norm
        y_padding_norm = y_range_norm * 0.05
        y_lim_shared_norm = (y_min_norm - y_padding_norm, y_max_norm + y_padding_norm)
        
        # Plot 1: Normalized Raw vs Smoothed
        ax1_norm = axes_norm[0, 0]
        ax1_norm.plot(wavenumbers, intensities_normalized, 'b-', label='Normalized Spectrum', alpha=0.8, linewidth=1.5)
        ax1_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax1_norm.set_ylabel('Normalized Intensity', fontsize=12)
        ax1_norm.set_title('Step 1-2: Normalized Spectrum', fontsize=12, fontweight='bold')
        ax1_norm.set_ylim(y_lim_shared_norm)
        ax1_norm.legend(fontsize=10)
        ax1_norm.grid(True, alpha=0.3)
        ax1_norm.axvline(250, color='gray', linestyle='--', alpha=0.5, label='Filter cutoff')
        
        # Plot 2: Normalized Lieberfit Baseline
        ax2_norm = axes_norm[0, 1]
        ax2_norm.plot(wavenumbers, intensities_normalized, 'gray', label='Normalized Spectrum', alpha=0.6, linewidth=1)
        ax2_norm.plot(wavenumbers, baseline_lieberfit_normalized, 'r--', label=f'Lieberfit Baseline (order={poly_order}, iter={tot_iter})', linewidth=2)
        ax2_norm.plot(wavenumbers, corrected_spectrum_normalized, 'g-', label='Lieberfit-Corrected Spectrum', linewidth=1.5)
        ax2_norm.axhline(y=bg_intensity_avg_normalized, color='orange', linestyle='--', linewidth=2, label=f'Baseline Avg (250-1250 cm^-1): {bg_intensity_avg_normalized:.4f}')
        ax2_norm.fill_between(wavenumbers[baseline_bg_mask], baseline_lieberfit_normalized[baseline_bg_mask],
                             alpha=0.2, color='orange', label='Baseline Background Region (250-1250 cm^-1)')
        ax2_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax2_norm.set_ylabel('Normalized Intensity', fontsize=12)
        ax2_norm.set_title('Step 4: Normalized Lieberfit Baseline Correction', fontsize=12, fontweight='bold')
        ax2_norm.set_ylim(y_lim_shared_norm)
        ax2_norm.legend(fontsize=9)
        ax2_norm.grid(True, alpha=0.3)
        
        # Plot 3: Peak Detection on Normalized Corrected Spectrum
        ax3_norm = axes_norm[1, 0]
        ax3_norm.plot(wavenumbers, corrected_spectrum_normalized, 'g-', label='Normalized Lieberfit-Corrected Spectrum', linewidth=1.5)
        if raman_peak_wavenumber is not None:
            ax3_norm.axvline(raman_peak_wavenumber, color='blue', linestyle=':', alpha=0.7, 
                           label=f'Raman Peak ({raman_peak_wavenumber:.1f} cm^-1)')
            raman_peak_mask = (wavenumbers >= raman_peak_min) & (wavenumbers <= raman_peak_max)
            ax3_norm.fill_between(wavenumbers[raman_peak_mask], corrected_spectrum_normalized[raman_peak_mask],
                                 alpha=0.15, color='blue', label=f'Raman Region ({raman_peak_min}-{raman_peak_max} cm^-1)')
        if gband_peak_info_normalized:
            gband_wavenumber_norm = gband_peak_info_normalized.get('wavenumber')
            if gband_wavenumber_norm is not None:
                ax3_norm.axvline(gband_wavenumber_norm, color='green', linestyle=':', alpha=0.7, 
                               label=f'G-band Peak ({gband_wavenumber_norm:.1f} cm^-1)')
                gband_mask_norm = (wavenumbers >= gband_peak_min_normalized) & (wavenumbers <= gband_peak_max_normalized)
                ax3_norm.fill_between(wavenumbers[gband_mask_norm], corrected_spectrum_normalized[gband_mask_norm],
                                     alpha=0.15, color='green', label=f'G-band Region ({gband_peak_min_normalized}-{gband_peak_max_normalized} cm^-1)')
        ax3_norm.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax3_norm.set_ylabel('Normalized Intensity', fontsize=12)
        ax3_norm.set_title('Step 5: Peak Detection on Normalized Corrected Spectrum', fontsize=12, fontweight='bold')
        ax3_norm.set_ylim(y_lim_shared_norm)
        ax3_norm.legend(fontsize=9, loc='upper right')
        ax3_norm.grid(True, alpha=0.3)
        
        # Plot 4: Normalized Corrected Spectrum with Fitted Peaks
        ax4_norm = axes_norm[1, 1]
        ax4_norm.plot(wavenumbers, corrected_spectrum_normalized, 'g-', label='Normalized Corrected Spectrum', linewidth=1.5, alpha=0.7)
        
        # Plot G-band peak if available (on normalized spectrum)
        if gband_peak_info_normalized and gband_peak_info_normalized.get('fit_params') is not None:
            fit_x = np.linspace(wavenumbers[0], wavenumbers[-1], 1000)
            if gband_peak_info_normalized['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *gband_peak_info_normalized['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *gband_peak_info_normalized['fit_params'])
            ax4_norm.plot(fit_x, fit_y, 'g--', linewidth=2, alpha=0.8, label='G-band Peak Fit (Lorentzian)')
            baseline_offset = gband_peak_info_normalized['fit_params'][3] if len(gband_peak_info_normalized['fit_params']) > 3 else 0.0
            gband_fit_mask = (fit_x >= gband_peak_min_normalized) & (fit_x <= gband_peak_max_normalized)
            ax4_norm.fill_between(
                fit_x[gband_fit_mask],
                baseline_offset,
                fit_y[gband_fit_mask],
                alpha=0.2,
                color='green',
                label='G-band Peak Area (Lorentzian fit)'
            )
            gband_wavenumber_norm = gband_peak_info_normalized.get('wavenumber')
            gband_intensity_norm = gband_peak_info_normalized.get('intensity')
            if gband_wavenumber_norm is not None and gband_intensity_norm is not None:
                ax4_norm.plot(gband_wavenumber_norm, gband_intensity_norm, 'go', markersize=10, 
                            label=f'G-band Peak ({gband_wavenumber_norm:.1f} cm⁻¹)')
        
        ax4_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax4_norm.set_ylabel('Normalized Intensity', fontsize=12)
        ax4_norm.set_title('Step 5-6: Normalized G-band Peak Detection and Area Calculation', fontsize=12, fontweight='bold')
        ax4_norm.set_ylim(y_lim_shared_norm)
        ax4_norm.legend(fontsize=9, loc='upper right')
        ax4_norm.grid(True, alpha=0.3)
        
        # Plot 5: Normalized Fluorescence Window - Comparison of Two Methods
        ax5_norm = axes_norm[2, 0]
        ax5_norm.plot(wavenumbers, intensities_normalized, 'gray', label='Normalized Spectrum', alpha=0.6, linewidth=1)
        ax5_norm.plot(wavenumbers, baseline_lieberfit_normalized, 'r--', label='Lieberfit Baseline (normalized)', linewidth=2, alpha=0.8)
        ax5_norm.axhline(y=bg_intensity_avg_normalized, color='orange', linestyle='--', linewidth=2, 
                        label=f'Baseline Avg (250-1250 cm^-1): {bg_intensity_avg_normalized:.4f}')
        ax5_norm.plot(wavenumbers, baseline_lieberfit_bg_subtracted_normalized, 'purple', label='Baseline (bg-subtracted)', linewidth=2, alpha=0.8)
        
        # Normalized Method 1: Fill area under background-subtracted Lieberfit baseline
        ax5_norm.fill_between(wavenumbers[fluo_mask], 0, baseline_fluo_normalized,
                             alpha=0.3, color='purple', label=f'Normalized Method 1 AUC = {fluorescence_method1_normalized:.4f}')
        
        # Normalized Method 2: Show raw normalized fluorescence region, G-band, and background
        ax5_norm.fill_between(wavenumbers[fluo_mask], 0, intensities_normalized[fluo_mask],
                             alpha=0.2, color='blue', label=f'Normalized Raw AUC = {raw_fluo_auc_normalized:.4f}')
        
        # Mark G-band area if available
        if gband_area_normalized is not None and gband_area_normalized > 0:
            gband_fluo_overlap_norm = (wavenumbers >= max(1250, gband_peak_min_normalized)) & (wavenumbers <= min(wavenumbers[fluo_mask][-1], gband_peak_max_normalized))
            if np.any(gband_fluo_overlap_norm):
                ax5_norm.fill_between(wavenumbers[gband_fluo_overlap_norm], 0, intensities_normalized[gband_fluo_overlap_norm],
                                     alpha=0.15, color='green', label=f'G-band area (normalized) = {gband_area_normalized:.4f}')
        
        # Show background level in fluorescence region
        ax5_norm.fill_between(wavenumbers[fluo_mask], 0, bg_intensity_avg_normalized,
                             alpha=0.1, color='cyan', label=f'Background AUC (>=1250) = {bg_auc_fluo_window_normalized:.4f}')
        
        ax5_norm.axvline(1250, color='purple', linestyle=':', alpha=0.7, label='Fluorescence cutoff (1250 cm^-1)')
        ax5_norm.fill_between(wavenumbers[baseline_bg_mask], baseline_lieberfit_normalized[baseline_bg_mask],
                             alpha=0.2, color='orange', label='Baseline Background Region (250-1250 cm^-1)')
        ax5_norm.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax5_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax5_norm.set_ylabel('Normalized Intensity', fontsize=12)
        ax5_norm.set_title('Step 8: Normalized Fluorescence Calculation - Two Methods Comparison', fontsize=12, fontweight='bold')
        ax5_norm.set_ylim(y_lim_shared_norm)
        ax5_norm.legend(fontsize=8, loc='upper left')
        ax5_norm.grid(True, alpha=0.3)
        ax5_norm.text(0.02, 0.02, f'Norm M1: {fluorescence_method1_normalized:.4f}\nNorm M2: {fluorescence_method2_normalized:.4f}\nNorm Raw AUC: {raw_fluo_auc_normalized:.4f}', 
                transform=ax5_norm.transAxes, fontsize=8, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 6: Normalized Summary - All Steps Overlaid
        ax6_norm = axes_norm[2, 1]
        ax6_norm.plot(wavenumbers, intensities_normalized, 'b-', label='Normalized', alpha=0.8, linewidth=1.5)
        ax6_norm.plot(wavenumbers, baseline_lieberfit_normalized, 'r--', label='Lieberfit Baseline (normalized)', linewidth=2)
        ax6_norm.axhline(y=bg_intensity_avg_normalized, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
                        label=f'Baseline Avg (250-1250): {bg_intensity_avg_normalized:.4f}')
        ax6_norm.plot(wavenumbers, baseline_lieberfit_bg_subtracted_normalized, 'purple', label='Baseline (bg-subtracted)', linewidth=1.5, linestyle=':')
        ax6_norm.plot(wavenumbers, corrected_spectrum_normalized, 'g-', label='Corrected Spectrum (normalized)', linewidth=1.5)
        ax6_norm.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Plot G-band peak if available
        if gband_peak_info_normalized and gband_peak_info_normalized.get('fit_params') is not None:
            fit_x = np.linspace(wavenumbers[0], wavenumbers[-1], 1000)
            if gband_peak_info_normalized['fit_type'] == 'gaussian':
                fit_y = gaussian(fit_x, *gband_peak_info_normalized['fit_params'])
            else:
                fit_y = lorentzian(fit_x, *gband_peak_info_normalized['fit_params'])
            ax6_norm.plot(fit_x, fit_y, 'g--', linewidth=1.5, alpha=0.6, label='G-band Peak Fit (Lorentzian)')
            baseline_offset = gband_peak_info_normalized['fit_params'][3] if len(gband_peak_info_normalized['fit_params']) > 3 else 0.0
            gband_fit_mask = (fit_x >= gband_peak_min_normalized) & (fit_x <= gband_peak_max_normalized)
            ax6_norm.fill_between(
                fit_x[gband_fit_mask],
                baseline_offset,
                fit_y[gband_fit_mask],
                alpha=0.15,
                color='green',
                label='G-band Area (Lorentzian fit)'
            )
            gband_wavenumber_norm = gband_peak_info_normalized.get('wavenumber')
            gband_intensity_norm = gband_peak_info_normalized.get('intensity')
            if gband_wavenumber_norm is not None and gband_intensity_norm is not None:
                ax6_norm.plot(gband_wavenumber_norm, gband_intensity_norm, 'go', markersize=8, 
                            label=f'G-band ({gband_wavenumber_norm:.1f})')
        
        ax6_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
        ax6_norm.set_ylabel('Normalized Intensity', fontsize=12)
        ax6_norm.set_title('Summary: All Normalized Processing Steps', fontsize=12, fontweight='bold')
        ax6_norm.set_ylim(y_lim_shared_norm)
        ax6_norm.legend(fontsize=9, loc='upper right')
        ax6_norm.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save normalized intensity plot
        output_path_norm = output_dir / f"test_scan_{scan_number}_normalized_intensity.png"
        plt.savefig(output_path_norm, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure without showing
        if verbose:
            print(f"\n[OK] Normalized intensity plot saved to: {output_path_norm}")
    else:
        output_path_norm = None
    
    if verbose:
        print("\n[OK] Test completed successfully!")
    
    return {
        "scan_number": scan_number,
        "datetime": datetime_val,
        "seconds": seconds,
        "raman_peak": raman_peak_info,
        
        # Raw (Original) Fluorescent Intensity Analysis
        "raw": {
            "fluorescence_method1": fluorescence_method1,  # Lieberfit baseline AUC
            "fluorescence_method2": fluorescence_method2,  # Raw AUC - G-band - Background AUC
            "fluorescence_raw_auc": raw_fluo_auc,  # Raw AUC without background correction (>= 1250 cm^-1)
            "gband_peak": gband_peak_info,  # G-band analysis for raw spectrum
        },
        
        # Normalized Fluorescent Intensity Analysis
        "normalized": {
            "fluorescence_method1": fluorescence_method1_normalized,  # Lieberfit baseline AUC on normalized spectrum
            "fluorescence_method2": fluorescence_method2_normalized,  # Raw normalized AUC - G-band - Background AUC
            "fluorescence_raw_auc": raw_fluo_auc_normalized,  # Raw normalized AUC without background correction (>= 1250 cm^-1)
            "gband_peak": gband_peak_info_normalized,  # G-band analysis for normalized spectrum
        },
        
        "baseline_bg_intensity": baseline_bg_intensity,  # Background average (250-1250 cm^-1) for raw spectrum
        "baseline_bg_intensity_normalized": bg_intensity_avg_normalized,  # Background average for normalized spectrum
        "output_dir": str(output_dir),
        "figure_path": str(output_path) if output_path else None,  # Raw intensity plot path
        "figure_path_normalized": str(output_path_norm) if output_path_norm else None,  # Normalized intensity plot path
        "poly_order": poly_order,  # Include parameters used
        "tot_iter": tot_iter,
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test new processing method with Lieberfit optimization')
    parser.add_argument('--profile', type=str, default=DEFAULT_TEST_PROFILE,
                       help=f'Profile name from config (default: {DEFAULT_TEST_PROFILE})')
    parser.add_argument('--scan-number', type=int, default=None,
                       help='Scan number to process (default: 100 or middle scan)')
    parser.add_argument('--optimize', action='store_true', 
                       help='Run parameter optimization to find best Lieberfit parameters')
    parser.add_argument('--order', type=int, default=None,
                       help='Custom polynomial order for Lieberfit (overrides config and optimization)')
    parser.add_argument('--iter', type=int, default=None,
                       help='Custom iteration count for Lieberfit (overrides config and optimization)')
    
    args = parser.parse_args()
    
    test_new_method(
        scan_number=args.scan_number,
        optimize_params=args.optimize,
        custom_order=args.order,
        custom_tot_iter=args.iter,
        profile_name=args.profile
    )


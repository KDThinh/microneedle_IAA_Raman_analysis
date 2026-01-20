"""Command-line interface for SWNT IAA analysis."""

import time
import typer
from pathlib import Path
from typing import Optional

from .pipeline import RamanPipeline
from .io.config import load_profile_config, list_profiles, find_latest_results_folder
from .io.exporter import load_processed_data

app = typer.Typer(
    name="swnt-iaa-analysis",
    help="SWNT IAA Raman Analysis Pipeline",
    add_completion=False
)


@app.command()
def analyze(
    profile: str = typer.Argument(..., help="Profile name from config.yaml"),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file (default: ./config.yaml)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: ./results_<profile>)"
    ),
    algorithm: str = typer.Option(
        "v4",
        "--algorithm",
        "-a",
        help="Algorithm version: 'v3' or 'v4' (default: v4)"
    ),
):
    """
    Run complete Raman analysis pipeline for a single profile.
    
    Examples:
        swnt-iaa-analysis analyze bokchoy_control_6to22_Temp_Hum_Variable_Run1
        swnt-iaa-analysis analyze bokchoy_control_6to22_Temp_Hum_Variable_Run1 --output ./my_results
    """
    # Determine config path
    if config_path is None:
        # Try to find config.yaml in current directory or package
        current_dir = Path.cwd()
        config_path = current_dir / "config.yaml"
        if not config_path.exists():
            # Try package default
            import swnt_iaa_analysis
            package_dir = Path(swnt_iaa_analysis.__file__).parent.parent
            config_path = package_dir / "config.yaml"
            if not config_path.exists():
                typer.echo(f"Error: config.yaml not found. Please specify with --config", err=True)
                raise typer.Exit(1)
    
    config_path = str(Path(config_path).resolve())
    
    if not Path(config_path).exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    # Run pipeline
    pipeline = RamanPipeline(
        config_path=config_path,
        profile_name=profile,
        output_dir=output_dir,
        algorithm=algorithm,
    )
    
    import time
    start_time = time.time()
    try:
        results = pipeline.run()
        elapsed_time = time.time() - start_time
        typer.echo(f"\n✓ Analysis complete! Elapsed time: {elapsed_time:.2f} seconds")
        typer.echo(f"Results saved to: {pipeline.output_dir}")
        return results
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def list_profiles_cmd(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file (default: ./config.yaml)"
    ),
    experiment_type: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Filter by experiment type (e.g., 'in planta', 'in vitro')"
    ),
):
    """
    List available profiles in config.yaml.
    
    Examples:
        swnt-iaa-analysis list-profiles
        swnt-iaa-analysis list-profiles --experiment "in planta"
    """
    # Determine config path
    if config_path is None:
        current_dir = Path.cwd()
        config_path = current_dir / "config.yaml"
        if not config_path.exists():
            import swnt_iaa_analysis
            package_dir = Path(swnt_iaa_analysis.__file__).parent.parent
            config_path = package_dir / "config.yaml"
            if not config_path.exists():
                typer.echo(f"Error: config.yaml not found. Please specify with --config", err=True)
                raise typer.Exit(1)
    
    config_path = str(Path(config_path).resolve())
    
    if not Path(config_path).exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    try:
        profiles = list_profiles(config_path, experiment_type=experiment_type)
        if profiles:
            typer.echo(f"\nAvailable profiles:")
            for profile in profiles:
                typer.echo(f"  - {profile}")
        else:
            typer.echo("No profiles found matching criteria.")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def batch_process(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file"
    ),
    experiment_type: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Process all profiles of this experiment type"
    ),
    profiles: Optional[str] = typer.Option(
        None,
        "--profiles",
        "-p",
        help="Comma-separated list of profile names to process"
    ),
    algorithm: str = typer.Option(
        "v4",
        "--algorithm",
        "-a",
        help="Algorithm version: 'v3' or 'v4'"
    ),
):
    """
    Batch process multiple profiles.
    
    Examples:
        swnt-iaa-analysis batch-process --experiment "in planta"
        swnt-iaa-analysis batch-process --profiles "profile1,profile2"
    """
    # Determine config path
    if config_path is None:
        current_dir = Path.cwd()
        config_path = current_dir / "config.yaml"
        if not config_path.exists():
            import swnt_iaa_analysis
            package_dir = Path(swnt_iaa_analysis.__file__).parent.parent
            config_path = package_dir / "config.yaml"
    
    config_path = str(Path(config_path).resolve())
    
    if not Path(config_path).exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    # Get profiles to process
    if profiles:
        profile_list = [p.strip() for p in profiles.split(",")]
    elif experiment_type:
        profile_list = list_profiles(config_path, experiment_type=experiment_type)
    else:
        typer.echo("Error: Must specify either --profiles or --experiment", err=True)
        raise typer.Exit(1)
    
    if not profile_list:
        typer.echo("No profiles to process.", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Processing {len(profile_list)} profiles...")
    batch_start_time = time.time()
    
    for profile in profile_list:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Processing profile: {profile}")
        typer.echo(f"{'='*60}")
        
        try:
            pipeline = RamanPipeline(
                config_path=config_path,
                profile_name=profile,
                algorithm=algorithm,
            )
            pipeline.run()
            typer.echo(f"✓ Completed: {profile}")
        except Exception as e:
            typer.echo(f"✗ Error processing {profile}: {e}", err=True)
            continue
    
    elapsed_time = time.time() - batch_start_time
    typer.echo(f"\n✓ Batch processing complete! Elapsed time: {elapsed_time:.2f} seconds")


@app.command()
def reprocess(
    profile: str = typer.Argument(..., help="Profile name from config.yaml"),
    steps: Optional[str] = typer.Option(
        None,
        "--steps",
        "-s",
        help="Comma-separated list of steps: spike_removal,baseline_correction,ratios,fft (default: all)"
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file (default: ./config.yaml)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: auto-generated with timestamp)"
    ),
):
    """
    Re-process existing processed_data.csv with selective time-series processing steps.
    
    This command loads the latest processed_data.csv for a profile and allows you to
    re-run selected processing steps (spike removal, baseline correction, ratios, FFT)
    without re-running the time-consuming spectral processing.
    
    Examples:
        swnt_iaa_analysis reprocess Nb_Control_8to24_Temp_Hum_Const_Run2
        swnt_iaa_analysis reprocess Nb_Control_8to24_Temp_Hum_Const_Run2 --steps spike_removal,baseline_correction
        swnt_iaa_analysis reprocess Nb_Control_8to24_Temp_Hum_Const_Run2 --steps baseline_correction,ratios,fft
    """
    # Determine config path
    if config_path is None:
        current_dir = Path.cwd()
        config_path = current_dir / "config.yaml"
        if not config_path.exists():
            import swnt_iaa_analysis
            package_dir = Path(swnt_iaa_analysis.__file__).parent.parent
            config_path = package_dir / "config.yaml"
            if not config_path.exists():
                typer.echo(f"Error: config.yaml not found. Please specify with --config", err=True)
                raise typer.Exit(1)
    
    config_path = str(Path(config_path).resolve())
    
    if not Path(config_path).exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    # Find latest results folder
    typer.echo(f"Finding latest results folder for profile: {profile}...")
    results_folder = find_latest_results_folder(config_path, profile)
    
    if results_folder is None:
        typer.echo(
            f"Error: No results folder found for profile '{profile}'. "
            f"Please run 'analyze' command first to generate processed data.",
            err=True
        )
        raise typer.Exit(1)
    
    # Find processed_data.csv in results folder
    csv_path = results_folder / "processed_data.csv"
    
    if not csv_path.exists():
        typer.echo(
            f"Error: processed_data.csv not found in {results_folder}. "
            f"Please run 'analyze' command first to generate processed data.",
            err=True
        )
        raise typer.Exit(1)
    
    typer.echo(f"Found processed data: {csv_path}")
    
    # Parse steps
    if steps:
        steps_list = [s.strip() for s in steps.split(",")]
        valid_steps = {'spike_removal', 'baseline_correction', 'ratios', 'fft'}
        invalid_steps = [s for s in steps_list if s not in valid_steps]
        if invalid_steps:
            typer.echo(
                f"Error: Invalid steps: {invalid_steps}. "
                f"Valid steps are: {', '.join(sorted(valid_steps))}",
                err=True
            )
            raise typer.Exit(1)
    else:
        steps_list = None
    
    # Initialize pipeline
    pipeline = RamanPipeline(
        config_path=config_path,
        profile_name=profile,
        algorithm='v4',  # Re-processing is only supported for v4
    )
    
    start_time = time.time()
    try:
        results = pipeline.reprocess_from_csv(
            csv_path=csv_path,
            steps=steps_list,
            output_dir=output_dir
        )
        elapsed_time = time.time() - start_time
        typer.echo(f"\n✓ Re-processing complete! Elapsed time: {elapsed_time:.2f} seconds")
        typer.echo(f"Results saved to: {pipeline.output_dir}")
        return results
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


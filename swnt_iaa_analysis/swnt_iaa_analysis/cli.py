"""Command-line interface for SWNT IAA analysis."""

import typer
from pathlib import Path
from typing import Optional

from .pipeline import RamanPipeline
from .io.config import load_profile_config, list_profiles

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
    
    try:
        results = pipeline.run()
        typer.echo(f"\n✓ Analysis complete! Results saved to: {pipeline.output_dir}")
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
    
    typer.echo(f"\n✓ Batch processing complete!")


if __name__ == "__main__":
    app()


"""MBT CLI - Typer-based command-line interface."""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
import json

from mbt.core.compiler import Compiler
from mbt.core.runner import Runner

app = typer.Typer(
    name="mbt",
    help="MBT (Model Build Tool) - Declarative ML pipeline framework",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to create"),
):
    """Initialize a new MBT project."""
    project_path = Path(project_name)

    if project_path.exists():
        console.print(f"[red]Error: Directory {project_name} already exists[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Initializing MBT project: {project_name}[/bold]\n")

    # Create directory structure
    directories = [
        project_path / "pipelines",
        project_path / "includes",
        project_path / "lib",
        project_path / "sample_data",
        project_path / "tests",
        project_path / "target",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        console.print(f"  ✓ Created {directory}")

    # Create pyproject.toml
    pyproject_content = f"""[project]
name = "{project_name}"
version = "0.1.0"
dependencies = [
    "mbt-core>=0.1.0",
]

[tool.mbt]
profile = "{project_name}"
"""
    (project_path / "pyproject.toml").write_text(pyproject_content)
    console.print(f"  ✓ Created pyproject.toml")

    # Create lib/__init__.py
    (project_path / "lib" / "__init__.py").write_text("")
    console.print(f"  ✓ Created lib/__init__.py")

    # Create README.md
    readme_content = f"""# {project_name}

MBT project for ML pipelines.

## Getting Started

```bash
# Compile a pipeline
mbt compile <pipeline_name>

# Run a pipeline
mbt run --select <pipeline_name>
```
"""
    (project_path / "README.md").write_text(readme_content)
    console.print(f"  ✓ Created README.md")

    console.print(f"\n[green]✓ Project {project_name} initialized successfully![/green]")
    console.print(f"\nNext steps:")
    console.print(f"  cd {project_name}")
    console.print(f"  # Create your first pipeline in pipelines/")


@app.command()
def compile(
    pipeline: str = typer.Argument(..., help="Pipeline name (without .yaml extension)"),
    target: str = typer.Option("dev", "--target", "-t", help="Target environment"),
    vars: list[str] = typer.Option([], "--vars", help="Runtime variables (key=value)"),
):
    """Compile pipeline YAML to executable manifest."""
    try:
        project_root = Path.cwd()

        # Parse runtime variables
        runtime_vars = {}
        for var in vars:
            if "=" not in var:
                console.print(f"[red]Error: Invalid --vars format. Use: --vars key=value[/red]")
                raise typer.Exit(code=1)
            key, value = var.split("=", 1)
            runtime_vars[key] = value

        compiler = Compiler(project_root, runtime_vars=runtime_vars)

        console.print(f"[bold]Compiling pipeline: {pipeline}[/bold]")
        console.print(f"Target: {target}")
        if runtime_vars:
            console.print(f"Variables: {runtime_vars}\n")
        else:
            console.print()

        manifest = compiler.compile(pipeline, target)

        console.print(f"[green]✓ Compilation successful![/green]\n")
        console.print(f"Pipeline: {manifest.metadata.pipeline_name}")
        console.print(f"Type: {manifest.metadata.pipeline_type}")
        console.print(f"Steps: {len(manifest.steps)}")
        console.print(f"DAG batches: {len(manifest.dag.execution_batches)}")

    except Exception as e:
        console.print(f"[red]✗ Compilation failed: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def run(
    select: str = typer.Option(..., "--select", help="Pipeline name to run"),
    target: str = typer.Option("dev", "--target", "-t", help="Target environment"),
):
    """Run a compiled pipeline."""
    try:
        project_root = Path.cwd()

        # Load manifest
        manifest_path = project_root / "target" / select / "manifest.json"
        if not manifest_path.exists():
            console.print(f"[red]Error: Manifest not found. Run 'mbt compile {select}' first.[/red]")
            raise typer.Exit(code=1)

        with open(manifest_path) as f:
            from mbt.core.manifest import Manifest
            manifest = Manifest(**json.load(f))

        # Run pipeline
        runner = Runner(manifest, project_root)
        results = runner.run()

        # Print summary
        console.print("\n[bold]Run Summary:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", justify="right")

        for step_name, step_result in results.steps.items():
            status_emoji = "✓" if step_result["status"] == "success" else "✗"
            status_color = "green" if step_result["status"] == "success" else "red"
            table.add_row(
                step_name,
                f"[{status_color}]{status_emoji} {step_result['status']}[/{status_color}]",
                f"{step_result['duration_seconds']:.2f}s",
            )

        console.print(table)

        if results.status == "success":
            console.print(f"\n[green]✓ Pipeline completed successfully![/green]")
        else:
            console.print(f"\n[red]✗ Pipeline failed[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]✗ Execution failed: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def validate(
    pipeline: str = typer.Argument(None, help="Pipeline name to validate (or all if not specified)"),
):
    """Validate pipeline YAML schemas."""
    try:
        project_root = Path.cwd()
        pipelines_dir = project_root / "pipelines"

        if not pipelines_dir.exists():
            console.print("[red]Error: pipelines/ directory not found[/red]")
            raise typer.Exit(code=1)

        # Find pipeline files
        if pipeline:
            pipeline_files = [pipelines_dir / f"{pipeline}.yaml"]
        else:
            pipeline_files = list(pipelines_dir.glob("*.yaml"))
            # Exclude base pipelines (starting with _)
            pipeline_files = [p for p in pipeline_files if not p.stem.startswith("_")]

        console.print(f"[bold]Validating {len(pipeline_files)} pipeline(s)[/bold]\n")

        errors = []
        for pipeline_file in pipeline_files:
            try:
                compiler = Compiler(project_root)
                # Just validate schema, don't save manifest
                import yaml
                with open(pipeline_file) as f:
                    yaml_dict = yaml.safe_load(f)
                    compiler._validate_schema(yaml_dict)

                console.print(f"  [green]✓[/green] {pipeline_file.stem}")
            except Exception as e:
                console.print(f"  [red]✗[/red] {pipeline_file.stem}: {str(e)}")
                errors.append((pipeline_file.stem, str(e)))

        if errors:
            console.print(f"\n[red]Validation failed for {len(errors)} pipeline(s)[/red]")
            raise typer.Exit(code=1)
        else:
            console.print(f"\n[green]✓ All pipelines valid![/green]")

    except Exception as e:
        console.print(f"[red]✗ Validation failed: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def test(
    pipeline: str = typer.Argument(None, help="Pipeline name to test (or all if not specified)"),
):
    """Run DAG assertion tests."""
    try:
        project_root = Path.cwd()
        tests_dir = project_root / "tests"

        if not tests_dir.exists():
            console.print("[yellow]No tests/ directory found. Skipping tests.[/yellow]")
            return

        # Find test files
        if pipeline:
            test_files = [tests_dir / f"{pipeline}.test.yaml"]
        else:
            test_files = list(tests_dir.glob("*.test.yaml"))

        if not test_files:
            console.print("[yellow]No test files found in tests/[/yellow]")
            return

        console.print(f"[bold]Running {len(test_files)} test file(s)[/bold]\n")

        from mbt.testing.assertions import run_assertions
        from mbt.core.compiler import Compiler

        total_assertions = 0
        total_passed = 0
        total_failed = 0

        for test_file in test_files:
            if not test_file.exists():
                console.print(f"[yellow]⚠[/yellow] Test file not found: {test_file}")
                continue

            # Load test specification
            import yaml
            with open(test_file) as f:
                test_spec = yaml.safe_load(f)

            test_pipeline = test_spec.get("pipeline")
            assertions = test_spec.get("assertions", [])

            if not test_pipeline:
                console.print(f"[red]✗[/red] {test_file.stem}: No pipeline specified")
                total_failed += len(assertions)
                continue

            console.print(f"[cyan]{test_file.stem}[/cyan] (pipeline: {test_pipeline})")

            # Compile pipeline to get manifest
            try:
                compiler = Compiler(project_root)
                manifest = compiler.compile(test_pipeline, target="dev")
                manifest_dict = manifest.model_dump()
            except Exception as e:
                console.print(f"  [red]✗[/red] Failed to compile pipeline: {str(e)}")
                total_failed += len(assertions)
                continue

            # Run assertions
            results = run_assertions(manifest_dict, assertions)

            for result in results:
                total_assertions += 1
                if result.passed:
                    total_passed += 1
                    console.print(f"  [green]✓[/green] {result.assertion_type}: {result.message}")
                else:
                    total_failed += 1
                    console.print(f"  [red]✗[/red] {result.assertion_type}: {result.message}")

            console.print()

        # Summary
        if total_assertions == 0:
            console.print("[yellow]No assertions found[/yellow]")
        elif total_failed == 0:
            console.print(f"[green]✓ All {total_assertions} assertion(s) passed![/green]")
        else:
            console.print(
                f"[red]✗ {total_failed}/{total_assertions} assertion(s) failed[/red]"
            )
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗ Test execution failed: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def dag(
    pipeline: str = typer.Argument(..., help="Pipeline name"),
    target: str = typer.Option("dev", "--target", "-t", help="Target environment"),
    mermaid: bool = typer.Option(False, "--mermaid", help="Output Mermaid diagram format"),
):
    """Visualize pipeline DAG."""
    try:
        project_root = Path.cwd()
        manifest_path = project_root / "target" / pipeline / "manifest.json"

        if not manifest_path.exists():
            console.print(f"[red]Error: Manifest not found. Run 'mbt compile {pipeline}' first.[/red]")
            raise typer.Exit(code=1)

        # Load manifest
        with open(manifest_path) as f:
            manifest_dict = json.load(f)

        dag_def = manifest_dict.get("dag", {})
        execution_batches = dag_def.get("execution_batches", [])
        parent_map = dag_def.get("parent_map", {})

        if mermaid:
            # Generate Mermaid diagram
            console.print("```mermaid")
            console.print("graph TD")

            # Add nodes and edges
            for step_name, parents in parent_map.items():
                safe_step = step_name.replace("_", "")
                console.print(f"    {safe_step}[{step_name}]")
                for parent in parents:
                    safe_parent = parent.replace("_", "")
                    console.print(f"    {safe_parent} --> {safe_step}")

            console.print("```")
        else:
            # ASCII visualization
            console.print(f"[bold]DAG for pipeline: {pipeline}[/bold]\n")
            console.print(f"Total steps: {len(parent_map)}")
            console.print(f"Execution batches: {len(execution_batches)}\n")

            console.print("[bold]Execution Order:[/bold]")
            for i, batch in enumerate(execution_batches, 1):
                console.print(f"  Batch {i}: {', '.join(batch)}")

            console.print("\n[bold]Dependencies:[/bold]")
            for step_name, parents in parent_map.items():
                if parents:
                    console.print(f"  {step_name} depends on: {', '.join(parents)}")
                else:
                    console.print(f"  {step_name} (no dependencies)")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗ Failed to visualize DAG: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def debug(
    target: str = typer.Option("dev", "--target", "-t", help="Target environment"),
):
    """Debug connections and configuration for a target."""
    try:
        project_root = Path.cwd()

        console.print(f"[bold]Debugging MBT configuration for target: {target}[/bold]\n")

        # Check project structure
        console.print("[bold]Project Structure:[/bold]")

        required_dirs = ["pipelines", "target"]
        optional_dirs = ["lib", "tests", "sample_data"]

        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                console.print(f"  [green]✓[/green] {dir_name}/")
            else:
                console.print(f"  [red]✗[/red] {dir_name}/ (missing)")

        for dir_name in optional_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                console.print(f"  [green]✓[/green] {dir_name}/")

        # Check Python environment
        console.print("\n[bold]Python Environment:[/bold]")
        import sys
        from mbt import __version__
        console.print(f"  Python: {sys.version.split()[0]}")
        console.print(f"  MBT: {__version__}")

        # Check installed adapters
        console.print("\n[bold]Installed Adapters:[/bold]")
        from mbt.core.registry import PluginRegistry

        registry = PluginRegistry()

        adapter_groups = [
            "mbt.frameworks",
            "mbt.model_registries",
            "mbt.data_connectors",
            "mbt.storage",
        ]

        for group in adapter_groups:
            try:
                adapters = registry.list_group(group)
                if adapters:
                    console.print(f"  {group}:")
                    for adapter in adapters:
                        console.print(f"    [green]✓[/green] {adapter}")
                else:
                    console.print(f"  {group}: [yellow](none)[/yellow]")
            except Exception:
                console.print(f"  {group}: [yellow](none)[/yellow]")

        # Check MLflow connection (if available)
        console.print("\n[bold]MLflow Connection:[/bold]")
        try:
            import mlflow
            tracking_uri = mlflow.get_tracking_uri()
            console.print(f"  Tracking URI: {tracking_uri}")

            # Try to connect
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            console.print(f"  [green]✓[/green] Connected ({len(experiments)} experiments)")
        except ImportError:
            console.print(f"  [yellow]MLflow not installed[/yellow]")
        except Exception as e:
            console.print(f"  [red]✗[/red] Connection failed: {str(e)}")

        console.print("\n[green]✓ Debug check complete[/green]")

    except Exception as e:
        console.print(f"[red]✗ Debug failed: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Show MBT version."""
    from mbt import __version__
    console.print(f"MBT version: {__version__}")


if __name__ == "__main__":
    app()

from __future__ import annotations

from rich.console import Console

from .pipeline import SophiaPipeline


def main() -> None:
    console = Console()
    console.print("[bold cyan]Sophia 🧠 – Interview Question Coach[/bold cyan]")
    console.print("Type 'exit' to quit.\n")

    with console.status("[bold green]Loading Sophia Neural Network... (This may take a moment)[/bold green]", spinner="dots"):
        pipeline = SophiaPipeline()
    
    console.print("[dim]Model loaded successfully.[/dim]\n")

    while True:
        question = console.input("[bold green]Your interview question> [/bold green]").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("\nGood luck with your interviews! 👋")
            break

        try:
            response = pipeline.answer(question)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            console.print("If this is a missing model error, run: [yellow]python -m scripts.train_classifier[/yellow]")
            continue

        console.print("\n" + "=" * 60)
        
        # Format confidence as percentage if it exists
        conf_str = f"{response.confidence_score:.1%}" if response.confidence_score is not None else "N/A"
        lat_str = f"{response.latency_ms}ms" if response.latency_ms is not None else "N/A"
        
        header = (
            f"[bold]Topic:[/bold] {response.topic} | "
            f"[dim]Conf: {conf_str} | Latency: {lat_str}[/dim]"
        )
        console.print(header)
        console.print(f"[dim]Source: {response.source_type}[/dim]")
        
        console.print("\n[bold]Technical answer:[/bold]")
        console.print(response.technical_answer, markup=False)
        console.print("\n[bold]Follow-up question:[/bold]")
        console.print(response.follow_up_question, markup=False)
        console.print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

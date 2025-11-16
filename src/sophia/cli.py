from __future__ import annotations

from rich.console import Console

from .pipeline import SophiaPipeline


def main() -> None:
    console = Console()
    console.print("[bold cyan]Sophia 🧠 – Interview Question Coach[/bold cyan]")
    console.print("Type 'exit' to quit.\n")

    pipeline = SophiaPipeline()

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
        console.print(f"[bold]Topic:[/bold] {response.topic}   [bold]Source:[/bold] {response.source_type}")
        console.print("\n[bold]Technical answer:[/bold]")
        console.print(response.technical_answer)
        console.print("\n[bold]Simple explanation:[/bold]")
        console.print(response.simple_explanation)
        console.print("\n[bold]Follow-up question:[/bold]")
        console.print(response.follow_up_question)
        console.print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

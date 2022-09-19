import typer

from .commands import add_new_model_command

app = typer.Typer()


@app.command()
def add_new_model():
    typer.echo("Almost there! We need to ask you a few questions")
    add_new_model_command()


if __name__ == "__main__":
    app()

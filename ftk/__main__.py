import click
import pytorch_lightning as pl
from ftk.api import SimpleModel


@click.group()
def main():
    pass


@main.command()
@click.option("--fast-dev-run", is_flag=True)
def train(fast_dev_run):
    trainer = pl.Trainer(fast_dev_run=fast_dev_run)
    model = SimpleModel({"lr": 0.8})
    trainer.fit(model)


@main.command("find-lr")
def find_lr():
    trainer = pl.Trainer()
    model = SimpleModel()
    lr_finder = trainer.lr_find(model)
    print(lr_finder.suggestion())
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr.jpg")


if __name__ == "__main__":
    main()

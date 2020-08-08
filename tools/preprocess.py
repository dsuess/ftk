import click
import spacy
from tqdm import tqdm


@click.group()
def main():
    pass


@main.command("split-words")
@click.argument(
    "infile", required=True, type=click.Path(dir_okay=False, exists=True), nargs=-1
)
@click.argument(
    "outfile", required=True, type=click.Path(dir_okay=False, writable=True)
)
@click.option("--model", default="en_core_web_sm")
@click.option("--minlen", default=4)
def split_words(infile, outfile, model, minlen):
    nlp = spacy.load(model)
    nlp.max_length = 1e9
    with open(outfile, "w") as buf_out:
        iterator = tqdm(infile)
        for path in iterator:
            iterator.set_description_str(f"Processing {path}")
            with open(path) as buf_in:
                doc = nlp(buf_in.read())

            for token in doc:
                if token.dep in {0, 445}:  # uknown & punctuation
                    continue
                if len(token.text) < minlen:
                    continue
                buf_out.write(f"{token.text}\n")


if __name__ == "__main__":
    main()


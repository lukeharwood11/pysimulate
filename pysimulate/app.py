from pysimulate.application import App


def main():
    app = App(screen_dim=(800, 800), caption="Example Application", full_screen=True)
    document = app.document

    document.pack()
    app.launch()


if __name__ == "__main__":
    main()

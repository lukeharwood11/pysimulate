from pysimulate.application import App, Initializer

def main():
    app = App(screen_dim=(800, 800), caption="Example Application", full_screen=True)
    app.launch()

if __name__ == "__main__":
    main()
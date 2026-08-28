from pathlib import Path

from fastapi.responses import HTMLResponse

from augur.server import TextGenerator, create_app



def showcase_html() -> str:
    return (Path(__file__).with_name("showcase.html")).read_text()



def create_showcase_app(engine: TextGenerator):
    app = create_app(engine)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def showcase() -> str:
        return showcase_html()

    return app

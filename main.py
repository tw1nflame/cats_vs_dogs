

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import utils

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def upload(request: Request, file: UploadFile = File(...)):
    error = False
    result = None
    try:
        result, image = utils.get_result(file)
    except Exception as ex:
        error = ex
    return templates.TemplateResponse("index.html", {"request": request, 'result': result, "error": error, 'image': image})

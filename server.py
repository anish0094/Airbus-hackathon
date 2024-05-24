from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
import uvicorn
import io
from prediction import read_image, predict, process_image


app = FastAPI()

# @app.get('/')
# def hello_world(name:str):
#     return {'hello': f'{name}'}


@app.post('/post-image')

# async def create_file(file: Annotated[UploadFile, File()]):

#     contents = await file.read()

#     image = process_image(contents)
#     predictions = predict(image)

#     return  {"predictions": predictions} # {
#     #     "file_size": len(file),
#     #     "token": token,
#     #     "fileb_content_type": fileb.content_type,
#     # }
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the file uploaded by the user
        contents = await file.read()
        image = await read_image(io.BytesIO(contents))

        # Apply preprocessing
        image = process_image(image)

        # Make prediction
        predictions = predict(image)

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')

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
async def predict_image(file: UploadFile = File()):
    try:
        # Read the file uploaded by the user
        print(file)
        contents = await file.read()
        image = await read_image(io.BytesIO(contents))
        print(image)

        # Apply preprocessing
        image = process_image(image)

        # Make prediction
        predictions = predict(image)

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')

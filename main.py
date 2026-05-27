import cv2
import numpy as np
from numpy import dot
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BLUR_THRESHOLD       = 100
SIMILARITY_THRESHOLD = 0.5
DET_SIZE             = (640, 640)
MODEL_NAME           = "buffalo_s"

face_app = FaceAnalysis(name=MODEL_NAME, providers=["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DnnExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DET_SIZE)

def decode_image(file_bytes: bytes, label: str) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode {label}. Ensure it is a valid image file."
        )
    return img


def detect_and_align(img: np.ndarray, label: str):
    faces = face_app.get(img)

    if len(faces) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in {label}."
        )
    if len(faces) > 1:
        print(f"[WARN] {label} has {len(faces)} faces. Using most prominent one.")

    face    = faces[0]
    aligned = face_align.norm_crop(img, landmark=face.kps)
    return face, aligned


def get_blur_score(aligned: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_embedding(face, aligned: np.ndarray, label: str) -> np.ndarray:
    blur  = get_blur_score(aligned)
    sharp = blur >= BLUR_THRESHOLD

    if sharp:
        return face.normed_embedding, blur, False

    enhanced = cv2.detailEnhance(aligned)
    enhanced_resized = cv2.resize(enhanced, (640, 640))
    redetected = face_app.get(enhanced_resized)

    if len(redetected) == 0:
        return face.normed_embedding, blur, False

    print(f"[INFO] {label} embedding extracted from enhanced crop.")
    return redetected[0].normed_embedding, blur, True


app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/compare")
async def compare_faces(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image")
):
    bytes1 = await image1.read()
    bytes2 = await image2.read()

    img1 = decode_image(bytes1, label="image1")
    img2 = decode_image(bytes2, label="image2")

    face1, aligned1 = detect_and_align(img1, label="image1")
    face2, aligned2 = detect_and_align(img2, label="image2")

    embed1, blur1, enhanced1 = get_embedding(face1, aligned1, label="image1")
    embed2, blur2, enhanced2 = get_embedding(face2, aligned2, label="image2")

    similarity = float(dot(embed1, embed2))
    match      = similarity >= SIMILARITY_THRESHOLD

    return {
        "match": match,
        "similarity_score" : round(similarity, 4),
        "threshold"        : SIMILARITY_THRESHOLD,

    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
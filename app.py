from fastapi import FastAPI, UploadFile, File
import shutil
from utils import extract_key_frames, check_quality, generate_prompt
import cv2
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API đang chạy"}

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    # Lưu tệp video
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trích xuất khung hình
    frames = extract_key_frames(video_path)
    best_frame = max(frames, key=check_quality)

    # Sinh prompt từ khung hình tốt nhất
    prompt = generate_prompt(best_frame)

    # Chuyển frame sang base64 để trả về
    _, buffer = cv2.imencode(".jpg", best_frame)
    img_bytes = buffer.tobytes()

    os.remove(video_path)  # Xóa file tạm

    return {
        "output": f"( {prompt} )",
        "image_bytes": img_bytes.hex()  # trả về dạng hex để dễ lưu trữ
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

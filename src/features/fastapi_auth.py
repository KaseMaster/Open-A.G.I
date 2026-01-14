from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import hashlib
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

@app.get("/auth/session/verify")
async def verify_session(token: str = Query(...)):
    try:
        if len(token) > 10:
            address = "0x" + hashlib.sha256(token.encode()).hexdigest()[:40]
            return {
                "ok": True,
                "address": address,
                "timestamp": int(time.time())
            }
        else:
            return {"ok": False, "error": "Invalid token"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "openagi-auth"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8182)
from pathlib import Path
from threading import Thread,Lock
from uuid import uuid4
import json
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from howvision.config import load_config
from howvision.core.fs import safe_name,resolve_under,IMAGES,VIDEOS
from howvision.pipeline.run import execute
_jobs={}; _lock=Lock()
def create_app(config="configs/default.yaml"):
    cfg=load_config(config); data=Path(cfg["paths"]["data"]); out=Path(cfg["paths"]["output"]); model=Path(cfg["paths"]["hand_model"]); data.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True); app=FastAPI(title="HOW Vision",version="0.2.0")
    @app.get("/api/health")
    def health():
        try: import mediapipe; dep=True
        except: dep=False
        return {"status":"READY" if dep and model.exists() else "SETUP_REQUIRED","dependencies_ready":dep,"model_ready":model.exists(),"model_size":model.stat().st_size if model.exists() else 0,"active_model":"MediaPipe Hand Landmarker float16 v1 + Rule v1"}
    @app.post("/api/input/upload")
    async def upload(file:UploadFile=File(...)):
        name=safe_name(file.filename or ""); p=resolve_under(data,name)
        if p.suffix.lower() not in IMAGES|VIDEOS: raise HTTPException(400,"unsupported extension")
        with p.open("wb") as target:
            while chunk:=await file.read(1024*1024): target.write(chunk)
        return {"name":name,"size":p.stat().st_size}
    @app.get("/api/input")
    def inputs(): return [{"name":p.name,"size":p.stat().st_size,"type":"video" if p.suffix.lower() in VIDEOS else "image"} for p in sorted(data.iterdir()) if p.is_file()]
    @app.post("/api/runs")
    def start(body:dict):
        p=resolve_under(data,body.get("input",""))
        if not p.exists(): raise HTTPException(404,"input not found")
        job_id=f"job_{uuid4().hex[:10]}"; _jobs[job_id]={"job_id":job_id,"status":"QUEUED","stage":"QUEUED","progress":0,"input":p.name}
        def update(event):
            with _lock: _jobs[job_id].update(event)
        def work():
            try:
                update({"status":"RUNNING","stage":"INITIALIZING"}); rid,rd,metrics=execute(cfg,p,update); update({"status":"COMPLETED","stage":"COMPLETED","progress":1,"run_id":rid,"metrics":metrics})
            except Exception as exc: update({"status":"FAILED","stage":"FAILED","error":str(exc)})
        Thread(target=work,daemon=True).start(); return _jobs[job_id]
    @app.get("/api/jobs")
    def jobs(): return list(_jobs.values())
    @app.get("/api/jobs/{job_id}")
    def job(job_id:str):
        if job_id not in _jobs: raise HTTPException(404,"job not found")
        return _jobs[job_id]
    @app.get("/api/runs")
    def runs():
        items=[]
        for d in sorted(out.glob("run_*"),reverse=True):
            try: items.append(json.loads((d/"manifest.json").read_text()))
            except: pass
        return items
    @app.get("/api/runs/{run_id}/frames/{frame_id}")
    def frame(run_id:str,frame_id:int):
        if not run_id.startswith("run_") or not run_id[4:].isdigit(): raise HTTPException(400,"invalid run id")
        p=out/run_id/"overlays"/f"frame_{frame_id:06d}.jpg"
        if not p.exists(): raise HTTPException(404,"frame not found")
        return FileResponse(p)
    @app.get("/api/runs/{run_id}/csv")
    def csv(run_id:str):
        p=out/run_id/"predictions.csv"
        if not p.exists(): raise HTTPException(404,"not found")
        return FileResponse(p,filename=f"{run_id}_predictions.csv")
    app.mount("/",StaticFiles(directory="frontend",html=True),name="frontend"); return app

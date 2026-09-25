from pathlib import Path
import csv,json,logging,platform,subprocess,sys,time,cv2,yaml
from howvision import __version__
from howvision.types import *
from howvision.pipeline.source import frames
from howvision.core.fs import allocate_run,sha256,atomic_json
from howvision.perception.hands import MediaPipeTaskHandDetector
from howvision.perception.wheel import GeometricWheelDetector
from howvision.classifiers.rule import ContactFeatureExtractor,RuleBasedHOWClassifier
from howvision.temporal.engine import TemporalDecisionEngine
from howvision.rendering.overlay import OverlayRenderer
CSV_COLUMNS=["run_id","source_file","frame_id","sequence_index","timestamp_ms","timestamp_native","left_hand_visible","right_hand_visible","left_hand_on","right_hand_on","how_state","how_status","confidence","wheel_visible","wheel_confidence","model_version","decision_version"]
def execute(cfg,input_path,progress=None,hand_detector=None,wheel_detector=None):
    rid,rd=allocate_run(Path(cfg["paths"]["output"])); (rd/"overlays").mkdir(); (rd/"logs").mkdir(); yaml.safe_dump({k:v for k,v in cfg.items() if not k.startswith("_")},(rd/"config.yaml").open("w")); logging.basicConfig(filename=rd/"logs/run.log",level=logging.INFO,force=True); started=time.time()
    model=Path(cfg["paths"]["hand_model"]); manifest={"run_id":rid,"status":"RUNNING","creation_time":started,"start_time":started,"input_files":[input_path.name],"input_hashes":{input_path.name:sha256(input_path)},"input_type":input_path.suffix.lower(),"frame_count":0,"processed_frame_count":0,"failed_frame_count":0,"model_version":"mediapipe-hand-landmarker-f16-v1+rule-v1","model_hash":sha256(model) if model.exists() else None,"hand_detector":"MediaPipeTaskHandDetector","wheel_detector":"GeometricWheelDetector","classifier":"RuleBasedHOWClassifier","temporal_configuration":cfg["temporal"],"application_version":__version__,"git_commit":_git(),"python_version":sys.version,"platform":platform.platform(),"gpu_information":{"cuda_available":False},"codec":None}; atomic_json(rd/"manifest.json",manifest)
    hd=hand_detector or MediaPipeTaskHandDetector(cfg["hand_detection"],model); wd=wheel_detector or GeometricWheelDetector(cfg["wheel_detection"]); extractor=ContactFeatureExtractor(); classifier=RuleBasedHOWClassifier(cfg["classifier"]); temporal=TemporalDecisionEngine(cfg["temporal"]); renderer=OverlayRenderer(); rows=[]; timings=[]; writer=None
    try:
        source=list(frames(input_path)); total_frames=len(source)
        with (rd/"predictions.jsonl").open("w",encoding="utf-8") as jf:
            for fr in source:
                t=time.monotonic()
                try:
                    hands=hd.predict(fr); wheel=wd.predict(fr); features=extractor.extract(hands,wheel); raw=classifier.predict(features); smooth=temporal.process(raw); left=next((h for h in hands if h.side==HandSide.LEFT),None); right=next((h for h in hands if h.side==HandSide.RIGHT),None); result=FrameResult(rid,fr.source_file,fr.frame_id,fr.sequence_index,fr.timestamp_ms,fr.timestamp_native,left,right,wheel,features,raw,smooth,to_status(smooth.state),smooth.confidence); overlay=renderer.render(fr.image,result); cv2.imwrite(str(rd/"overlays"/f"frame_{fr.frame_id:06d}.jpg"),overlay)
                    if fr.media_type=="video":
                        if writer is None:
                            cap=cv2.VideoCapture(str(input_path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25.; cap.release(); codec=cfg["rendering"].get("codec","mp4v"); writer=cv2.VideoWriter(str(rd/"output.mp4"),cv2.VideoWriter_fourcc(*codec),fps,(fr.width,fr.height)); manifest["codec"]=codec
                        writer.write(overlay)
                    else: cv2.imwrite(str(rd/f"output{input_path.suffix.lower()}"),overlay)
                    jf.write(json.dumps(result.dict(),default=str)+"\n"); rows.append(_row(result)); manifest["processed_frame_count"]+=1
                    if progress: progress({"status":"RUNNING","stage":"INFERENCE","run_id":rid,"current_frame":fr.frame_id,"processed":manifest["processed_frame_count"],"total":total_frames,"progress":manifest["processed_frame_count"]/max(total_frames,1),"fps":manifest["processed_frame_count"]/max(time.time()-started,.001),"how_status":result.final_status.value,"confidence":result.confidence,"preview":f"/api/runs/{rid}/frames/{fr.frame_id}"})
                except Exception: manifest["failed_frame_count"]+=1; logging.exception("frame failure %s",fr.frame_id)
                manifest["frame_count"]+=1; timings.append(time.monotonic()-t)
        if writer: writer.release()
        with (rd/"predictions.csv").open("w",newline="",encoding="utf-8") as f: out=csv.DictWriter(f,fieldnames=CSV_COLUMNS); out.writeheader(); out.writerows(rows)
        elapsed=time.time()-started; metrics={"total_processing_time":elapsed,"average_fps":len(rows)/elapsed if elapsed else 0,"peak_fps":1/min(timings) if timings else 0,"frames_processed":len(rows),"frames_failed":manifest["failed_frame_count"],"frames_skipped":0}; atomic_json(rd/"metrics.json",metrics); manifest.update(status="COMPLETED",completion_time=time.time()); atomic_json(rd/"manifest.json",manifest); return rid,rd,metrics
    except Exception as exc:
        if writer: writer.release()
        manifest.update(status="FAILED",completion_time=time.time(),error=str(exc)); atomic_json(rd/"manifest.json",manifest); raise
def _row(r):
    state=r.temporal_prediction.state; values=[r.run_id,r.source_file,r.frame_id,r.sequence_index,r.timestamp_ms,json.dumps(r.timestamp_native),bool(r.left_hand),bool(r.right_hand),state in {HOWState.LEFT_ON,HOWState.BOTH_ON},state in {HOWState.RIGHT_ON,HOWState.BOTH_ON},state.value,r.final_status.value,r.confidence,r.wheel.visible,r.wheel.confidence,r.model_version,r.decision_version]; return dict(zip(CSV_COLUMNS,values))
def _git():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except:return None

"""Local read-only training dashboard and lightweight Trainer logging callback."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from project_config import load_config, runtime_paths


class RunLog:
    """Persist a current snapshot atomically and append chart history."""

    def __init__(self, directory):
        root = Path(directory)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{time.time_ns()}"
        self.directory = root / "runs" / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        pointer = root / "current_run.json.tmp"
        pointer.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        pointer.replace(root / "current_run.json")
        self.state = {"status": "ready", "run_id": run_id, "step": 0, "total_steps": 0,
                      "epoch": 0, "metrics": {}, "task_accuracy": None}

    def update(self, event, **values):
        self.state.update(values)
        self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(self.state, allow_nan=False)
        temporary = self.directory / "status.json.tmp"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.directory / "status.json")
        record = {**self.state, "event": event}
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")


def create_training_callback(directory, train_examples, model, max_steps=0, run_log=None):
    """Import training libraries only inside the notebook, never in the web server."""
    from transformers import TrainerCallback
    import torch

    class TrainingMonitor(TrainerCallback):
        def __init__(self):
            self.log = run_log or RunLog(directory)
            self.started = None
            self.microbatches = 0
            self.parameters = {
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "stored_parameter_elements": sum(p.numel() for p in model.parameters()),
            }
            # Quantized storage does not expose the original logical parameter count.
            self.log.update("ready", status="ready", train_examples=train_examples, parameters=self.parameters,
                            total_steps=max_steps)

        def snapshot(self, args, state, event, **extra):
            if not state.is_world_process_zero:
                return
            elapsed = time.monotonic() - self.started if self.started is not None else 0
            step = state.global_step
            total = state.max_steps
            remaining = elapsed / step * max(0, total - step) if step else None
            free, memory = torch.cuda.mem_get_info()
            values = {
                "step": step, "total_steps": total, "epoch": state.epoch or 0,
                "elapsed_seconds": elapsed, "eta_seconds": remaining,
                "microbatches": self.microbatches,
                "processed_examples_estimate": self.microbatches * args.per_device_train_batch_size * args.world_size,
                "input_tokens_seen": getattr(state, "num_input_tokens_seen", 0),
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "device_vram_used_gib": (memory - free) / 1024**3,
                "device_vram_total_gib": memory / 1024**3,
                "torch_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                **extra,
            }
            self.log.update(event, **values)

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            self.snapshot(args, state, "start", status="training")

        def on_substep_end(self, args, state, control, **kwargs):
            self.microbatches += 1

        def on_step_end(self, args, state, control, **kwargs):
            self.microbatches += 1
            self.snapshot(args, state, "step", status="training")

        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics = dict(self.log.state["metrics"])
            logged = {}
            for name, value in (logs or {}).items():
                if isinstance(value, (int, float)):
                    logged[name] = value if math.isfinite(value) else None
            metrics.update(logged)
            self.snapshot(args, state, "log", metrics=metrics, logged_metrics=logged)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.snapshot(args, state, "evaluation", status="validation_complete")

        def on_train_end(self, args, state, control, **kwargs):
            self.snapshot(args, state, "end", status="training_complete", eta_seconds=0)

        def set_status(self, status, **values):
            self.log.update("status", status=status, **values)

    return TrainingMonitor()


def recover_completed_run(directory, history, train_metrics, evaluation_metrics,
                          step, total_steps, epoch, parameters=None, memory=None):
    """Display actual Trainer history when a completed run had no callback.

    Missing per-step timings and token/microbatch counts stay unreported.
    This function neither loads nor trains a model.
    """
    log = RunLog(directory)
    metrics = {}
    for entry in history:
        logged = {key: value if math.isfinite(value) else None
                  for key, value in entry.items()
                  if key not in ("step", "epoch") and isinstance(value, (int, float))}
        metrics.update(logged)
        log.update("log", status="recovering_completed_run",
                   step=entry.get("step", step), total_steps=total_steps,
                   epoch=entry.get("epoch", epoch), metrics=dict(metrics),
                   logged_metrics=logged, recovered_from_trainer_history=True)
    metrics.update(evaluation_metrics)
    log.update("recovered", status="training_complete_recovered",
               step=step, total_steps=total_steps, epoch=epoch,
               elapsed_seconds=train_metrics.get("train_runtime"), eta_seconds=0,
               metrics=metrics, parameters=parameters or {},
               **(memory or {}))
    return log


def read_run(directory):
    directory = Path(directory)
    pointer = directory / "current_run.json"
    if pointer.exists():
        run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
        if not isinstance(run_id, str) or not run_id or any(c not in "0123456789T-" for c in run_id):
            raise ValueError("Invalid run ID")
        directory = directory / "runs" / run_id
    current = {"status": "waiting", "step": 0, "total_steps": 0,
               "metrics": {}, "task_accuracy": None}
    status_file = directory / "status.json"
    if status_file.exists():
        current = json.loads(status_file.read_text(encoding="utf-8"))
    points = []
    history = directory / "history.jsonl"
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrent append may still be completing the final line.
            if row.get("event") == "log":
                points.append({"step": row["step"], "metrics": row.get("logged_metrics", {})})
    evaluation = directory / "task_evaluation.json"
    if evaluation.exists():
        current["task_evaluation"] = json.loads(evaluation.read_text(encoding="utf-8"))
    return {"state": current, "points": points}


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QLoRA Training Monitor</title><style>
*{box-sizing:border-box}body{margin:0;background:#10151f;color:#e5edf8;font:15px system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:28px}h1{margin:0;font-size:28px}h2{font-size:17px}
.muted{color:#9bacbf}.top{display:flex;justify-content:space-between;align-items:center;gap:20px}
.badge{background:#21324b;color:#92c9ff;padding:8px 14px;border-radius:20px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}
.card,.panel{background:#192231;border:1px solid #2a374b;border-radius:12px;padding:18px}
.label{color:#9bacbf;font-size:13px}.value{font-size:25px;margin-top:8px;font-variant-numeric:tabular-nums}
.bar{height:12px;background:#27344a;border-radius:8px;overflow:hidden;margin:18px 0 8px}
#bar{height:100%;background:#72b7ff;transition:width .4s;width:0%}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}canvas{width:100%;height:240px}
.legend{font-size:12px;color:#9bacbf}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#ffacac}
.notice{line-height:1.6;margin-top:20px}.blue{color:#72b7ff}.green{color:#7ae0b3}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}}
</style></head><body><main>
<div class="top"><div><h1>QLoRA Training Monitor</h1><p class="muted">Local, read-only dashboard · refreshes every 2 seconds</p></div><span id="status" class="badge">Waiting</span></div>
<div class="bar"><div id="bar"></div></div><div id="progress" class="muted">Waiting for the notebook to start a run.</div>
<div class="grid">
<div class="card"><div class="label">Optimizer updates</div><div class="value" id="steps">—</div></div>
<div class="card"><div class="label">Epoch / passes through training data</div><div class="value" id="epoch">—</div></div>
<div class="card"><div class="label">Training loss</div><div class="value" id="loss">—</div></div>
<div class="card"><div class="label">Validation loss</div><div class="value" id="eval">—</div></div>
<div class="card"><div class="label">Gradient norm (backpropagation)</div><div class="value" id="grad">—</div></div>
<div class="card"><div class="label">Learning rate</div><div class="value" id="lr">—</div></div>
<div class="card"><div class="label">Device VRAM used / total</div><div class="value" id="vram">—</div></div>
<div class="card"><div class="label">Peak PyTorch allocation</div><div class="value" id="peak">—</div></div>
<div class="card"><div class="label">Elapsed / estimated remaining</div><div class="value" id="time">—</div></div>
<div class="card"><div class="label">Microbatches / estimated examples processed</div><div class="value" id="samples">—</div></div>
<div class="card"><div class="label">Input tokens processed</div><div class="value" id="tokens">—</div></div>
<div class="card"><div class="label">Trainable adapter parameters</div><div class="value" id="parameters">—</div></div>
</div>
<div class="charts"><div class="panel"><h2>Loss by optimizer step</h2><canvas id="lossChart"></canvas><span class="legend"><span class="blue">● Training</span> &nbsp; <span class="green">● Validation</span></span></div>
<div class="panel"><h2>Gradient norm by optimizer step</h2><canvas id="gradChart"></canvas><span class="legend">Gradient norm is not a percentage of knowledge learned.</span></div></div>
<div class="panel notice"><h2>Held-out factual accuracy</h2><p id="accuracy">Not evaluated. Training loss and token prediction accuracy are not factual task correctness.</p>
<p class="muted">A progress percentage describes planned optimizer updates, not how much knowledge the model acquired. Keep final-test questions out of training and tuning.</p>
<div id="saved" class="muted"></div><pre id="error"></pre></div>
<p class="muted" id="updated"></p>
</main><script>
const $=id=>document.getElementById(id);
const num=(v,n=4)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';
const integer=v=>typeof v==='number'?v.toLocaleString():'—';
const duration=v=>typeof v==='number'?Math.floor(v/60)+'m '+Math.floor(v%60)+'s':'—';
function chart(id,points,series){
 const canvas=$(id),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=240;
 canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);
 const all=series.flatMap(s=>points.filter(p=>typeof p.metrics[s.key]==='number').map(p=>({x:p.step,y:p.metrics[s.key]})));
 if(!all.length){c.fillStyle='#9bacbf';c.font='14px system-ui';c.fillText('No logged values yet.',20,35);return;}
 const maxX=Math.max(1,...all.map(p=>p.x)),minY=Math.min(...all.map(p=>p.y)),maxY=Math.max(...all.map(p=>p.y));
 const range=maxY-minY||Math.max(Math.abs(maxY)*.1,.1),left=55,right=15,top=15,bottom=30;
 const x=v=>left+v/maxX*(w-left-right),y=v=>top+(maxY-v)/range*(h-top-bottom);
 c.font='11px system-ui';c.strokeStyle='#2a374b';c.fillStyle='#9bacbf';
 for(let i=0;i<5;i++){const yy=top+i*(h-top-bottom)/4;c.beginPath();c.moveTo(left,yy);c.lineTo(w-right,yy);c.stroke();c.fillText((maxY-i*range/4).toFixed(3),0,yy+4);}
 c.fillText('Step 0',left,h-5);c.fillText('Step '+maxX,Math.max(left,w-85),h-5);
 for(const s of series){const entries=new Map();for(const p of points){if(typeof p.metrics[s.key]==='number')entries.set(p.step,p.metrics[s.key]);}
 c.strokeStyle=s.color;c.fillStyle=s.color;c.lineWidth=2;c.beginPath();let first=true;
 for(const [step,value] of entries){if(first)c.moveTo(x(step),y(value));else c.lineTo(x(step),y(value));first=false;}c.stroke();
 for(const [step,value] of entries){c.beginPath();c.arc(x(step),y(value),3,0,Math.PI*2);c.fill();}}
}
async function refresh(){try{
 const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error('Dashboard request failed');
 const data=await response.json(),s=data.state,m=s.metrics||{},total=s.total_steps||0,step=s.step||0;
 $('status').textContent=(s.status||'waiting').replaceAll('_',' ');
 $('steps').textContent=total?step+' / '+total:'—';$('bar').style.width=(total?Math.min(100,step/total*100):0)+'%';
 $('progress').textContent=total?(step/total*100).toFixed(1)+'% of planned updates completed':'Waiting for the notebook to start a run.';
 $('epoch').textContent=num(s.epoch,3);$('loss').textContent=num(m.loss);$('eval').textContent=num(m.eval_loss);
 $('grad').textContent=num(m.grad_norm);$('lr').textContent=typeof m.learning_rate==='number'?m.learning_rate.toExponential(2):'—';
 $('vram').textContent=s.device_vram_total_gib?num(s.device_vram_used_gib,2)+' / '+num(s.device_vram_total_gib,2)+' GiB':'—';
 $('peak').textContent=typeof s.torch_peak_allocated_gib==='number'?num(s.torch_peak_allocated_gib,2)+' GiB':'—';
 $('time').textContent=duration(s.elapsed_seconds)+' / '+duration(s.eta_seconds);
 $('samples').textContent=integer(s.microbatches)+' / '+integer(s.processed_examples_estimate);
 $('tokens').textContent=integer(s.input_tokens_seen);$('parameters').textContent=integer(s.parameters?.trainable);
 const evaluation=s.task_evaluation;
 $('accuracy').textContent=evaluation&&Number.isInteger(evaluation.correct)&&evaluation.total>0?
 (evaluation.correct/evaluation.total*100).toFixed(1)+'% ('+evaluation.correct+' / '+evaluation.total+') — '+(evaluation.method||'See evaluation report'):
 'Not evaluated. Training loss and token prediction accuracy are not factual task correctness.';
 $('saved').textContent=s.adapter_path?'Saved adapter: '+s.adapter_path:'';$('error').textContent=s.error||'';
 $('updated').textContent=s.updated_utc?'Last log update: '+s.updated_utc:'';
 chart('lossChart',data.points,[{key:'loss',color:'#72b7ff'},{key:'eval_loss',color:'#7ae0b3'}]);
 chart('gradChart',data.points,[{key:'grad_norm',color:'#ffbd73'}]);
}catch(error){$('error').textContent=error.message;}}
refresh();setInterval(refresh,2000);
</script></body></html>"""


def make_handler(directory):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, content_type = HTML.encode(), "text/html; charset=utf-8"
            elif self.path == "/api/status":
                try:
                    body = json.dumps(read_run(directory), allow_nan=False).encode()
                except (OSError, ValueError) as exc:
                    self.send_error(500, str(exc))
                    return
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(ROOT, args.config)
    args.port = args.port if args.port is not None else config["dashboard_port"]
    args.log_dir = args.log_dir or runtime_paths(config, ROOT)["output"] / "monitor"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.log_dir))
    print(f"Training dashboard: http://127.0.0.1:{args.port}", flush=True)
    print("Read-only monitor; opening it does not start training. Ctrl+C stops the dashboard.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

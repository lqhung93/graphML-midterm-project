"""Check that a chunked run carries its full training state across sessions.

Bit-identical continuation is impossible here: the negative sampler's RNG lives
in Base.so with no accessor, so its stream restarts each session. What resume
must guarantee is that everything we CAN carry survives the round-trip intact --
above all the Adam moments, whose loss is what makes naive warm-starting jolt
the learning rate at every seam.

Run: python test_resume.py     (MKG-Y, a few epochs, CPU is fine)
"""
import os
import subprocess
import sys
import tempfile

import torch

DS = "MKG-Y"
COMMON = ["-dataset=" + DS, "-batch_size=1024", "-margin=4", "-dim=200",
          "-mu=0.01", "-neg_num=128", "-learning_rate=2e-5"]


def run(args):
    r = subprocess.run([sys.executable, "run_adamf_mat.py"] + COMMON + args,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def main():
    d = tempfile.mkdtemp(prefix="resume-")
    state = os.path.join(d, "resume_state.pth")

    # session 1: 2 of 4 epochs, writes the resume state
    out = run(["-epoch=2", "-total_epoch=4", "-save=" + d + "/s1",
               "-resume_out=" + state])
    assert "Skipping evaluation" in out, "a partial run must not evaluate"
    assert os.path.exists(state), "session 1 wrote no resume state"

    a = torch.load(state, map_location="cpu", weights_only=False)
    assert a["epochs_done"] == 2, a["epochs_done"]
    assert a["optimizer"]["state"], "Adam moments for D were not saved"
    assert a["optimizer_g"]["state"], "Adam moments for G were not saved"

    # session 2: the remaining 2 epochs, resuming
    out = run(["-epoch=2", "-total_epoch=4", "-save=" + d + "/s2",
               "-resume=" + state, "-resume_out=" + d + "/state2.pth"])
    assert "Resumed from" in out, "session 2 did not load the resume state"
    assert "Skipping evaluation" not in out, "the final session must evaluate"
    assert "averaged(filter)" in out, "no link-prediction result reported"

    b = torch.load(d + "/state2.pth", map_location="cpu", weights_only=False)
    assert b["epochs_done"] == 4, b["epochs_done"]

    # the resumed session must CONTINUE, not restart: Adam's step counter keeps rising
    step_a = next(iter(a["optimizer"]["state"].values()))["step"]
    step_b = next(iter(b["optimizer"]["state"].values()))["step"]
    assert step_b > step_a, "Adam step counter did not advance past the seam (%s -> %s)" % (step_a, step_b)

    # and the weights must have moved on from where session 1 stopped
    w_a = a["model"]["ent_embeddings.weight"]
    w_b = b["model"]["ent_embeddings.weight"]
    assert not torch.equal(w_a, w_b), "session 2 did not train"

    print("OK: state round-trips, Adam step %s -> %s, %d -> %d epochs"
          % (step_a, step_b, a["epochs_done"], b["epochs_done"]))


if __name__ == "__main__":
    main()

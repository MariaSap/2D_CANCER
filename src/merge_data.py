import os, shutil, glob

def merge_real_synth(real_root, synth_root, out_root):
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    classes = sorted(os.listdir(real_root))
    for cls in classes:
        dst = os.path.join(out_root, cls)
        os.makedirs(dst, exist_ok=True)
        for p in glob.glob(os.path.join(real_root, cls, "*")):
            shutil.copy(p, dst)
        synth_cls = os.path.join(synth_root, cls)
        if os.path.exists(synth_cls):
            for p in glob.glob(os.path.join(synth_cls), "*"):
                shutil.copy(p,dst)
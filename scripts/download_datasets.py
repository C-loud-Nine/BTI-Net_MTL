"""Downloads BUSI, BRISC and HAM10000 into data/raw/ and checks them against the dataset descriptions and splits/.

    python scripts/download_datasets.py

Kaggle sources need a Kaggle API token (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."""
import base64, csv, json, os, re, time, zipfile, shutil, urllib.request, urllib.error, urllib.parse


ROOT = os.environ.get("BTINET_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
DOWNLOADS = os.path.join(RAW, "_downloads")
STAGING = os.path.join(RAW, "_staging")
RUN_ID = time.strftime("%Y%m%d_%H%M%S")


SOURCES = {
    "BUSI": [("kaggle", "sabahesaraki/breast-ultrasound-images-dataset"),
             ("kaggle", "subhajournal/busi-breast-ultrasound-images-dataset"),
             ("kaggle", "aryashah2k/breast-ultrasound-images-dataset")],
    "HAM10000": [("dataverse", "doi:10.7910/DVN/DBW86T"),
                 ("kaggle+kaggle", "kmader/skin-cancer-mnist-ham10000|surajghuwalewala/ham1000-segmentation-and-classification")],
    "BRISC": [("kaggle", "briscdataset/brisc2025")],
}
UA = {"User-Agent": "btinet-download"}


def log(msg):
    print(time.strftime("%H:%M:%S") + " | " + msg, flush=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

_OPENER = urllib.request.build_opener(_NoRedirect)

def http_open(url, headers=None, timeout=120):
    h = dict(UA, **(headers or {}))
    first_host = urllib.parse.urlparse(url).netloc
    for _ in range(10):
        if urllib.parse.urlparse(url).netloc != first_host:
            h = {k: v for k, v in h.items() if k.lower() != "authorization"}
        try:
            return _OPENER.open(urllib.request.Request(url, headers=h), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
    raise RuntimeError("too many redirects")

def http_json(url, headers=None):
    with http_open(url, headers) as r:
        return json.loads(r.read().decode())

def download(url, dest, name, headers=None):
    if os.path.isfile(dest):
        return dest
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    h = dict(headers or {})
    if have:
        h["Range"] = "bytes=" + str(have) + "-"
    r = http_open(url, h)
    code = getattr(r, "status", 200)
    if have and code != 206:
        r.close()
        os.replace(part, part + ".stale_" + time.strftime("%H%M%S"))
        have = 0
        r = http_open(url, headers)
    total = r.headers.get("Content-Length")
    total = (int(total) + have) if total else 0
    t0, done, last = time.time(), have, 0.0
    with open(part, "ab" if have else "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 1:
                last = time.time()
    r.close()
    if total and done != total:
        raise IOError("incomplete download of " + os.path.basename(dest) + ": " + str(done) + " of " + str(total) + " bytes")
    os.replace(part, dest)
    log(name + ": downloaded " + os.path.basename(dest) + " (" + format(done / 1e6, ".1f") + " MB in " + format(time.time() - t0, ".0f") + " s)")
    return dest


def kaggle_headers():
    cfg = os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if not tok and os.path.isfile(os.path.join(cfg, "access_token")):
        tok = open(os.path.join(cfg, "access_token")).read().strip()
    if tok:
        return {"Authorization": "Bearer " + tok}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key) and os.path.isfile(os.path.join(cfg, "kaggle.json")):
        j = json.load(open(os.path.join(cfg, "kaggle.json")))
        user, key = j.get("username"), j.get("key")
    if key and key.startswith("KGAT_"):
        return {"Authorization": "Bearer " + key}
    if user and key:
        return {"Authorization": "Basic " + base64.b64encode((user + ":" + key).encode()).decode()}
    raise RuntimeError("no Kaggle credentials found (~/.kaggle/kaggle.json, ~/.kaggle/access_token or KAGGLE_* variables)")

def fetch_kaggle(slug, name):
    dest = os.path.join(DOWNLOADS, "kaggle__" + slug.replace("/", "__") + ".zip")
    return [download("https://www.kaggle.com/api/v1/datasets/download/" + slug, dest, name, kaggle_headers())]

def fetch_dataverse(doi, name):
    meta = http_json("https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=" + doi)
    files = meta["data"]["latestVersion"]["files"]
    want = ("HAM10000_images_part_1", "HAM10000_images_part_2", "HAM10000_segmentations_lesion_tschandl", "HAM10000_metadata")
    out = []
    for f in files:
        df = f["dataFile"]
        fname = df.get("originalFileName") or df["filename"]
        if not fname.startswith(want):
            continue
        url = "https://dataverse.harvard.edu/api/access/datafile/" + str(df["id"])
        if df.get("originalFileName"):
            url += "?format=original"
        out.append(download(url, os.path.join(DOWNLOADS, "dataverse__" + fname), name))
    if len(out) < 4:
        raise RuntimeError("Dataverse listing did not contain the four HAM10000 files")
    return out

def extract_all(paths, target, name):
    os.makedirs(target, exist_ok=False)
    for p in paths:
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                sub = os.path.join(target, re.sub(r"\.zip$", "", os.path.basename(p)).split("__")[-1])
                z.extractall(sub)
        else:
            os.makedirs(os.path.join(target, "files"), exist_ok=True)
            shutil.copy2(p, os.path.join(target, "files", os.path.basename(p).split("__")[-1]))
    for _ in range(3):
        inner = [os.path.join(d, f) for d, _, fs in os.walk(target) for f in fs if f.lower().endswith(".zip")]
        new = [z for z in inner if not os.path.isdir(z[:-4])]
        if not new:
            break
        for z in new:
            if zipfile.is_zipfile(z):
                with zipfile.ZipFile(z) as zz:
                    zz.extractall(z[:-4])
    log(name + ": extracted to " + target)


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def walk_files(root):
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(IMG_EXT)]

def manifest(name):
    with open(os.path.join(ROOT, "splits", name + "_partitions.csv"), newline="") as f:
        return list(csv.DictReader(f))

def verify_busi(root):
    for d, ds, _ in os.walk(root):
        sub = {x.lower(): os.path.join(d, x) for x in ds}
        if {"normal", "benign", "malignant"} <= set(sub):
            counts, masks, present = {}, {}, set()
            for c in ("normal", "benign", "malignant"):
                fs = os.listdir(sub[c])
                imgs = [f for f in fs if f.lower().endswith(".png") and "_mask" not in f]
                counts[c] = len(imgs)
                masks[c] = sum(1 for f in imgs if any(g.startswith(f[:-4] + "_mask") for g in fs))
                present |= {c + "/" + f for f in imgs}
            man = manifest("busi")
            miss = [r["file"] for r in man if r["file"] not in present]
            ok = counts == {"normal": 133, "benign": 437, "malignant": 210} and masks == counts and not miss
            return ok, {"root": d, "images": counts, "images_with_mask": masks, "expected": "133/437/210 = 780",
                        "partition_files_found": str(len(man) - len(miss)) + " of " + str(len(man)), "missing_examples": miss[:5]}
    return False, {"error": "no folder with normal/benign/malignant found"}

def verify_ham(root):
    pat = re.compile(r"(ISIC_\d{7})(_segmentation)?\.(jpg|jpeg|png)$", re.I)
    imgs, masks = set(), set()
    for f in walk_files(root):
        m = pat.search(os.path.basename(f))
        if m:
            in_mask_dir = any(("mask" in p.lower() or "segmentation" in p.lower()) for p in os.path.relpath(f, root).split(os.sep)[:-1])
            (masks if (m.group(2) or in_mask_dir) else imgs).add(m.group(1).upper())
    man = manifest("ham10000")
    ids = {r["file"][:-4].upper() for r in man}
    meta = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().startswith("ham10000_metadata")]
    ok = len(imgs) == 10015 and len(masks) == 10015 and ids <= imgs and ids <= masks
    return ok, {"images": len(imgs), "masks": len(masks), "expected": "10,015 images and 10,015 lesion masks",
                "partition_ids_with_image_and_mask": len(ids & imgs & masks), "metadata": [os.path.relpath(m, root) for m in meta]}

def verify_brisc(root):
    cls = ("glioma", "meningioma", "pituitary", "no_tumor")
    cnt, tumour_stems, mask_stems = {}, set(), set()
    for f in walk_files(root):
        parts = [p.lower() for p in f.split(os.sep)]
        if "classification_task" in parts and parts[-2] in cls and parts[-3] in ("train", "test"):
            cnt[parts[-3] + "/" + parts[-2]] = cnt.get(parts[-3] + "/" + parts[-2], 0) + 1
            if parts[-2] != "no_tumor":
                tumour_stems.add(os.path.splitext(parts[-1])[0])
        if "segmentation_task" in parts and "masks" in parts:
            mask_stems.add(os.path.splitext(parts[-1])[0])
    n_tr = sum(v for k, v in cnt.items() if k.startswith("train/"))
    n_te = sum(v for k, v in cnt.items() if k.startswith("test/"))
    no_mask = sorted(tumour_stems - mask_stems)
    ok = (n_tr, n_te) == (5000, 1000) and not no_mask
    return ok, {"train_pool": n_tr, "test": n_te, "per_class": dict(sorted(cnt.items())), "masks": len(mask_stems),
                "tumour_images_without_mask": len(no_mask), "expected": "5,000 / 1,000 images; every tumour image with a mask"}

VERIFY = {"BUSI": verify_busi, "HAM10000": verify_ham, "BRISC": verify_brisc}


def get_dataset(name):
    final = os.path.join(RAW, name)
    if os.path.isdir(final):
        ok, info = VERIFY[name](final)
        log(name + ": already present at " + final + " | verified " + str(ok))
        return ok, info
    errors = []
    for kind, ref in SOURCES[name]:
        log(name + ": trying " + kind + " " + ref)
        try:
            if kind == "kaggle":
                paths = fetch_kaggle(ref, name)
            elif kind == "kaggle+kaggle":
                paths = []
                for slug in ref.split("|"):
                    paths += fetch_kaggle(slug, name)
            else:
                paths = fetch_dataverse(ref, name)
            stage_dir = os.path.join(STAGING, name + "_" + RUN_ID + "_" + kind.replace("+", "_"))
            extract_all(paths, stage_dir, name)
            ok, info = VERIFY[name](stage_dir)
            if ok:
                os.rename(stage_dir, final)
                info = VERIFY[name](final)[1]
                log(name + ": verified and placed at " + final)
                return True, info
            errors.append(kind + ":" + ref + " -> did not verify: " + json.dumps(info)[:400])
            log(name + ": " + errors[-1] + " (kept for inspection at " + stage_dir + ")")
        except Exception as e:
            errors.append(kind + ":" + ref + " -> " + type(e).__name__ + ": " + str(e)[:300])
            log(name + ": " + errors[-1])
    return False, {"errors": errors}


def main():
    for d in (DOWNLOADS, STAGING):
        os.makedirs(d, exist_ok=True)
    done = {name: get_dataset(name)[0] for name in ("BUSI", "BRISC", "HAM10000")}
    log("finished: " + ", ".join(k + (" OK" if v else " FAILED") for k, v in done.items()))


if __name__ == "__main__":
    main()

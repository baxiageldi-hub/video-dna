#!/usr/bin/env python3
"""
VIDEO DNA — ÖLÇÜM MOTORU (Katman 1: yapay zekâsız, sayısal ölçüm)

Kullanım:
  python measure.py VIDEO_YOLU_veya_LINK [-o cikti] [--lang tr] [--whisper small]
                    [--skip speech,ocr] [--fps 10] [--ocr-fps 2]

Çıktı (cikti/<video_adı>/):
  measure.json       -> tüm sayısal metrikler (her metrik 'olcum' veya 'yaklasik' etiketli)
  contact_sheet.jpg  -> her sahneden bir kare + zaman kodu (AI yorumu için)
  transcript.txt     -> zaman kodlu transkript (AI yorumu için)
"""
import argparse, json, os, re, subprocess, sys, math, collections, shutil

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import numpy as np
import cv2

VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
TRIGGERS = ["bakın", "bak", "dikkat", "ama", "peki", "işte", "şimdi", "inanılmaz", "sır", "kimse",
            "asla", "hemen", "neden", "nasıl", "gerçek", "aslında", "sonunda", "mutlaka", "hiç",
            "look", "wait", "but", "here", "secret", "never", "nobody", "actually", "watch", "why", "how"]
DIRECT = ["sen", "siz", "senin", "sizin", "sana", "size", "seni", "sizi", "you", "your"]
STOP = set("ve bir bu da de için ile çok ne o şu gibi daha mi mı mu mü ki ya en the a an and of to is in it that".split())


# ----------------------------------------------------------------- yardımcılar
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def r(x, n=3):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)
    except Exception:
        return None


def stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "ort": r(a.mean()), "medyan": r(np.median(a)), "min": r(a.min()),
            "max": r(a.max()), "p25": r(np.percentile(a, 25)), "p75": r(np.percentile(a, 75))}


def tc(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def fetch(src, workdir):
    if not re.match(r"^https?://", src):
        return src
    if not shutil.which("yt-dlp"):
        sys.exit("Link için yt-dlp gerekli: pip install yt-dlp")
    out = os.path.join(workdir, "source.%(ext)s")
    p = run(["yt-dlp", "-q", "--no-playlist", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
             "-o", out, src])
    files = [f for f in os.listdir(workdir) if f.startswith("source.")]
    if p.returncode != 0 or not files:
        sys.exit("İndirme başarısız (platform engellemiş olabilir). Videoyu indirip dosya yolu verin.\n" + p.stderr[-800:])
    return os.path.join(workdir, files[0])


# ----------------------------------------------------------------- 1. temel bilgi
def probe(path):
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    info = json.loads(p.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"]))
    if rot in (90, 270):
        w, h = h, w
    dur = float(info["format"].get("duration", v.get("duration", 0)))
    ar = w / h
    label = "9:16 dikey" if abs(ar - 9 / 16) < 0.05 else "16:9 yatay" if abs(ar - 16 / 9) < 0.05 else \
        "1:1 kare" if abs(ar - 1) < 0.05 else "4:5" if abs(ar - 0.8) < 0.05 else f"{ar:.2f}"
    return {"sure_sn": r(dur, 2), "fps": r(fps, 2), "genislik": w, "yukseklik": h, "en_boy": label,
            "ses_var": any(s["codec_type"] == "audio" for s in info["streams"])}


# ----------------------------------------------------------------- 2. tek geçişte görüntü analizi
def visual_pass(path, base, an_fps, ocr_fps, ocr_dir):
    cap = cv2.VideoCapture(path)
    fps = base["fps"] or 30
    step = max(1, int(round(fps / an_fps)))
    ocr_step = max(1, int(round(fps / ocr_fps)))
    diffs, times = [], []           # her kare için küçük gri fark (sahne içi olaylar ve geçiş tipi için)
    flow_rows = []                  # örneklenmiş optik akış
    color_px, bright, sat, contrast = [], [], [], []
    ocr_frames = []
    hists = []
    prev_tiny, prev_flow = None, None
    first_frame = last_frame = None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        tiny = cv2.cvtColor(cv2.resize(fr, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_tiny is not None:
            diffs.append(float(np.mean(np.abs(tiny - prev_tiny))))
            times.append(t)
        prev_tiny = tiny
        if first_frame is None:
            first_frame = cv2.resize(fr, (160, 160))
        last_frame = fr
        if i % step == 0:
            h0, w0 = fr.shape[:2]
            sw = 256
            sh = int(h0 * sw / w0)
            g = cv2.cvtColor(cv2.resize(fr, (sw, sh)), cv2.COLOR_BGR2GRAY)
            if prev_flow is not None:
                f = cv2.calcOpticalFlowFarneback(prev_flow, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                ys, xs = np.mgrid[0:sh, 0:sw]
                px, py = xs - sw / 2, ys - sh / 2
                dx, dy = f[..., 0], f[..., 1]
                mdx, mdy = float(dx.mean()), float(dy.mean())
                zoom = float((dx * px + dy * py).sum() / (px ** 2 + py ** 2).sum())  # ölçek değişimi / adım
                local = float(np.mean(np.hypot(dx - mdx, dy - mdy)))
                dt = step / fps
                flow_rows.append({"t": t, "pan": mdx / sw * 100 / dt, "tilt": mdy / sh * 100 / dt,
                                  "zoom": zoom * 100 / dt, "local": local / sw * 100 / dt})
            prev_flow = g
            hsv_s = cv2.cvtColor(cv2.resize(fr, (96, 96)), cv2.COLOR_BGR2HSV)
            hh = cv2.calcHist([hsv_s], [0, 1], None, [16, 16], [0, 180, 0, 256]); cv2.normalize(hh, hh)
            hists.append((t, hh))
            if i % (step * max(1, an_fps // 2)) == 0:   # ~2 fps renk örneği
                small = cv2.resize(fr, (96, int(96 * fr.shape[0] / fr.shape[1])))
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
                bright.append((t, float(hsv[..., 2].mean() / 2.55)))
                sat.append(float(hsv[..., 1].mean() / 2.55))
                contrast.append(float(lab[..., 0].std() / 2.55))
                px_ = small.reshape(-1, 3)
                color_px.append(px_[np.random.default_rng(i).choice(len(px_), min(400, len(px_)), replace=False)])
        if ocr_dir and i % ocr_step == 0:
            fn = os.path.join(ocr_dir, f"{i:06d}.jpg")
            cv2.imwrite(fn, fr)
            ocr_frames.append((t, fn))
        i += 1
    cap.release()
    return dict(diffs=np.array(diffs), times=np.array(times), flow=flow_rows, color_px=color_px,
                bright=bright, sat=sat, contrast=contrast, ocr_frames=ocr_frames, hists=hists, an_fps=an_fps,
                first=first_frame, last=last_frame)


# ----------------------------------------------------------------- 3. sahneler
def scenes_and_events(path, base, vp):
    try:
        from scenedetect import detect, AdaptiveDetector
        sl = detect(path, AdaptiveDetector())
        bounds = [(s.get_seconds(), e.get_seconds()) for s, e in sl]
    except Exception as e:
        print("  ! PySceneDetect hatası:", e)
        bounds = []
    dur = base["sure_sn"]
    if not bounds:
        bounds = [(0.0, dur)]
    cuts = [b[0] for b in bounds[1:]]
    # yumuşak geçiş (fade/dissolve) tespiti: önce ve sonra stabil, arada renk dağılımı tamamen değişen bölge
    soft = []
    H_ = vp["hists"]; k = max(1, int(vp["an_fps"]))
    corr = lambda a, b: cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
    cand = []
    for i in range(k, len(H_) - 2 * k):
        ti = H_[i][0]
        if any(abs(c - (ti + 0.5)) < 1.0 for c in cuts):
            continue
        if corr(H_[i][1], H_[i + k][1]) < 0.4 and corr(H_[i - k][1], H_[i][1]) > 0.85 \
                and corr(H_[i + k][1], H_[i + 2 * k][1]) > 0.85:
            cand.append((corr(H_[i][1], H_[i + k][1]), ti + 0.5))
    for c_, tm in sorted(cand):
        if all(abs(tm - x) > 1.0 for x in soft):
            soft.append(tm)
    if soft:
        allc = sorted(cuts + soft)
        edges = [0.0] + allc + [dur]
        bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        cuts = allc
    lens = [e - s for s, e in bounds]
    d, t = vp["diffs"], vp["times"]
    fps = base["fps"]
    # geçiş tipi: kesme anındaki fark tek karede mi yoğunlaşmış?
    types = []
    for c in cuts:
        if any(abs(c - x) < 1e-6 for x in soft):
            types.append("yumusak_animasyonlu"); continue
        idx = int(np.searchsorted(t, c))
        w = d[max(0, idx - 6): idx + 6]
        share = float(w.max() / (w.sum() + 1e-6)) if w.size else 1
        types.append("sert" if share > 0.45 else "yumusak_animasyonlu")
    # sahne içi görsel olaylar: yerel medyanın çok üstündeki fark tepeleri (kesme olmayanlar)
    events = []
    if d.size > 10:
        k = int(fps)  # 1 sn pencere
        med = np.array([np.median(d[max(0, j - k): j + k]) for j in range(len(d))])
        mad = np.median(np.abs(d - np.median(d))) + 1e-6
        thr = med + 6 * mad
        last = -1
        for j in range(len(d)):
            if d[j] > thr[j] and d[j] > 3 and (t[j] - last) > 0.3:
                if all(abs(t[j] - c) > 0.25 for c in cuts):
                    events.append(float(t[j]))
                    last = t[j]
    n_soft = types.count("yumusak_animasyonlu")
    return {
        "sahne_sayisi": len(bounds),
        "sahne_sureleri": stats(lens),
        "dakikada_kesme": r(len(cuts) / (dur / 60)) if dur else None,
        "sert_kesme_orani": r((len(cuts) - n_soft) / len(cuts)) if cuts else None,
        "yumusak_gecis_orani": r(n_soft / len(cuts)) if cuts else None,
        "sahne_ici_gorsel_olay_sayisi": len(events),
        "toplam_gorsel_olay_dakikada": r((len(cuts) + len(events)) / (dur / 60)) if dur else None,
        "liste": [{"bas": r(s, 2), "bit": r(e, 2), "sure": r(e - s, 2),
                   "gecis": types[i - 1] if i > 0 else "baslangic"} for i, (s, e) in enumerate(bounds)],
        "_cuts": cuts, "_events": events, "_bounds": bounds,
    }


# ----------------------------------------------------------------- 4. hareket
def motion(vp, dur):
    rows = vp["flow"]
    if not rows:
        return {}
    arr = {k: np.array([x[k] for x in rows]) for k in ("t", "pan", "tilt", "zoom", "local")}
    labels = []
    for p, tl, z, lo in zip(arr["pan"], arr["tilt"], arr["zoom"], arr["local"]):
        if abs(z) > 3:
            labels.append("zoom_in" if z > 0 else "zoom_out")
        elif abs(p) > 4:
            labels.append("pan")
        elif abs(tl) > 4:
            labels.append("tilt")
        elif lo > 4:
            labels.append("nesne_hareketi")
        else:
            labels.append("sabit")
    cnt = collections.Counter(labels)
    n = len(labels)
    # hareket segmentleri ve ease analizi
    segs, cur = [], None
    for i, lab in enumerate(labels):
        if lab in ("zoom_in", "zoom_out", "pan", "tilt"):
            if cur and cur["tip"] == lab and i <= cur["son"] + 3:
                cur["son"] = i
            else:
                cur = {"tip": lab, "ilk": i, "son": i}
                segs.append(cur)
        # farklı etiket gelirse segment hemen kapanmaz; 3 örneğe kadar boşluk tolere edilir
    segs = [x for x in segs if x["son"] - x["ilk"] >= 2]   # en az ~0.3 sn
    seg_out, ease_in, ease_out, considered = [], 0, 0, 0
    for s in segs:
        key = {"zoom_in": "zoom", "zoom_out": "zoom", "pan": "pan", "tilt": "tilt"}[s["tip"]]
        lo_, hi_ = max(0, s["ilk"] - 2), min(n, s["son"] + 3)
        sp = np.abs(arr[key][lo_:hi_])
        if len(sp) >= 5:
            considered += 1
            th = max(1, len(sp) // 3)
            mid = sp[th:-th].mean() if len(sp) > 2 * th else sp.mean()
            ease_in += sp[:th].mean() < 0.6 * mid
            ease_out += sp[-th:].mean() < 0.6 * mid
        seg_out.append({"tip": s["tip"], "bas": r(arr["t"][s["ilk"]], 2), "bit": r(arr["t"][s["son"]], 2),
                        "ort_hiz_yuzde_sn": r(np.abs(arr[key][s["ilk"]:s["son"] + 1]).mean(), 2)})
    return {
        "_not": "pan/tilt: kare genişliğinin/yüksekliğinin %'si/sn; zoom: ölçek değişimi %/sn (yaklasik, optik akış)",
        "zaman_dagilimi_yuzde": {k: r(v / n * 100, 1) for k, v in cnt.items()},
        "kamera_hareketsiz_orani": r(cnt.get("sabit", 0) / n),
        "zoom_in_hizi": stats(arr["zoom"][arr["zoom"] > 3]),
        "zoom_out_hizi": stats(-arr["zoom"][arr["zoom"] < -3]),
        "pan_hizi": stats(np.abs(arr["pan"][np.abs(arr["pan"]) > 4])),
        "hareket_segment_sayisi": len(seg_out),
        "dakikada_kamera_hareketi": r(len(seg_out) / (dur / 60)) if dur else None,
        "ease_in_orani": r(ease_in / considered) if considered else None,
        "ease_out_orani": r(ease_out / considered) if considered else None,
        "segmentler": seg_out[:200],
        "_labels": list(zip(arr["t"].tolist(), labels)),
        "_local": arr["local"].tolist(), "_t": arr["t"].tolist(),
    }


# ----------------------------------------------------------------- 5. renk
def color(vp):
    if not vp["color_px"]:
        return {}
    px = np.concatenate(vp["color_px"]).astype(np.float32)
    k = 6
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, lab, cen = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=k) / len(lab)
    order = np.argsort(-share)
    pal = [{"hex": "#%02X%02X%02X" % (int(cen[i][2]), int(cen[i][1]), int(cen[i][0])), "pay": r(share[i])} for i in order]
    b = np.array([x[1] for x in vp["bright"]])
    return {"palet": pal, "parlaklik_ort": r(b.mean(), 1), "parlaklik_std": r(b.std(), 1),
            "doygunluk_ort": r(np.mean(vp["sat"]), 1), "kontrast_ort": r(np.mean(vp["contrast"]), 1),
            "_not": "0-100 ölçeği; palet k-means (olcum)"}


# ----------------------------------------------------------------- 6. ses
def audio(path, workdir, base):
    if not base["ses_var"]:
        return {}, None, None
    wav = os.path.join(workdir, "audio.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", "22050", wav])
    out = {}
    p = run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"])
    tail = p.stderr[p.stderr.rfind("Summary:"):]
    for key, pat in [("lufs_entegre", r"I:\s+(-?[\d.]+) LUFS"), ("lra", r"LRA:\s+(-?[\d.]+) LU"),
                     ("true_peak_dbfs", r"Peak:\s+(-?[\d.]+) dBFS")]:
        m = re.search(pat, tail)
        out[key] = float(m.group(1)) if m else None
    try:
        import librosa
    except ImportError:
        out["_hata"] = "librosa yok"
        return out, None, None
    y, sr = librosa.load(wav, sr=22050, mono=True)
    hop = 2205  # 0.1 sn
    rms = librosa.feature.rms(y=y, frame_length=4410, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
    tt = np.arange(len(db)) * hop / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_t = librosa.frames_to_time(beats, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    on = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    on_fr = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    envn = onset_env / (onset_env.max() + 1e-9)
    if len(on_fr):
        thr_ = max(0.35, float(np.percentile(envn[on_fr], 80)))
        strong = librosa.frames_to_time(on_fr[envn[on_fr] >= thr_], sr=sr)
    else:
        strong = np.array([])
    # sessizlikler
    sil, start = [], None
    for ti, v in zip(tt, db):
        if v < -40 and start is None:
            start = ti
        elif v >= -40 and start is not None:
            if ti - start >= 0.3:
                sil.append((r(start, 2), r(ti - start, 2)))
            start = None
    drops = [r(tt[j], 2) for j in range(1, len(db)) if db[j - 1] - db[j] > 20]
    # perde (konuşma ifadesi için yaklaşık)
    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, hop_length=512)
        f0 = f0[(f0 > 75) & (f0 < 390)]
        semis = 12 * np.log2(f0 / np.median(f0)) if f0.size else np.array([])
        pitch_std = r(semis.std(), 2) if semis.size else None
    except Exception:
        pitch_std = None
    out.update({
        "bpm": r(np.atleast_1d(tempo)[0], 1), "vurus_sayisi": int(len(beat_t)),
        "onset_dakikada": r(len(on) / (base["sure_sn"] / 60)),
        "guclu_ses_olayi_sayisi": int(len(strong)),
        "rms_db_std": r(db.std(), 2),
        "sessizlikler": sil[:100], "sessizlik_sayisi": len(sil),
        "ani_ses_kesilmeleri": drops[:50],
        "perde_degiskenligi_yarimton": pitch_std,
        "_not": "LUFS/peak: ffmpeg ebur128 (olcum). BPM: librosa (olcum, konuşmalı videolarda yanılabilir). perde: yaklasik",
    })
    return out, {"beats": beat_t.tolist(), "onsets": on.tolist(), "strong": strong.tolist(),
                 "db": db.tolist(), "tt": tt.tolist()}, wav


# ----------------------------------------------------------------- 7. konuşma
def speech(wav, lang, model_name, dur):
    if not wav:
        return {}, []
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"_hata": "faster-whisper yok (pip install faster-whisper)"}, []
    m = WhisperModel(model_name, device="cpu", compute_type="int8")
    import librosa
    y16, _ = librosa.load(wav, sr=16000, mono=True)  # PyAV sürüm sorunlarından kaçınmak için dizi ver
    segs, info = m.transcribe(y16.astype(np.float32), language=lang, word_timestamps=True, vad_filter=True)
    words, sents = [], []
    for s in segs:
        sents.append({"bas": r(s.start, 2), "bit": r(s.end, 2), "metin": s.text.strip()})
        for w in s.words or []:
            words.append({"w": w.word.strip(), "s": float(w.start), "e": float(w.end)})
    if not words:
        return {"kelime_sayisi": 0, "dil": info.language}, []
    norm = lambda x: re.sub(r"[^\wçğıöşüâîû]", "", x.lower())
    syl = sum(sum(ch in VOWELS for ch in w["w"]) for w in words)
    speak_time = sum(w["e"] - w["s"] for w in words)
    gaps = [(words[i]["s"], words[i]["s"] - words[i - 1]["e"]) for i in range(1, len(words))]
    pauses = [(r(t, 2), r(g, 2)) for t, g in gaps if g >= 0.25]
    span = words[-1]["e"] - words[0]["s"]
    toks = [norm(w["w"]) for w in words]
    freq = collections.Counter(t for t in toks if t and t not in STOP and len(t) > 2)
    trig = {}
    for i, tk in enumerate(toks):
        if tk in TRIGGERS:
            trig.setdefault(tk, []).append(r(words[i]["s"], 2))
    # 5 sn pencerede hız
    win = []
    for a in np.arange(0, dur, 5):
        n = sum(1 for w in words if a <= w["s"] < a + 5)
        win.append(n * 12)
    return {
        "dil": info.language,
        "kelime_sayisi": len(words),
        "wpm_tum_video": r(len(words) / (dur / 60)),
        "wpm_konusma_suresi": r(len(words) / (span / 60)) if span > 0 else None,
        "hece_saniye": r(syl / span) if span > 0 else None,
        "konusma_kaplama_orani": r(span / dur),
        "ilk_kelime_sn": r(words[0]["s"], 2),
        "ilk_3sn_kelime": " ".join(w["w"] for w in words if w["s"] < 3),
        "duraksamalar": {"sayi": len(pauses), "sure": stats([p[1] for p in pauses]), "liste": pauses[:80]},
        "hiz_egrisi_wpm_5sn": win,
        "hiz_degiskenligi": r(np.std(win) / (np.mean(win) + 1e-6)),
        "cumle_uzunlugu_kelime": stats([len(s["metin"].split()) for s in sents]),
        "soru_sayisi": sum(s["metin"].endswith("?") for s in sents),
        "soru_dakikada": r(sum(s["metin"].endswith("?") for s in sents) / (dur / 60)),
        "soru_zamanlari": [s["bas"] for s in sents if s["metin"].endswith("?")],
        "dogrudan_hitap_sayisi": sum(t in DIRECT for t in toks),
        "dogrudan_hitap_100_kelimede": r(sum(t in DIRECT for t in toks) / len(toks) * 100),
        "tetikleyici_100_kelimede": r(sum(t in TRIGGERS for t in toks) / len(toks) * 100),
        "sayi_kullanimi": sum(bool(re.search(r"\d", w["w"])) for w in words),
        "tetikleyici_kelimeler": trig,
        "en_sik_kelimeler": freq.most_common(15),
        "ilk_cumle": sents[0]["metin"] if sents else "",
        "son_cumle": sents[-1]["metin"] if sents else "",
        "_cumleler": sents,
    }, words


# ----------------------------------------------------------------- 8. ekran yazısı (OCR)
def ocr(frames, base, words):
    if not frames:
        return {}
    try:
        import pytesseract
    except ImportError:
        return {"_hata": "pytesseract yok"}
    langs = "tur+eng"
    W, H = base["genislik"], base["yukseklik"]
    per = []
    for t, fn in frames:
        img = cv2.imread(fn)
        sc_ = 720 / max(img.shape[:2]) if max(img.shape[:2]) > 720 else 1
        img = cv2.resize(img, None, fx=sc_, fy=sc_) if sc_ != 1 else img
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # video yazıları genelde yüksek kontrastlı: açık (beyaz/sarı) veya koyu yazı maskeleri
        light = ((g > 200) | ((hsv[..., 1] > 120) & (hsv[..., 2] > 220))).astype(np.uint8) * 255
        dark = (g < 45).astype(np.uint8) * 255
        found = []
        for mask in (light, dark):
            fill = mask.mean() / 255
            if fill < 0.002 or fill > 0.6:      # neredeyse boş ya da tamamen dolu maske: yazı olası değil
                continue
            variant = 255 - mask                 # beyaz zemin üstünde siyah yazı
            try:
                d = pytesseract.image_to_data(variant, lang=langs, output_type=pytesseract.Output.DICT,
                                              config="--psm 11", timeout=8)
            except RuntimeError:
                continue
            except Exception:
                continue
            for k_ in ("left", "top", "width", "height"):
                d[k_] = [int(v / sc_) for v in d[k_]]
            for j, txt in enumerate(d["text"]):
                txt = txt.strip()
                if len(txt) >= 2 and re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2}", txt) and float(d["conf"][j]) > 70:
                    found.append((txt, d["left"][j], d["top"][j], d["width"][j], d["height"][j]))
        # tekrarları at
        uniq = {}
        for f in found:
            uniq.setdefault(f[0].lower(), f)
        per.append((t, list(uniq.values())))
        os.remove(fn)
    frames_with = [p for p in per if p[1]]
    # metin olayları (aynı metin ardışık karelerde -> tek olay)
    events, cur = [], None
    for t, items in per:
        ws = set(x[0].lower() for x in items)
        if cur and ws and len(ws & cur["ws"]) / max(1, len(ws | cur["ws"])) > 0.5:
            cur["bit"] = t
            cur["ws"] |= ws
        else:
            if ws:
                cur = {"bas": t, "bit": t, "ws": ws, "items": items}
                events.append(cur)
            else:
                cur = None
    step = (frames[1][0] - frames[0][0]) if len(frames) > 1 else 0.5
    heights = [x[4] / H * 100 for _, it in per for x in it]
    ys = [(x[2] + x[4] / 2) / H for _, it in per for x in it]
    area = []
    safe_in = safe_tot = 0
    for _, it in per:
        if it:
            x0 = min(x[1] for x in it); y0 = min(x[2] for x in it)
            x1 = max(x[1] + x[3] for x in it); y1 = max(x[2] + x[4] for x in it)
            area.append((x1 - x0) * (y1 - y0) / (W * H) * 100)
        for x in it:
            safe_tot += 1
            cy, cx = (x[2] + x[4] / 2) / H, (x[1] + x[3] / 2) / W
            safe_in += (0.10 < cy < 0.80) and (cx < 0.85)
    # altyazı-konuşma gecikmesi
    lags = []
    if words:
        norm = lambda s: re.sub(r"[^\wçğıöşü]", "", s.lower())
        for ev in events:
            for wtxt in ev["ws"]:
                cands = [w for w in words if norm(w["w"]) == norm(wtxt) and abs(w["s"] - ev["bas"]) < 2]
                if cands:
                    c = min(cands, key=lambda w: abs(w["s"] - ev["bas"]))
                    lags.append(ev["bas"] - c["s"])
    pos = collections.Counter("ust" if y < 0.33 else "orta" if y < 0.66 else "alt" for y in ys)
    return {
        "_not": "Tesseract OCR (yaklasik): stilize/hareketli yazılarda kaçırma olabilir",
        "yazili_kare_orani": r(len(frames_with) / len(per)),
        "metin_olayi_sayisi": len(events),
        "metin_olayi_dakikada": r(len(events) / (base["sure_sn"] / 60)),
        "metin_ekranda_kalma_sn": stats([e["bit"] - e["bas"] + step for e in events]),
        "olay_basina_kelime": stats([len(e["ws"]) for e in events]),
        "font_yuksekligi_ekran_yuzde": stats(heights),
        "metin_alani_ekran_yuzde": stats(area),
        "konum_dagilimi": {k: r(v / max(1, len(ys))) for k, v in pos.items()},
        "guvenli_alan_orani": r(safe_in / safe_tot) if safe_tot else None,
        "altyazi_konusma_gecikmesi_sn": stats(lags),
        "ornekler": [{"bas": r(e["bas"], 2), "metin": " ".join(sorted(e["ws"]))[:80]} for e in events[:40]],
        "_event_times": [e["bas"] for e in events],
    }


# ----------------------------------------------------------------- 9. türetilmiş: senkron, dikkat, hook, loop
def near_frac(a, b, tol):
    if not a or not b:
        return None
    b = np.array(b)
    return r(np.mean([np.min(np.abs(b - x)) <= tol for x in a]))


def derived(base, sc, mo, au_raw, sp, words, oc, vp):
    dur = base["sure_sn"]
    cuts, ev = sc["_cuts"], sc["_events"]
    out = {}
    # senkron
    if au_raw:
        beats = au_raw["beats"]
        bps = len(beats) / dur if dur else 0
        out["senkron"] = {
            "kesme_vurus_eslesme": near_frac(cuts, beats, 0.1),
            "rastlanti_beklentisi": r(min(1, 2 * 0.1 * bps)),
            "kesme_ses_olayi_eslesme": near_frac(cuts, au_raw["strong"], 0.1),
            "gorsel_olay_ses_olayi_eslesme": near_frac(cuts + ev, au_raw["strong"], 0.15),
            "_not": "eşleşme oranı rastlantı beklentisinden belirgin yüksekse bilinçli senkron vardır",
        }
    if words:
        out.setdefault("senkron", {})["kesme_kelime_basi_eslesme"] = near_frac(cuts, [w["s"] for w in words], 0.15)
    # uyaran zaman çizelgesi
    stim = sorted(set([round(x, 2) for x in cuts + ev + (oc.get("_event_times", []) if oc else [])]
                      + [s["bas"] for s in mo.get("segmentler", [])]))
    gaps = np.diff([0] + stim + [dur]) if dur else []
    mono = []
    pts = [0] + stim + [dur]
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a > 3:
            mono.append({"bas": r(a, 2), "bit": r(b, 2), "sure": r(b - a, 2)})
    per_sec = [0] * (int(math.ceil(dur)) + 1)
    for s in stim:
        per_sec[int(s)] += 1
    pi = []
    if au_raw:
        st = np.array(au_raw["strong"]) if au_raw["strong"] else np.array([])
        for x in cuts + ev:
            if st.size and np.min(np.abs(st - x)) < 0.2:
                pi.append(r(x, 2))
    out["dikkat"] = {
        "_not": "retention TAHMİNİ değildir; görsel/işitsel uyaran yoğunluğundan türetilen risk göstergesidir (yaklasik)",
        "uyaran_sayisi": len(stim),
        "uyaran_dakikada": r(len(stim) / (dur / 60)) if dur else None,
        "uyaran_araligi_sn": stats(gaps),
        "monoton_bolumler_3sn_ustu": mono,
        "monoton_sure_orani": r(sum(m["sure"] for m in mono) / dur) if dur else None,
        "pattern_interrupt_sayisi": len(pi),
        "pattern_interrupt_dakikada": r(len(pi) / (dur / 60)) if dur else None,
        "pattern_interrupt_zamanlari": pi,
        "saniye_basina_uyaran": per_sec,
    }
    # hook
    lab = mo.get("_labels", [])
    loc = np.array(mo.get("_local", [])); lt = np.array(mo.get("_t", []))
    h = {}
    for w_ in (1, 3, 5):
        h[f"ilk_{w_}sn_gorsel_olay"] = sum(1 for x in cuts + ev if x < w_)
    allv = sorted(cuts + ev)
    h["ilk_gorsel_olay_sn"] = r(allv[0], 2) if allv else None
    h["ilk_kesme_sn"] = r(cuts[0], 2) if cuts else None
    h["ilk_1sn_hareket_orani"] = r(sum(1 for t, l in lab if t < 1 and l != "sabit") / max(1, sum(1 for t, l in lab if t < 1)))
    if loc.size:
        h["ilk_3sn_hareket_yogunlugu_oran"] = r(loc[lt < 3].mean() / (loc.mean() + 1e-6)) if (lt < 3).any() else None
    if au_raw:
        db = np.array(au_raw["db"]); tt = np.array(au_raw["tt"])
        h["ilk_3sn_ses_seviyesi_fark_db"] = r(db[tt < 3].mean() - db.mean(), 2) if (tt < 3).any() else None
    if sp:
        h["ilk_kelime_sn"] = sp.get("ilk_kelime_sn")
        h["ilk_3sn_kelime"] = sp.get("ilk_3sn_kelime")
        h["ilk_3sn_soru_var"] = any(t < 3 for t in sp.get("soru_zamanlari", []))
    if oc and oc.get("_event_times"):
        h["ilk_ekran_yazisi_sn"] = r(min(oc["_event_times"]), 2)
    out["hook"] = h
    # loop
    lp = {}
    if vp["first"] is not None and vp["last"] is not None:
        a = cv2.cvtColor(vp["first"], cv2.COLOR_BGR2HSV)
        b = cv2.cvtColor(cv2.resize(vp["last"], (160, 160)), cv2.COLOR_BGR2HSV)
        ha = cv2.calcHist([a], [0, 1], None, [30, 32], [0, 180, 0, 256]); cv2.normalize(ha, ha)
        hb = cv2.calcHist([b], [0, 1], None, [30, 32], [0, 180, 0, 256]); cv2.normalize(hb, hb)
        lp["ilk_son_kare_renk_benzerligi"] = r(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
        ga = cv2.cvtColor(vp["first"], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(cv2.resize(vp["last"], (160, 160)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lp["ilk_son_kare_yapi_benzerligi"] = r(cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)[0][0])
    if sp and sp.get("son_cumle"):
        sc_ = sp["son_cumle"].strip()
        lp["son_cumle_noktasiz_bitiyor"] = not sc_.endswith((".", "!", "?"))
        fa = set(re.findall(r"\w+", sp.get("ilk_cumle", "").lower()))
        fb = set(re.findall(r"\w+", sc_.lower()))
        lp["ilk_son_cumle_kelime_ortakligi"] = r(len(fa & fb) / max(1, len(fa | fb)))
    if au_raw:
        db = np.array(au_raw["db"]); tt = np.array(au_raw["tt"])
        lp["son_0_5sn_ses_db"] = r(db[tt > dur - 0.5].mean(), 1) if (tt > dur - 0.5).any() else None
    out["loop"] = lp
    return out


# ----------------------------------------------------------------- 10. AI için malzeme
def contact_sheet(path, bounds, outfn, base, maxn=40):
    cap = cv2.VideoCapture(path)
    pick = bounds if len(bounds) <= maxn else [bounds[int(i)] for i in np.linspace(0, len(bounds) - 1, maxn)]
    tiles = []
    tw = 240
    for s, e in pick:
        t = (s + e) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        th = int(fr.shape[0] * tw / fr.shape[1])
        fr = cv2.resize(fr, (tw, th))
        cv2.rectangle(fr, (0, 0), (tw, 22), (0, 0, 0), -1)
        cv2.putText(fr, f"{tc(s)}-{tc(e)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(fr)
    cap.release()
    if not tiles:
        return
    th = tiles[0].shape[0]
    tiles = [cv2.resize(x, (tw, th)) for x in tiles]
    cols = 6 if len(tiles) > 12 else 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(outfn, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not k.startswith("_") or k == "_not"}
    if isinstance(o, list):
        return [clean(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="video dosyası, link veya her satırında bir link/dosya olan .txt")
    ap.add_argument("-o", "--out", default="cikti")
    ap.add_argument("--name")
    ap.add_argument("--lang", default="tr", help="konuşma dili (auto için 'auto')")
    ap.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--fps", type=int, default=10, help="optik akış örnekleme fps")
    ap.add_argument("--ocr-fps", type=float, default=2)
    ap.add_argument("--skip", default="", help="virgülle: speech,ocr,audio,motion")
    a = ap.parse_args()
    srcs = []
    for s_ in a.src:
        if s_.endswith(".txt") and os.path.exists(s_):
            srcs += [l.strip() for l in open(s_, encoding="utf-8") if l.strip() and not l.startswith("#")]
        else:
            srcs.append(s_)
    if len(srcs) > 1:
        a.name = None
    for i, s_ in enumerate(srcs):
        print(f"\n=== [{i + 1}/{len(srcs)}] {s_}")
        try:
            analyze(s_, a)
        except SystemExit as e:
            print("  ! atlandı:", e)
        except Exception as e:
            print("  ! hata:", repr(e))


def analyze(src, a):
    a = argparse.Namespace(**vars(a)); a.src = src
    skip = set(x.strip() for x in a.skip.split(",") if x.strip())
    name = a.name or re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(a.src.rstrip("/")))[0])[:60]
    od = os.path.join(a.out, name)
    os.makedirs(od, exist_ok=True)
    path = fetch(a.src, od)
    print(f"[1/7] Temel bilgi: {path}")
    base = probe(path)
    if base["sure_sn"] and base["sure_sn"] > 180 and a.ocr_fps > 1:
        a.ocr_fps = 1
    ocr_dir = None if "ocr" in skip else os.path.join(od, "_ocr")
    if ocr_dir:
        os.makedirs(ocr_dir, exist_ok=True)
    print("[2/7] Görüntü geçişi (fark, optik akış, renk)")
    vp = visual_pass(path, base, a.fps, a.ocr_fps, ocr_dir)
    print("[3/7] Sahne tespiti")
    sc = scenes_and_events(path, base, vp)
    mo = {} if "motion" in skip else motion(vp, base["sure_sn"])
    co = color(vp)
    print("[4/7] Ses")
    au, au_raw, wav = ({}, None, None) if "audio" in skip else audio(path, od, base)
    print("[5/7] Konuşma (Whisper)")
    sp, words = ({}, []) if "speech" in skip else speech(wav, None if a.lang == "auto" else a.lang, a.whisper, base["sure_sn"])
    print("[6/7] Ekran yazısı (OCR)")
    oc = {} if "ocr" in skip else ocr(vp["ocr_frames"], base, words)
    if ocr_dir and os.path.isdir(ocr_dir):
        shutil.rmtree(ocr_dir, ignore_errors=True)
    print("[7/7] Senkron, dikkat, hook, loop")
    de = derived(base, sc, mo, au_raw, sp, words, oc, vp)
    contact_sheet(path, sc["_bounds"], os.path.join(od, "contact_sheet.jpg"), base)
    if sp.get("_cumleler"):
        with open(os.path.join(od, "transcript.txt"), "w", encoding="utf-8") as f:
            for s in sp["_cumleler"]:
                f.write(f"[{tc(s['bas'])} - {tc(s['bit'])}] {s['metin']}\n")
    result = {"kaynak": a.src, "video": name, "temel": base, "sahne": sc, "hareket": mo, "renk": co,
              "ses": au, "konusma": sp, "ekran_yazisi": oc, **de,
              "olcum_turleri": {"olcum": "temel, sahne, renk, ses(LUFS/peak/BPM), konusma(WPM/hece/duraksama)",
                                "yaklasik": "hareket(optik akış), ekran_yazisi(OCR), dikkat, hook, loop, senkron"}}
    if wav and os.path.exists(wav):
        os.remove(wav)
    with open(os.path.join(od, "measure.json"), "w", encoding="utf-8") as f:
        json.dump(clean(result), f, ensure_ascii=False, indent=1)
    print("Bitti ->", od)


if __name__ == "__main__":
    main()

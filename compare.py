#!/usr/bin/env python3
"""
VIDEO DNA — KATMAN 3: KARŞILAŞTIRMA VE KURAL ÇIKARIMI

Başarılı videoları kontrol grubuyla karşılaştırır; sadece gerçekten ayırt edici olan metrikleri "kural" yapar.

Kullanım:
  python compare.py cikti/ --meta meta.csv

meta.csv sütunları (video = cikti/ altındaki klasör adı):
  video, grup, izlenme, kanal_ort_izlenme, takipci, ort_izlenme_yuzdesi, kaydetme, paylasim
  - 'grup' = basarili / kontrol  (boşsa performansa göre otomatik ayrılır)
  - performans = izlenme / kanal_ort_izlenme  (yoksa izlenme / takipci, o da yoksa izlenme)
  - ort_izlenme_yuzdesi varsa (gerçek retention) ayrıca korelasyon hesaplanır

Çıktılar (cikti/ içine):
  stil_profili.json   -> makinece okunur kural seti
  rapor.md            -> insan için karşılaştırma raporu
  uretim_promptu.md   -> konu bağımsız üretim promptu (kurallardan otomatik)
"""
import argparse, csv, glob, json, os, collections
import numpy as np

SKIP_KEYS = {"liste", "segmentler", "ornekler", "sessizlikler", "ani_ses_kesilmeleri", "saniye_basina_uyaran",
             "hiz_egrisi_wpm_5sn", "soru_zamanlari", "pattern_interrupt_zamanlari", "monoton_bolumler_3sn_ustu",
             "en_sik_kelimeler", "palet", "tetikleyici_kelimeler", "kaynak", "video", "olcum_turleri",
             "genislik", "yukseklik", "fps",
             # video süresine bağlı ham sayılar (dakika başı karşılıkları kullanılıyor)
             "sahne_sayisi", "hareket_segment_sayisi", "metin_olayi_sayisi", "pattern_interrupt_sayisi",
             "uyaran_sayisi", "sessizlik_sayisi", "vurus_sayisi", "guclu_ses_olayi_sayisi", "kelime_sayisi",
             "soru_sayisi", "dogrudan_hitap_sayisi", "sayi_kullanimi", "sahne_ici_gorsel_olay_sayisi"}


def flat_measure(o, p=""):
    out = {}
    if isinstance(o, dict):
        if "n" in o and set(o) <= {"n", "ort", "medyan", "min", "max", "p25", "p75"}:
            if o.get("n") and "medyan" in o:
                out[p + ".medyan"] = o["medyan"]
            return out
        for k, v in o.items():
            if k in SKIP_KEYS or k.startswith("_"):
                continue
            out.update(flat_measure(v, f"{p}.{k}" if p else k))
    elif isinstance(o, bool):
        out[p] = float(o)
    elif isinstance(o, (int, float)):
        out[p] = float(o)
    elif isinstance(o, str) and p.endswith("en_boy"):
        out[p] = o
    return out


def flat_ai(o, p="ai"):
    out = {}
    if isinstance(o, dict):
        if "deger" in o and "durum" in o:
            if o["durum"] == "OK":
                out[p] = o["deger"] if not isinstance(o["deger"], bool) else float(o["deger"])
            return out
        for k, v in o.items():
            if k.startswith("_") or k in ("kanit", "degerler", "secilen_ornek", "uzunluklar", "guven"):
                continue
            out.update(flat_ai(v, f"{p}.{k}"))
    elif isinstance(o, list):
        for it in o:
            if isinstance(it, dict) and "oge" in it:
                out[f"{p}={it['oge']}"] = 1.0
    return out


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fnum(x):
    try:
        return float(str(x).replace(",", ".").replace(" ", ""))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--meta")
    a = ap.parse_args()

    rows = {}
    for f in glob.glob(os.path.join(a.dir, "*", "measure.json")):
        v = os.path.basename(os.path.dirname(f))
        feat = flat_measure(json.load(open(f, encoding="utf-8")))
        ai = os.path.join(os.path.dirname(f), "ai_konsensus.json")
        if os.path.exists(ai):
            feat.update(flat_ai(json.load(open(ai, encoding="utf-8"))))
        rows[v] = feat
    if not rows:
        raise SystemExit("measure.json bulunamadı")

    meta = {}
    if a.meta:
        for r in csv.DictReader(open(a.meta, encoding="utf-8-sig")):
            meta[r["video"].strip()] = r
    perf, retention = {}, {}
    for v in rows:
        m = meta.get(v, {})
        iz, ko, tk = fnum(m.get("izlenme")), fnum(m.get("kanal_ort_izlenme")), fnum(m.get("takipci"))
        perf[v] = iz / ko if iz and ko else iz / tk if iz and tk else iz
        if fnum(m.get("ort_izlenme_yuzdesi")) is not None:
            retention[v] = fnum(m.get("ort_izlenme_yuzdesi"))

    grp = {v: (meta.get(v, {}).get("grup") or "").strip().lower() for v in rows}
    if not any(g in ("basarili", "kontrol") for g in grp.values()):
        pv = {v: p for v, p in perf.items() if p is not None}
        if len(pv) >= 6:
            order = sorted(pv, key=pv.get)
            t = len(order) // 3
            for v in order[:t]:
                grp[v] = "kontrol"
            for v in order[-t:]:
                grp[v] = "basarili"
    B = [v for v in rows if grp[v] == "basarili"]
    K = [v for v in rows if grp[v] == "kontrol"]
    if not B:          # grup bilgisi yoksa hepsini başarılı say, yalnızca tutarlılık analizi yap
        B = list(rows)
    kontrol_var = len(K) >= 3

    keys = sorted(set().union(*[r.keys() for r in rows.values()]))
    kurallar, standartlar, elenen, kategorik = [], [], [], []
    for k in keys:
        bv = [rows[v][k] for v in B if k in rows[v]]
        kv = [rows[v][k] for v in K if k in rows[v]]
        if not bv:
            continue
        if isinstance(bv[0], str):
            cb = collections.Counter(bv)
            top, n = cb.most_common(1)[0]
            pb = n / len(bv)
            pk = (sum(1 for x in kv if x == top) / len(kv)) if kv else None
            item = {"metrik": k, "baskin_deger": top, "basarili_oran": round(pb, 2),
                    "kontrol_oran": None if pk is None else round(pk, 2), "n_basarili": len(bv), "n_kontrol": len(kv)}
            if kontrol_var and pk is not None and pb - pk >= 0.3 and pb >= 0.5:
                item["sinif"] = "AYIRT_EDICI"
            elif pb >= 0.7:
                item["sinif"] = "NIS_STANDARDI"
            else:
                item["sinif"] = "KURAL_DEGIL"
            kategorik.append(item)
            continue
        bv = np.array(bv, float); kv = np.array(kv, float)
        med = float(np.median(bv)); p25, p75 = (float(x) for x in np.percentile(bv, [25, 75]))
        item = {"metrik": k, "basarili_medyan": round(med, 3), "basarili_aralik": [round(p25, 3), round(p75, 3)],
                "n_basarili": len(bv)}
        if len(kv):
            item.update({"kontrol_medyan": round(float(np.median(kv)), 3), "n_kontrol": len(kv)})
        allv = [v for v in rows if k in rows[v] and perf.get(v) is not None]
        rho = spearman([rows[v][k] for v in allv], [perf[v] for v in allv]) if len(allv) >= 6 else None
        if rho is not None:
            item["performans_korelasyonu"] = round(rho, 2)
        rv = [v for v in rows if k in rows[v] and v in retention]
        rr = spearman([rows[v][k] for v in rv], [retention[v] for v in rv]) if len(rv) >= 6 else None
        if rr is not None:
            item["retention_korelasyonu"] = round(rr, 2)
        cv = (p75 - p25) / (abs(med) + 1e-9)
        if kontrol_var and len(bv) >= 3 and len(kv) >= 3:
            d = cliffs_delta(bv, kv)
            item["etki_buyuklugu_cliffs_delta"] = round(float(d), 2)
            item["yon"] = "basarililarda_yuksek" if d > 0 else "basarililarda_dusuk"
            if abs(d) >= 0.474:
                item["sinif"] = "GUCLU_KURAL"; kurallar.append(item); continue
            if abs(d) >= 0.33:
                item["sinif"] = "ORTA_KURAL"; kurallar.append(item); continue
        if len(bv) >= 3 and cv < 0.25:
            item["sinif"] = "NIS_STANDARDI"; standartlar.append(item)
        else:
            item["sinif"] = "KURAL_DEGIL"; elenen.append(item)

    kurallar.sort(key=lambda x: -abs(x.get("etki_buyuklugu_cliffs_delta", 0)))
    uyarilar = []
    if not kontrol_var:
        uyarilar.append("Kontrol grubu yok (<3 video). Sadece 'ortak özellikler' bulundu; bunların başarıyı açıkladığı KANITLANMADI.")
    if len(B) < 10:
        uyarilar.append(f"Başarılı grupta {len(B)} video var. 10'un altında sonuçlar keşif niteliğindedir.")
    if not retention:
        uyarilar.append("Gerçek izlenme yüzdesi (retention) verisi yok; 'dikkat' metrikleri doğrulanamadı.")

    prof = {"videolar": {"basarili": B, "kontrol": K}, "uyarilar": uyarilar, "kurallar": kurallar,
            "nis_standartlari": standartlar, "kategorik": kategorik, "elenen_sayisi": len(elenen)}
    json.dump(prof, open(os.path.join(a.dir, "stil_profili.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- rapor
    L = ["# Video DNA — Karşılaştırma Raporu", "",
         f"Başarılı: {len(B)} video | Kontrol: {len(K)} video", ""]
    for u in uyarilar:
        L.append(f"> ⚠️ {u}")
    L += ["", "## 1. Ayırt edici kurallar (başarılıları kontrolden ayıran)", "",
          "| Metrik | Başarılı medyan [aralık] | Kontrol medyan | Etki (Cliff δ) | Sınıf |", "|---|---|---|---|---|"]
    for i in kurallar:
        L.append(f"| {i['metrik']} | {i['basarili_medyan']} {i['basarili_aralik']} | {i.get('kontrol_medyan', '-')} | "
                 f"{i.get('etki_buyuklugu_cliffs_delta', '-')} | {i['sinif']} |")
    if not kurallar:
        L.append("| (yok) | | | | |")
    L += ["", "## 2. Niş standartları (başarılılarda tutarlı, ama tek başına fark yaratmıyor)", "",
          "| Metrik | Medyan | Aralık |", "|---|---|---|"]
    for i in standartlar:
        L.append(f"| {i['metrik']} | {i['basarili_medyan']} | {i['basarili_aralik']} |")
    L += ["", "## 3. Kategorik bulgular (AI konsensüsü + format)", "",
          "| Alan | Baskın değer | Başarılı oran | Kontrol oran | Sınıf |", "|---|---|---|---|---|"]
    for i in kategorik:
        L.append(f"| {i['metrik']} | {i['baskin_deger']} | {i['basarili_oran']} | {i['kontrol_oran']} | {i['sinif']} |")
    L += ["", f"Kural olamayacak kadar dağınık bulunan metrik sayısı: {len(elenen)}", ""]
    open(os.path.join(a.dir, "rapor.md"), "w", encoding="utf-8").write("\n".join(L))

    # ---- üretim promptu
    P = ["# ÜRETİM PROMPTU (Video DNA profilinden otomatik)", "",
         "Aşağıdaki kurallara uyan bir video kurgusu / animasyon kodu üret. Konu: {KONU}. Metin: {METIN}.",
         "Sayısal aralıklar referans videolardan ölçülmüştür; aralığın dışına çıkma.", "",
         "## Zorunlu kurallar (başarıyı ayırt edenler)"]
    for i in kurallar:
        P.append(f"- `{i['metrik']}`: hedef {i['basarili_aralik'][0]} – {i['basarili_aralik'][1]} "
                 f"(medyan {i['basarili_medyan']}; kontrol grubu {i.get('kontrol_medyan', '?')})")
    P.append("\n## Niş standartları (uyulması beklenen taban)")
    for i in standartlar:
        P.append(f"- `{i['metrik']}`: {i['basarili_aralik'][0]} – {i['basarili_aralik'][1]}")
    P.append("\n## Yapı ve psikoloji")
    for i in kategorik:
        if i["sinif"] != "KURAL_DEGIL":
            P.append(f"- `{i['metrik']}` = **{i['baskin_deger']}** (başarılılarda %{int(i['basarili_oran'] * 100)})")
    P += ["", "## Çıktı biçimi",
          "1. Zaman kodlu kurgu planı: zaman | görsel | hareket | metin/altyazı | ses efekti | amaç",
          "2. Her sahne için animasyon parametreleri (süre, easing, zoom %, pan %)",
          "3. Hook (ilk 3 sn) ve kapanış/loop ayrıntılı",
          "4. Hangi kurala nasıl uyulduğunu gösteren kontrol listesi"]
    if uyarilar:
        P += ["", "## Not", *[f"- {u}" for u in uyarilar]]
    open(os.path.join(a.dir, "uretim_promptu.md"), "w", encoding="utf-8").write("\n".join(P))
    print(f"Kural: {len(kurallar)} | Standart: {len(standartlar)} | Kategorik: {len(kategorik)} | Elenen: {len(elenen)}")
    print("->", os.path.join(a.dir, "rapor.md"), "| stil_profili.json | uretim_promptu.md")


if __name__ == "__main__":
    main()

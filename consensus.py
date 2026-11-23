#!/usr/bin/env python3
"""
VIDEO DNA — AI YORUMLARINI UZLAŞTIRMA
Aynı video için bağımsız üretilmiş ai_1.json, ai_2.json, ai_3.json ... dosyalarını birleştirir.
- Kategorik alanlar: çoğunluk oyu + uyum oranı (uyum < 2/3 ise 'BELIRSIZ')
- Sayısal alanlar: medyan + yayılım (max-min)
- Listeler: en az yarısında geçen öğeler
Kullanım: python consensus.py cikti/<video>   ->  ai_konsensus.json
"""
import json, sys, os, glob, collections, statistics

ESIK = 2 / 3


def merge(vals, path=""):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if all(isinstance(v, dict) for v in vals):
        keys = set().union(*[v.keys() for v in vals])
        return {k: merge([v.get(k) for v in vals], f"{path}.{k}") for k in sorted(keys)}
    if all(isinstance(v, bool) for v in vals):
        c = collections.Counter(vals)
        val, n = c.most_common(1)[0]
        return {"deger": val, "uyum": round(n / len(vals), 2), "durum": "OK" if n / len(vals) >= ESIK else "BELIRSIZ"}
    if all(isinstance(v, (int, float)) for v in vals):
        return {"deger": round(statistics.median(vals), 2), "yayilim": round(max(vals) - min(vals), 2),
                "durum": "OK" if (max(vals) - min(vals)) <= max(1.0, 0.25 * abs(statistics.median(vals))) else "BELIRSIZ"}
    if all(isinstance(v, str) for v in vals):
        if path.endswith(("kanit", "saklanan_bilgi", "vaat", "font_tahmini", "soru", "not", "formul")):
            return {"degerler": vals}
        c = collections.Counter(v.strip().lower() for v in vals)
        val, n = c.most_common(1)[0]
        return {"deger": val, "uyum": round(n / len(vals), 2), "durum": "OK" if n / len(vals) >= ESIK else "BELIRSIZ"}
    if all(isinstance(v, list) for v in vals):
        flat = [x for v in vals for x in v]
        if all(isinstance(x, str) for x in flat):
            c = collections.Counter(x.strip().lower() for v in vals for x in set(v))
            return [{"oge": k, "oran": round(n / len(vals), 2)} for k, n in c.most_common() if n / len(vals) >= 0.5]
        # nesne listeleri (hikâye blokları vb.) — en uzun ortak uzunluktaki listeyi değil, medyan uzunluktakini al
        L = sorted(vals, key=len)[len(vals) // 2]
        return {"secilen_ornek": L, "uzunluklar": [len(v) for v in vals]}
    return {"degerler": vals, "durum": "KARISIK_TIP"}


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "ai_[0-9]*.json")))
    if len(files) < 2:
        sys.exit("En az 2 (önerilen 3) bağımsız AI çıktısı gerekli: ai_1.json, ai_2.json, ai_3.json")
    runs = []
    for f in files:
        txt = open(f, encoding="utf-8").read().strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
        runs.append(json.loads(txt))
    out = merge(runs)
    flags = []

    def walk(o, p=""):
        if isinstance(o, dict):
            if o.get("durum") == "BELIRSIZ":
                flags.append(p)
            for k, v in o.items():
                walk(v, f"{p}.{k}" if p else k)

    walk(out)
    out["_ozet"] = {"calistirma_sayisi": len(runs), "belirsiz_alanlar": flags,
                    "not": "BELIRSIZ alanlar kural çıkarımında kullanılmaz (compare.py bunları atlar)"}
    json.dump(out, open(os.path.join(d, "ai_konsensus.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(runs)} çalıştırma birleştirildi. Belirsiz alan: {len(flags)} -> {d}/ai_konsensus.json")
    for f in flags:
        print("  BELIRSIZ:", f)


if __name__ == "__main__":
    main()

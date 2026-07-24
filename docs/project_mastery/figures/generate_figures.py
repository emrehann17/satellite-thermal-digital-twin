#!/usr/bin/env python3
"""
generate_figures.py — PROJECT_MASTERY_GUIDE için tüm şema ve veri-güdümlü
figürleri üretir. Salt-okunur: yalnızca outputs/ altındaki mevcut canonical
JSON'ları okur; hiçbir bilimsel çıktıyı değiştirmez.

Çalıştırma:
    python docs/project_mastery/figures/generate_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "none"
DPI = 200

# Renk paleti (tema-nötr, baskı-dostu)
C_INPUT = "#2d6a9f"
C_PROC = "#3c8d40"
C_MODEL = "#b5651d"
C_DIAG = "#7a4fa3"
C_DANGER = "#b23b3b"
C_NEUTRAL = "#4a4a4a"
C_LIGHT = "#eef2f6"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=C_NEUTRAL, tc="#111111",
        fs=9, bold=False, radius=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
                       linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal",
            zorder=3, wrap=True)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color=C_NEUTRAL, style="-|>", lw=1.4, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, zorder=1,
                        shrinkA=3, shrinkB=3)
    ax.add_patch(a)


def newfig(w=12, h=8, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if title:
        ax.text(50, 98, title, ha="center", va="top", fontsize=14, weight="bold")
    return fig, ax


def save(fig, name):
    out = HERE / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out.name)


def load(rel):
    p = ROOT / rel
    try:
        return json.load(open(p))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FIG 0: Bir sayfalık zihinsel model
# ---------------------------------------------------------------------------
def fig_mental_model():
    fig, ax = newfig(13, 8.2, "Şekil 0 — Projenin Bir Sayfalık Zihinsel Modeli")
    box(ax, 3, 84, 94, 8,
        "BİLİMSEL SORU: Termal/kuruluk feature'ları (LST anomaly, TVDI, downscaled/fused LST)\n"
        "statik baseline'a (elevation, slope, landcover, NDVI) göre yanmış-alan AYRIMINA ölçülebilir katkı sağlar mı —\n"
        "ve bu katkı BÖLGELER ARASINDA genellenir mi?",
        fc=C_LIGHT, ec=C_INPUT, bold=True, fs=9.5)
    # inputs
    box(ax, 2, 68, 20, 10, "GİRDİLER\nLandsat/MODIS LST,\nNDVI, DEM, WorldCover,\nMCD64A1 BurnDate", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 25, 68, 22, 10, "AOI = deney\n(bölge+yıl+predictor/\nlabel/baseline pencereleri\n+rol)", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 50, 68, 22, 10, "Pencereler:\npredictor (yangından ÖNCE)\nbaseline (önceki yıllar)\nlabel (yangın sonrası)", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 75, 68, 22, 10, "ANALİZ BİRİMİ\n~500 m MCD64A1 hücresi\n(30 m piksel ASLA örnek değil)", fc="#fde9d9", ec=C_MODEL, fs=8, bold=True)
    for x in (12, 36, 61, 86):
        arrow(ax, (x, 68), (x, 60))
    box(ax, 6, 50, 40, 9, "WITHIN-REGION MODEL (Step8)\nbaseline vs. +thermal, spatial-block CV\n→ thermal katkı bootstrap-destekli", fc="#dcefdd", ec=C_PROC, fs=8.5, bold=True)
    box(ax, 54, 50, 40, 9, "CROSS-REGION TRANSFER (Step9)\nkaynakta eğit → hedefte test (source-only)\n→ discrimination genellenmiyor", fc="#f3ddd9", ec=C_DANGER, fs=8.5, bold=True)
    arrow(ax, (26, 50), (26, 44)); arrow(ax, (74, 50), (74, 44))
    box(ax, 6, 34, 40, 9, "ADAPTASYON (Step10)\nregion-wise z-score / CORAL\n(hedef-etiket-KÖRÜ)\n→ kısmi, yöne-bağlı toparlanma", fc="#ece3f3", ec=C_DIAG, fs=8.5)
    box(ax, 54, 34, 40, 9, "BELİRSİZLİK\nspatial-block bootstrap CI\n(p-value DEĞİL)", fc="#ece3f3", ec=C_DIAG, fs=8.5)
    arrow(ax, (26, 34), (26, 28)); arrow(ax, (74, 34), (74, 28))
    box(ax, 6, 18, 88, 9,
        "TEŞHİS (Step9E/F/G, domain-classifier, burned-pattern): NEDEN transfer başarısız? — covariate shift (domain AUC≈1.0),\n"
        "ilişki-yönü kayması (elevation ters dönüyor), fire-footprint geometrisi farkları. Hepsi betimsel; NEDENSEL DEĞİL.",
        fc="#efe7f5", ec=C_DIAG, fs=8.5)
    arrow(ax, (50, 18), (50, 12))
    box(ax, 6, 2, 88, 8,
        "CLAIM SINIRI: within-region thermal katkı DESTEKLENİR • doğrudan cross-region discrimination transferi DESTEKLENMEZ •\n"
        "operasyonel yangın tahmini / erken uyarı / nedensellik İDDİA EDİLMEZ • sonuçlar yalnızca incelenen olaylar içindir.",
        fc="#fbe3e3", ec=C_DANGER, bold=True, fs=8.5)
    save(fig, "fig00_mental_model.png")


# ---------------------------------------------------------------------------
# FIG 1: Uçtan uca veri akışı
# ---------------------------------------------------------------------------
def fig_dataflow():
    fig, ax = newfig(12, 9, "Şekil 1 — Uçtan Uca Veri Akışı (deney-farkında yol)")
    steps = [
        ("Earth Engine\nLandsat/MODIS/NDVI/DEM/\nWorldCover/MCD64A1", C_INPUT, 88),
        ("run_predictors_only.py\ndirekt/tiled local indirme\n(Drive YOK)", C_PROC, 76),
        ("Step5 LST anomaly +\nStep5C TVDI ürünleri\n(anomaly_zscore, current_tvdi...)", C_PROC, 64),
        ("Step7A-E downscaling+fusion\n(downscaled_lst, fused_lst)\nher deney kendi modelini eğitir", C_PROC, 52),
        ("Step6/6A/6B: raw MCD64A1 BurnDate\nexport + burned-landcover GATE\n(wildfire_candidate_pass?)", C_MODEL, 40),
        ("Step8A: 30 m → ~500 m aggregation\nlabel-honest modeling dataset\n(1 satır = 1 MCD64A1 hücresi)", C_MODEL, 28),
        ("Step8B-E: baseline vs +thermal\nspatial-block CV + bootstrap +\nablation + final rapor", C_MODEL, 16),
    ]
    centers = []
    for txt, col, y in steps:
        c = box(ax, 22, y, 56, 9, txt, fc="#ffffff", ec=col, fs=8.5, bold=True)
        centers.append((c[0], y))
    for i in range(len(steps) - 1):
        arrow(ax, (50, steps[i][2]), (50, steps[i + 1][2] + 9))
    # side annotations
    box(ax, 2, 40, 18, 9, "GATE diagnostic:\ndurdurmaz, rapor eder", fc=C_LIGHT, ec=C_MODEL, fs=7.5)
    arrow(ax, (20, 44.5), (22, 44.5), color=C_MODEL)
    box(ax, 80, 28, 18, 9, "30 m piksel ASLA\nlabel örneği değil\n(pseudo-replication)", fc="#fbe3e3", ec=C_DANGER, fs=7.5)
    arrow(ax, (80, 32.5), (78, 32.5), color=C_DANGER)
    box(ax, 80, 16, 18, 9, "MCD64A1 tek target;\nFIRMS asla target değil", fc="#fbe3e3", ec=C_DANGER, fs=7.5)
    arrow(ax, (80, 20.5), (78, 20.5), color=C_DANGER)
    ax.text(50, 6, "Kaynak: core/pipeline_orchestrator.py STAGE_ORDER = gate→predictors→scene-provenance→step7→seam-audit→seam-localization→step8",
            ha="center", fontsize=7.5, style="italic", color=C_NEUTRAL)
    save(fig, "fig01_dataflow.png")


# ---------------------------------------------------------------------------
# FIG 2: AOI deney yaşam döngüsü
# ---------------------------------------------------------------------------
def fig_lifecycle():
    fig, ax = newfig(12, 8, "Şekil 2 — AOI / Deney Yaşam Döngüsü")
    nodes = [
        (8, 80, "1. Olay seçimi\n(Akdeniz yangını)"),
        (38, 80, "2. Registry kaydı\ncore/regions.py\nEXPERIMENTS"),
        (68, 80, "3. AOI önizleme\npreview_experiment_aoi"),
        (68, 62, "4. GATE (Step6B)\nwildfire_candidate_pass?"),
        (38, 62, "5. Predictors\nStep5/5C (export)"),
        (8, 62, "6. Step7 A-E\ndownscaling/fusion"),
        (8, 44, "7. Step8 A-E\nwithin-region model"),
        (38, 44, "8. Robustness\nlarge/big-block"),
        (68, 44, "9. Transfer eşleştirme\nStep9A-D"),
        (68, 26, "10. Diagnostics\nStep9E/F/G, domain,\nburned-pattern"),
        (38, 26, "11. Step10\nself-cal adaptation"),
        (8, 26, "12. Synthesis + freeze\ntransfer-synthesis"),
    ]
    cols = [C_INPUT, C_INPUT, C_INPUT, C_MODEL, C_PROC, C_PROC, C_MODEL, C_MODEL, C_DANGER, C_DIAG, C_DIAG, C_NEUTRAL]
    cs = []
    for (x, y, t), col in zip(nodes, cols):
        cs.append(box(ax, x, y, 24, 11, t, ec=col, fs=8, bold=True))
    order = list(range(len(nodes)))
    for i in range(len(order) - 1):
        arrow(ax, cs[i], cs[i + 1], color="#888")
    box(ax, 30, 8, 40, 8, "GATE cropland_dominated_control dönerse → negative control (Kozan).\ndownstream_authorized=false ise gate geçmek Step7+ çalıştırma yetkisi VERMEZ.",
        fc="#fbe3e3", ec=C_DANGER, fs=8)
    save(fig, "fig02_lifecycle.png")


# ---------------------------------------------------------------------------
# FIG 3: Step8 within-region değerlendirme
# ---------------------------------------------------------------------------
def fig_step8():
    fig, ax = newfig(12, 8, "Şekil 3 — Step8 Within-Region Değerlendirme")
    box(ax, 30, 88, 40, 8, "Step8A: ~500 m modeling dataset\n(parquet, 1 satır = 1 MCD64A1 hücresi)", fc="#fde9d9", ec=C_MODEL, bold=True, fs=8.5)
    box(ax, 4, 70, 42, 9, "Model A — BASELINE (4)\nndvi_mean, elevation_mean,\nslope_mean, landcover_dominant", fc="#dcefdd", ec=C_PROC, fs=8.5)
    box(ax, 54, 70, 42, 9, "Model B — +THERMAL (10)\n+lst_anomaly, current_lst, current_tvdi,\ntvdi_difference, downscaled_lst, fused_lst", fc="#dcefdd", ec=C_PROC, fs=8.5)
    arrow(ax, (50, 88), (25, 79)); arrow(ax, (50, 88), (75, 79))
    box(ax, 20, 54, 60, 8, "Step8B: StratifiedGroupKFold (5-fold), spatial_block_id = (row//2, col//2)\n→ OOF tahminleri, delta_AUC & delta_PR-AUC", fc="#ffffff", ec=C_PROC, bold=True, fs=8.5)
    arrow(ax, (25, 70), (40, 62)); arrow(ax, (75, 70), (60, 62))
    box(ax, 4, 38, 28, 9, "Step8C: spatial-block\nbootstrap (1000)\n→ %95 CI (p-value DEĞİL)", fc="#ece3f3", ec=C_DIAG, fs=8)
    box(ax, 36, 38, 28, 9, "Step8D: thermal\nfeature ablation\n(11 model)", fc="#ece3f3", ec=C_DIAG, fs=8)
    box(ax, 68, 38, 28, 9, "large/big-block\nrobustness\n(10/20 hücre)", fc="#ece3f3", ec=C_DIAG, fs=8)
    for x in (18, 50, 82):
        arrow(ax, (50, 54), (x, 47))
    box(ax, 20, 22, 60, 8, "Step8E: yeniden eğitim YOK — B/C/D'yi tek rapora birleştirir\n(step8e_summary.md/json)", fc="#ffffff", ec=C_MODEL, bold=True, fs=8.5)
    for x in (18, 50, 82):
        arrow(ax, (x, 38), (50, 30))
    box(ax, 10, 6, 80, 9,
        "LEAKAGE BARİYERLERİ: random split YOK • 30 m piksel örnek değil • spatial-block gruplama •\n"
        "aynı seed/CV fold'ları • MCD64A1 tek target • preprocessing tüm veriden fit (within-region meşru)",
        fc="#fbe3e3", ec=C_DANGER, fs=8, bold=True)
    save(fig, "fig03_step8.png")


# ---------------------------------------------------------------------------
# FIG 4: Step9 source-only transfer
# ---------------------------------------------------------------------------
def fig_step9():
    fig, ax = newfig(12, 7.6, "Şekil 4 — Step9 Source-Only Cross-Region Transfer")
    box(ax, 4, 74, 40, 12, "KAYNAK (source) bölge\nStep8A dataset\n• preprocessing burada fit\n• eşik kaynak OOF F1'den seçilir", fc="#dcefdd", ec=C_PROC, bold=True, fs=8.5)
    box(ax, 56, 74, 40, 12, "HEDEF (target) bölge\nStep8A dataset\n• yalnızca predict\n• etiket fit'e ASLA girmez", fc="#f3ddd9", ec=C_DANGER, bold=True, fs=8.5)
    box(ax, 30, 56, 40, 9, "Step9B: kaynakta eğitilen model\nhedefte tahmin üretir (iki yönlü)", fc="#ffffff", ec=C_MODEL, bold=True, fs=8.5)
    arrow(ax, (24, 74), (44, 65)); arrow(ax, (76, 74), (56, 65))
    box(ax, 4, 40, 44, 9, "Step9A: girdi uygunluk denetimi\n(shared features, populasyon, gate)", fc="#ece3f3", ec=C_DIAG, fs=8)
    box(ax, 52, 40, 44, 9, "Step9C: hedef spatial-block bootstrap\n→ delta ROC/PR/Brier %95 CI", fc="#ece3f3", ec=C_DIAG, fs=8)
    arrow(ax, (40, 56), (26, 49)); arrow(ax, (60, 56), (74, 49))
    box(ax, 25, 24, 50, 9, "Step9D: iki yönlü final rapor\noverall_conclusion (makine-okunur, değişmez)", fc="#ffffff", ec=C_MODEL, bold=True, fs=8.5)
    arrow(ax, (26, 40), (40, 33)); arrow(ax, (74, 40), (60, 33))
    box(ax, 8, 6, 84, 11,
        "YORUM KURALLARI: ROC-AUC<0.5 otomatik 'ters çevrilmez' • Brier iyileşmesi discrimination başarısı DEĞİL •\n"
        "CI 0.5/0'ı içeriyorsa 'uncertain' • negatif transfer bilimsel olarak DEĞERLİ bir sonuçtur •\n"
        "Step7 downscaling modeli transfer EDİLMEZ (her bölge kendi modeli)",
        fc="#fbe3e3", ec=C_DANGER, fs=8, bold=True)
    save(fig, "fig04_step9.png")


# ---------------------------------------------------------------------------
# FIG 5: Step10 self-calibration
# ---------------------------------------------------------------------------
def fig_step10():
    fig, ax = newfig(12, 7.2, "Şekil 5 — Step10 Self-Calibrated (label-blind) Transfer")
    box(ax, 30, 86, 40, 9, "Step10A: immutable preregistration\n+ input audit (SHA-256, analysis_id)", fc="#ffffff", ec=C_MODEL, bold=True, fs=8.5)
    box(ax, 4, 66, 44, 12, "raw_source_only\n(hiç adaptasyon yok)", fc="#f3ddd9", ec=C_DANGER, fs=8.5, bold=True)
    box(ax, 52, 66, 44, 12, "region-wise z-score\nher bölge KENDİ (etiketsiz)\nmean/std'siyle standardize", fc="#e7eef5", ec=C_INPUT, fs=8.5, bold=True)
    box(ax, 28, 50, 44, 10, "CORAL (z-score sonrası)\nkovaryans hizalama (λ=1e-5)", fc="#e7eef5", ec=C_INPUT, fs=8.5, bold=True)
    arrow(ax, (50, 86), (26, 78)); arrow(ax, (50, 86), (74, 78)); arrow(ax, (74, 66), (56, 60))
    box(ax, 20, 32, 60, 9, "Step10B: label-blind fit/adapt/predict → step10_predictions.parquet\n(HEDEF ETİKETİ İÇERMEZ)", fc="#dcefdd", ec=C_PROC, bold=True, fs=8.5)
    arrow(ax, (50, 50), (50, 41))
    box(ax, 15, 16, 70, 9, "Step10C: etiket ŞİMDİ yüklenir → eşli N-yollu spatial-block bootstrap CI\nStep10D: yalnızca yorum (hesaplama YOK)", fc="#ece3f3", ec=C_DIAG, bold=True, fs=8.5)
    arrow(ax, (50, 32), (50, 25))
    box(ax, 10, 2, 80, 8,
        "TARGET-LABEL FIREWALL: hedef etiket adaptasyon/normalizasyon/CORAL/eşik/kalibrasyon için ASLA kullanılmaz.\n"
        "Sonuç: kısmi & asimetrik toparlanma; adapted transfer hâlâ within-region'ın altında (residual gap).",
        fc="#fbe3e3", ec=C_DANGER, bold=True, fs=8)
    save(fig, "fig05_step10.png")


# ---------------------------------------------------------------------------
# FIG 6: Leakage bariyer haritası
# ---------------------------------------------------------------------------
def fig_leakage():
    fig, ax = newfig(12, 8.4, "Şekil 6 — Leakage Bariyer Haritası")
    risks = [
        ("Random row split", "spatial-block StratifiedGroupKFold", "Step8B/9B/10"),
        ("30 m piksel = label örneği", "~500 m MCD64A1 hücresine aggregation", "Step8A"),
        ("Preprocessing target'tan fit", "yalnızca source'tan fit", "Step9B/10B"),
        ("Eşik target'tan seçilir", "eşik source OOF F1'den", "Step9B"),
        ("Target-label kalibrasyon", "target-label firewall", "Step10"),
        ("Tahmin post-hoc ters çevirme", "inverse AUC yalnız diagnostic", "Step9E/G"),
        ("Row bootstrap", "spatial-block bootstrap", "Step8C/9C/10C"),
        ("Landcover skaler sıralama", "kategorik one-hot/mode", "Step8A/B"),
        ("Pre-label yanmış hücre sızıntısı", "exclude_pre_label_burns", "Muğla/Evia"),
        ("Frozen çıktı üzerine yazma", "SHA-256 + preregistration", "robustness/Step10"),
    ]
    ax.text(27, 92, "LEAKAGE RİSKİ", ha="center", fontsize=10.5, weight="bold", color=C_DANGER)
    ax.text(60, 92, "KODDAKİ BARİYER", ha="center", fontsize=10.5, weight="bold", color=C_PROC)
    ax.text(90, 92, "NEREDE", ha="center", fontsize=10.5, weight="bold", color=C_NEUTRAL)
    y = 85
    for r, b, w in risks:
        box(ax, 2, y, 48, 6.6, r, fc="#fbe3e3", ec=C_DANGER, fs=8)
        box(ax, 52, y, 30, 6.6, b, fc="#dcefdd", ec=C_PROC, fs=8)
        box(ax, 83, y, 15, 6.6, w, fc=C_LIGHT, ec=C_NEUTRAL, fs=7.5)
        arrow(ax, (50, y + 3.3), (52, y + 3.3), color="#888")
        y -= 8.1
    save(fig, "fig06_leakage.png")


# ---------------------------------------------------------------------------
# FIG 7: CLI -> modül haritası
# ---------------------------------------------------------------------------
def fig_cli():
    fig, ax = newfig(13, 9, "Şekil 7 — CLI → Modül Haritası (scripts/main.py)")
    box(ax, 38, 92, 24, 6, "scripts/main.py", fc="#dbe7f3", ec=C_INPUT, bold=True, fs=10)
    box(ax, 40, 84, 20, 5, "pipeline_orchestrator", fc=C_LIGHT, ec=C_NEUTRAL, bold=True, fs=8.5)
    arrow(ax, (50, 92), (50, 89))
    cmds = [
        ("experiment", "run_*_only.py (gate/predictors/step7/step8)", C_PROC),
        ("transfer", "run_cross_region_transfer → step9a-d", C_MODEL),
        ("shift-audit", "run_cross_region_shift_audit → step9e", C_DIAG),
        ("transfer-explore", "run_exploratory_transfer_features → step9f", C_DIAG),
        ("self-cal-transfer / step10", "run_step10_... → step10a-d", C_DIAG),
        ("step8-robustness", "run_step8_large_block_robustness", C_MODEL),
        ("large-block-robustness", "..._primary_all_valid", C_MODEL),
        ("step8-big-block-robustness", "run_step8_big_block_robustness", C_MODEL),
        ("concept-shift", "step9g_univariate... / integration_v2 / report_revision", C_DIAG),
        ("concept-shift-compare", "step9g_multi_aoi_comparison", C_DIAG),
        ("transfer-synthesis", "multi_aoi_transfer_synthesis", C_DIAG),
        ("burned-pattern-audit", "burned_pattern_audit", C_DIAG),
        ("domain-classifier-audit", "domain_classifier_audit", C_DIAG),
        ("legacy", "run_legacy_kozan_pipeline (Step1→8E, Drive)", C_DANGER),
    ]
    y = 78
    for cmd, mod, col in cmds:
        box(ax, 3, y, 26, 4.7, cmd, fc="#ffffff", ec=col, bold=True, fs=8)
        box(ax, 33, y, 64, 4.7, mod, fc=C_LIGHT, ec=col, fs=8)
        arrow(ax, (50, 84), (16, y + 4.7), color="#bbb", lw=0.8)
        arrow(ax, (29, y + 2.35), (33, y + 2.35), color=col)
        y -= 5.3
    save(fig, "fig07_cli_map.png")


# ---------------------------------------------------------------------------
# FIG 8: Output namespace ağacı
# ---------------------------------------------------------------------------
def fig_namespace():
    fig, ax = newfig(11, 8.6, "Şekil 8 — Output Namespace Ağacı")
    lines = [
        (0, "outputs/", C_NEUTRAL, True),
        (1, "experiments/<experiment_id>/   (namespaced: manavgat, bejis, mugla, evia)", C_PROC, False),
        (2, "data/  gate_inputs/  step5/  step5c/  step7a-e/  step8a-e/  validation/labels/", C_PROC, False),
        (2, "qa/{seam_audit,seam_localization,source_scene_provenance}/  robustness/step8_big_blocks/", C_PROC, False),
        (1, "kozan-legacy/   (Kozan legacy paylaşılan yollar — README'de outputs/step5 yazan yer)", C_DANGER, False),
        (2, "step1..step8e/  step6/labels/  validation/labels/", C_DANGER, False),
        (1, "cross_region/<source>__<target>/   (8 çift)", C_MODEL, False),
        (2, "step9a/ step9b/ step9c/ step9d/ step9e/ [step9f] [step10]/", C_MODEL, False),
        (1, "robustness/", C_DIAG, False),
        (2, "step8_large_block/manavgat_2021__bejis_2022/   (burnable_tree_shrub_grass)", C_DIAG, False),
        (2, "step8_large_block_primary_all_valid/manavgat_2021__bejis_2022/   (all_valid)", C_DIAG, False),
        (1, "diagnostics/", C_DIAG, False),
        (2, "step9g_..._direction_reversal/<pair>/  ..._integration_v2/<pair>/", C_DIAG, False),
        (2, "burned_pattern_audit/  domain_classifier_audit/  multi_aoi_transfer_synthesis/", C_DIAG, False),
    ]
    y = 88
    for indent, txt, col, bold in lines:
        ax.text(4 + indent * 6, y, ("└─ " if indent else "") + txt, fontsize=8.7,
                family="DejaVu Sans Mono", color=col, weight="bold" if bold else "normal", va="top")
        y -= 5.6
    box(ax, 4, 4, 92, 8,
        "KURAL: Kozan-dışı deneyler ASLA legacy paylaşılan yollara yazmaz (_assert_context_is_safely_namespaced).\n"
        "Frozen robustness çıktıları orijinal Step8A-E'yi salt-okunur korur; ayrı köke yazar.",
        fc="#fbe3e3", ec=C_DANGER, fs=8)
    save(fig, "fig08_namespace.png")


# ---------------------------------------------------------------------------
# FIG 9: Feature lineage
# ---------------------------------------------------------------------------
def fig_feature_lineage():
    fig, ax = newfig(12, 8, "Şekil 9 — Feature Soyağacı (lineage)")
    box(ax, 2, 86, 20, 8, "Landsat LST\n(30 m)", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 24, 86, 18, 8, "MODIS LST\n(1 km)", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 44, 86, 16, 8, "Landsat\nNDVI", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 62, 86, 16, 8, "DEM\n(Copernicus)", fc="#dbe7f3", ec=C_INPUT, fs=8)
    box(ax, 80, 86, 18, 8, "ESA\nWorldCover", fc="#dbe7f3", ec=C_INPUT, fs=8)
    derived = [
        (2, 66, "Step5\nlst_anomaly_mean\n(current − baseline z)", C_PROC),
        (24, 66, "Step7C/D/E\ndownscaled_lst_mean\nfused_lst_mean", C_PROC),
        (44, 66, "Step5C TVDI\ncurrent_tvdi_mean\ntvdi_difference_mean", C_PROC),
        (62, 66, "elevation_mean\nslope_mean", C_PROC),
        (80, 66, "landcover_dominant\n(kategorik)", C_PROC),
    ]
    for x, y, t, c in derived:
        box(ax, x, y, 18, 11, t, ec=c, fs=7.8)
    arrow(ax, (12, 86), (11, 77)); arrow(ax, (33, 86), (33, 77))
    arrow(ax, (33, 86), (52, 77)); arrow(ax, (52, 86), (52, 77))
    arrow(ax, (70, 86), (71, 77)); arrow(ax, (89, 86), (89, 77))
    arrow(ax, (11, 86), (33, 77))  # landsat->downscaled target
    box(ax, 12, 44, 34, 9, "current_lst_mean\n(gözlemlenen current-period LST)", fc="#fde9d9", ec=C_MODEL, fs=8)
    arrow(ax, (11, 86), (20, 53))
    box(ax, 20, 26, 60, 9, "Step8A: 30 m → ~500 m MCD64A1 hücresi\n(sürekli: mean/median • kategorik: mode/fraction)", fc="#fde9d9", ec=C_MODEL, bold=True, fs=8.5)
    for x in (11, 33, 52, 71, 89, 29):
        arrow(ax, (x, 66 if x != 29 else 44), (50, 35), color="#bbb", lw=0.8)
    box(ax, 15, 10, 70, 9, "Step8B baseline (4): ndvi, elevation, slope, landcover  |  +thermal (6): lst_anomaly,\ncurrent_lst, current_tvdi, tvdi_difference, downscaled_lst, fused_lst", fc="#dcefdd", ec=C_PROC, bold=True, fs=8.3)
    arrow(ax, (50, 26), (50, 19))
    save(fig, "fig09_feature_lineage.png")


# ---------------------------------------------------------------------------
# FIG 10: Diagnostic soru haritası
# ---------------------------------------------------------------------------
def fig_diagnostics():
    fig, ax = newfig(12, 7.6, "Şekil 10 — Teşhis Soru Haritası (her diagnostic neyi yanıtlar)")
    items = [
        ("Step9E dağılım-kayması", "Feature dağılımları/ilişkileri bölgeler arasında ne kadar kayıyor?"),
        ("Step9F feature-representation", "Sabit feature altkümeleri / region-relative temsil transferi düzeltir mi? (kesifsel)"),
        ("Step9G univariate AUC yönü", "Hangi feature'ın burned ile ilişkisi ters dönüyor? (yalnız elevation bootstrap-destekli)"),
        ("concept-shift-compare", "Univariate reversal'lar çoklu AOI çiftinde tutarlı mı? (report-only)"),
        ("domain-classifier-audit", "İki bölge yalnız predictor'larla ne kadar ayırt edilebilir? (covariate separability ≈1.0)"),
        ("burned-pattern-audit", "Yanmış alanın mekansal yapısı/geometrisi/landcover'ı bölgeler arası nasıl farklı?"),
        ("large/big-block robustness", "Within-region thermal katkı daha büyük mekansal bloklarda korunuyor mu?"),
        ("transfer-synthesis", "Tüm within/raw/adapted/feature-stability bulguları tek tabloda ne söylüyor?"),
    ]
    y = 84
    for name, q in items:
        box(ax, 3, y, 30, 8, name, fc="#ece3f3", ec=C_DIAG, bold=True, fs=8.3)
        box(ax, 35, y, 62, 8, q, fc="#ffffff", ec=C_NEUTRAL, fs=8)
        arrow(ax, (33, y + 4), (35, y + 4), color=C_DIAG)
        y -= 9.8
    box(ax, 6, 2, 88, 6,
        "AYRIM: domain separability ≠ transfer success ≠ feature-label ilişki stabilitesi ≠ fire-footprint geometrisi. Hiçbiri nedensel değil.",
        fc="#fbe3e3", ec=C_DANGER, bold=True, fs=8)
    save(fig, "fig10_diagnostics.png")


# ---------------------------------------------------------------------------
# FIG 12: Covariate shift vs relationship shift
# ---------------------------------------------------------------------------
def fig_shift_concept():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    rng = np.random.default_rng(7)
    # Covariate shift
    a = ax[0]
    xs = np.linspace(-4, 8, 200)
    def g(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    a.fill_between(xs, g(xs, 0, 1), alpha=0.4, color=C_INPUT, label="Kaynak feature dağılımı")
    a.fill_between(xs, g(xs, 3.5, 1.2), alpha=0.4, color=C_MODEL, label="Hedef feature dağılımı")
    a.set_title("Covariate shift\nP(X) değişir, P(y|X) AYNI kalır", fontsize=11, weight="bold")
    a.legend(fontsize=8, loc="upper right")
    a.set_xlabel("feature değeri"); a.set_yticks([])
    a.text(0.5, -0.22, "Step10 (z-score/CORAL) bunu hedefler; domain AUC≈1.0 bunu doğrular",
           transform=a.transAxes, ha="center", fontsize=8.5, style="italic", color=C_NEUTRAL)
    # Relationship shift
    b = ax[1]
    x = np.linspace(0, 10, 40)
    b.scatter(x, 0.35 * x + rng.normal(0, 0.6, 40), color=C_INPUT, s=16, label="Kaynak: pozitif ilişki")
    b.scatter(x, -0.35 * x + 3.5 + rng.normal(0, 0.6, 40), color=C_DANGER, s=16, marker="^", label="Hedef: NEGATİF ilişki")
    b.plot(x, 0.35 * x, color=C_INPUT, lw=2)
    b.plot(x, -0.35 * x + 3.5, color=C_DANGER, lw=2, ls="--")
    b.set_title("Relationship / concept shift\nP(y|X) YÖNÜ değişir (ör. elevation_mean)", fontsize=11, weight="bold")
    b.legend(fontsize=8, loc="upper center")
    b.set_xlabel("feature (ör. elevation)"); b.set_ylabel("burned eğilimi"); b.set_yticks([])
    b.text(0.5, -0.22, "z-score/CORAL bunu DÜZELTEMEZ; residual gap ile tutarlı (Step9E/9G)",
           transform=b.transAxes, ha="center", fontsize=8.5, style="italic", color=C_NEUTRAL)
    fig.suptitle("Şekil 12 — Covariate Shift vs. Relationship (Concept) Shift", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, "fig12_shift_concept.png")


# ---------------------------------------------------------------------------
# FIG 11 + veri-güdümlü sonuç grafikleri
# ---------------------------------------------------------------------------
def fig_status_map():
    fig, ax = newfig(12, 6.6, "Şekil 11 — Güncel AOI / Aşama Durum Haritası")
    exps = ["kozan_2023", "manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021"]
    stages = ["Gate", "Predict", "Step7", "Step8", "Robust", "Step9", "Step10", "Diag"]
    # status matrix: 2=done,1=partial/pair-only,0=n/a
    M = {
        "kozan_2023":    [2, 2, 2, 2, 0, 0, 0, 1],   # legacy; control
        "manavgat_2021": [2, 2, 2, 2, 2, 2, 2, 2],
        "bejis_2022":    [2, 2, 2, 2, 2, 2, 2, 2],
        "mugla_2021":    [2, 2, 2, 2, 1, 2, 2, 2],
        "evia_2021":     [2, 2, 2, 2, 1, 2, 0, 1],
    }
    colmap = {2: C_PROC, 1: C_MODEL, 0: "#cccccc"}
    lbl = {2: "✓", 1: "~", 0: "–"}
    x0, y0, cw, ch = 22, 12, 9, 8
    for j, s in enumerate(stages):
        ax.text(x0 + j * cw + cw / 2, y0 + len(exps) * ch + 3, s, ha="center", fontsize=8.5, weight="bold", rotation=0)
    for i, e in enumerate(exps):
        yy = y0 + (len(exps) - 1 - i) * ch
        ax.text(x0 - 2, yy + ch / 2, e, ha="right", va="center", fontsize=9, weight="bold")
        for j in range(len(stages)):
            v = M[e][j]
            box(ax, x0 + j * cw, yy, cw - 1, ch - 1, lbl[v], fc=colmap[v], tc="white", bold=True, fs=11)
    ax.text(22, 6, "✓ tamam   ~ kısmi/çift-bazlı veya kontrol   – uygulanamaz.  "
                   "Kozan = negative control (cropland). Robust: manavgat/bejis frozen; mugla/evia big-block.",
            fontsize=8, color=C_NEUTRAL)
    save(fig, "fig11_status_map.png")


def fig_within_region_bars():
    data = {}
    import glob
    for f in sorted(glob.glob(str(ROOT / "outputs/experiments/*/step8b/step8b_model_comparison_metrics.json"))):
        exp = f.split("/experiments/")[1].split("/")[0]
        d = json.load(open(f))
        pm = d.get("population_metrics", {}).get("burnable_tree_shrub_grass")
        if pm:
            data[exp] = (pm["overall_baseline"]["roc_auc"], pm["overall_thermal"]["roc_auc"],
                         pm["overall_baseline"]["pr_auc"], pm["overall_thermal"]["pr_auc"])
    exps = list(data.keys())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(exps)); w = 0.36
    for ax, idx, title in [(axs[0], (0, 1), "ROC-AUC (within-region)"), (axs[1], (2, 3), "PR-AUC (within-region)")]:
        ax.bar(x - w/2, [data[e][idx[0]] for e in exps], w, label="baseline", color=C_INPUT)
        ax.bar(x + w/2, [data[e][idx[1]] for e in exps], w, label="+thermal", color=C_MODEL)
        ax.set_xticks(x); ax.set_xticklabels(exps, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, weight="bold"); ax.legend(fontsize=8)
        if "ROC" in title:
            ax.axhline(0.5, color=C_DANGER, ls="--", lw=1, alpha=0.7)
        ax.set_ylim(0, 1)
    fig.suptitle("Şekil 13 — Within-Region Baseline vs. Thermal (population: burnable_tree_shrub_grass)", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.005, "Kaynak: outputs/experiments/<exp>/step8b/step8b_model_comparison_metrics.json", ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    save(fig, "fig13_within_region_bars.png")


def fig_transfer_heatmap():
    d = json.load(open(ROOT / "outputs/diagnostics/multi_aoi_transfer_synthesis/bejis_2022__manavgat_2021__mugla_2021/multi_aoi_transfer_synthesis.json"))
    tm = d.get("transfer_matrix", {})
    # build from step9d directly for full 4-AOI incl evia
    import glob
    exps = ["manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021"]
    mat = np.full((4, 4), np.nan)
    for f in glob.glob(str(ROOT / "outputs/cross_region/*/step9d/final_cross_region_report.json")):
        rep = json.load(open(f))
        for ds in rep.get("direction_summaries", []):
            s = ds["source_experiment_id"]; t = ds["target_experiment_id"]
            tm2 = ds.get("thermal_target_metrics", {})
            if s in exps and t in exps and "roc_auc" in tm2:
                mat[exps.index(s), exps.index(t)] = tm2["roc_auc"]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.30, vmax=0.70)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(exps, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticklabels(exps, fontsize=8.5)
    ax.set_xlabel("HEDEF (target)", fontsize=9, weight="bold")
    ax.set_ylabel("KAYNAK (source)", fontsize=9, weight="bold")
    for i in range(4):
        for j in range(4):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9,
                        color="black", weight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="#999")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("raw transfer thermal ROC-AUC", fontsize=8.5)
    ax.set_title("Şekil 14 — Raw Cross-Region Transfer ROC-AUC Matrisi\n(0.50 = şans; kırmızı ≤ şans)", fontsize=11.5, weight="bold")
    fig.text(0.5, 0.005, "Kaynak: outputs/cross_region/<pair>/step9d/final_cross_region_report.json (population: burnable_tree_shrub_grass)",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig14_transfer_heatmap.png")


def fig_step10_recovery():
    import glob
    pairs = {}
    for f in sorted(glob.glob(str(ROOT / "outputs/cross_region/*/step10/step10_metrics.json"))):
        pair = f.split("/cross_region/")[1].split("/")[0]
        d = json.load(open(f))
        for dirn, md in d.get("point_metrics", {}).items():
            methods = ["raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore"]
            vals = [md.get(m, {}).get("thermal", {}).get("roc_auc") for m in methods]
            if all(v is not None for v in vals):
                pairs[dirn] = vals
    # unique directions
    dirs = list(pairs.keys())
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(dirs)); w = 0.26
    labels = ["raw", "z-score", "CORAL"]
    cols = [C_DANGER, C_INPUT, C_PROC]
    for k in range(3):
        ax.bar(x + (k - 1) * w, [pairs[dd][k] for dd in dirs], w, label=labels[k], color=cols[k])
    ax.axhline(0.5, color="black", ls="--", lw=1.2, label="şans (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([dd.replace("_2021", "").replace("_2022", "").replace("_to_", "→\n") for dd in dirs],
                       fontsize=7, rotation=0)
    ax.set_ylabel("thermal ROC-AUC", fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("Şekil 15 — Step10 Adaptasyon Toparlanması (thermal ROC-AUC; her yön)", fontsize=12, weight="bold")
    fig.text(0.5, 0.005, "Kaynak: outputs/cross_region/<pair>/step10/step10_metrics.json — toparlanma kısmi & asimetrik; çoğu yön hâlâ şans civarı.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig15_step10_recovery.png")


if __name__ == "__main__":
    fig_mental_model()
    fig_dataflow()
    fig_lifecycle()
    fig_step8()
    fig_step9()
    fig_step10()
    fig_leakage()
    fig_cli()
    fig_namespace()
    fig_feature_lineage()
    fig_diagnostics()
    fig_shift_concept()
    fig_status_map()
    fig_within_region_bars()
    fig_transfer_heatmap()
    fig_step10_recovery()
    print("ALL FIGURES DONE")

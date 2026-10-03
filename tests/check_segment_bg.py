"""验证颜色聚类背景剔除：背景不占编号、纯色前景保留、三处口径一致。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from server import pipeline as pipeline_engine  # noqa: E402
from server.algorithms import segmentation  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
    return cond


def white_bg_shapes():
    img = Image.new("RGB", (320, 240), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 110, 110], fill=(220, 30, 30))      # 纯红方块
    d.ellipse([180, 60, 260, 140], fill=(30, 30, 200))       # 纯蓝椭圆
    return img


def blue_bg_border_object():
    # 蓝底 + 一个贴边的巨大纯色前景（红矩形贴左边）
    img = Image.new("RGB", (320, 240), (40, 90, 220))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 60, 150, 200], fill=(230, 40, 40))
    return img


def main():
    ok = True

    print("== 白底 + 两个纯色形状 ==")
    r = segmentation.segment(white_bg_shapes(), {"method": "color", "colors": 6})
    print(f"  区域数={r['region_count']} 覆盖率={r['coverage']:.3f} 背景占比={r['background_coverage']:.3f}")
    for rg in r["regions"]:
        print(f"    区域 #{rg['id']} 面积={rg['area']} 主色={rg['mean_color']}")
    ok &= check("覆盖率远小于 100%", r["coverage"] < 0.6, f"({r['coverage']:.3f})")
    ok &= check("背景占大头", r["background_coverage"] > 0.4)
    ok &= check("区域编号连续从 1 开始", [rg["id"] for rg in sorted(r["regions"], key=lambda x: x["id"])] == list(range(1, r["region_count"] + 1)))
    colors = {tuple(round(c / 32) for c in rg["mean_color"]) for rg in r["regions"]}
    ok &= check("没有近白色区域混进来", all(not (c[0] >= 7 and c[1] >= 7 and c[2] >= 7) for c in colors), f"({colors})")

    print("== 蓝底 + 贴边纯色前景 ==")
    r2 = segmentation.segment(blue_bg_border_object(), {"method": "color", "colors": 6})
    print(f"  区域数={r2['region_count']} 覆盖率={r2['coverage']:.3f} 背景占比={r2['background_coverage']:.3f}")
    for rg in r2["regions"]:
        print(f"    区域 #{rg['id']} 面积={rg['area']} 主色={rg['mean_color']}")
    red_kept = any(rg["mean_color"][0] > 150 and rg["mean_color"][1] < 100 for rg in r2["regions"])
    blue_gone = all(not (rg["mean_color"][2] > 150 and rg["mean_color"][0] < 100) for rg in r2["regions"])
    ok &= check("贴边的纯红前景被保留", red_kept)
    ok &= check("蓝底未占编号", blue_gone)
    ok &= check("覆盖率≈前景占比", 0.2 < r2["coverage"] < 0.6, f"({r2['coverage']:.3f})")

    print("== 纯单色图（整图皆背景）==")
    r3 = segmentation.segment(Image.new("RGB", (200, 150), (255, 255, 255)), {"method": "color", "colors": 6})
    print(f"  区域数={r3['region_count']} 覆盖率={r3['coverage']:.3f} 背景占比={r3['background_coverage']:.3f}")
    ok &= check("无区域、覆盖率 0", r3["region_count"] == 0 and r3["coverage"] == 0)

    print("== 三处口径一致：分割函数 vs 滤镜链节点 ==")
    img = white_bg_shapes()
    direct = segmentation.segment(img, {"method": "color", "colors": 6, "alpha": 0.45})
    nodes = [{"id": "n1", "type": "segment",
              "params": {"method": "color", "colors": 6, "alpha": 0.45,
                         "value": 127, "block": 15},
              "inputs": []}]
    errors = pipeline_engine.validate(nodes)
    out = pipeline_engine.execute(img, nodes)
    meta = out["meta"]
    print(f"  直接调用: 区域={direct['region_count']} 覆盖率={direct['coverage']:.4f} 背景={direct['background_coverage']:.4f}")
    print(f"  滤镜链  : 区域={meta['region_count']} 覆盖率={meta['coverage']:.4f} 背景={meta['background_coverage']:.4f}")
    ok &= check("链路校验无错", not errors, f"({errors})")
    ok &= check("节点与分割页统计一致",
                meta["region_count"] == direct["region_count"]
                and meta["coverage"] == direct["coverage"]
                and meta["background_coverage"] == direct["background_coverage"])

    print("== 阈值/区域生长回归（背景本就不占编号）==")
    for m in ("threshold", "region"):
        rm = segmentation.segment(white_bg_shapes(), {"method": m})
        print(f"  {m}: 区域数={rm['region_count']} 覆盖率={rm['coverage']:.3f} 背景占比={rm['background_coverage']:.3f}")
        ok &= check(f"{m} 仍正常出区域", rm["region_count"] >= 1)
        ok &= check(f"{m} 覆盖率+背景占比≈1", abs(rm["coverage"] + rm["background_coverage"] - 1) < 0.05)

    print("\n" + ("全部通过 ✔" if ok else "存在失败项 �’"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

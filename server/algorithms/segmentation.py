"""图像分割：阈值分割 / 区域生长 / 颜色量化聚类。

纯 Pillow 实现：

- threshold：全局（Otsu 自动或手定）二值化 -> 前景/背景两个区域。
- region   ：自适应局部阈值 -> 二值 -> 连通域，得到多个空间区域。
- color    ：颜色量化（中位切分）-> 每个主色掩码做连通域 -> 颜色聚类区域。

背景口径（三种方法一致）：标签 0 永远是背景/未分类，不进入区域列表、
不计入覆盖率；叠加层里背景透传原图，只有前景区域被着色。
color 方法额外用「边界先验」识别纯色背景：主导画面四周边框的量化色
视为背景色，贴边的该色连通域判为背景（可用 remove_bg=False 关闭）；
被其他颜色包围、不贴边的同色连通域仍算前景，避免误删纯色前景物体。

输出：半透明彩色覆盖层（每区域一色）+ 区域边界 + 区域统计（数量/覆盖率/最大区域）。
"""
import colorsys
from collections import Counter

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .. import config
from . import util


_PALETTE = [
    (244, 67, 54), (33, 150, 243), (255, 193, 7), (76, 175, 80),
    (156, 39, 176), (0, 188, 212), (255, 87, 34), (63, 81, 181),
    (255, 235, 59), (0, 150, 136), (233, 30, 99), (121, 85, 72),
]


def _otsu(gray):
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return 127
    sum_all = sum(i * c for i, c in enumerate(hist))
    w_b = 0.0
    sum_b = 0.0
    best_t, best_v = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        v = w_b * w_f * (m_b - m_f) ** 2
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def _binary_mask(gray, params):
    """根据 method 生成二值掩码（L 图像，255 为前景）。"""
    method = params.get("method", "threshold")
    if method == "region":
        block = float(params.get("block", 15))
        local = gray.filter(ImageFilter.BoxBlur(block / 2.0))
        return ImageChops.subtract(gray, local).point(lambda v: 255 if v >= 0 else 0)
    value = params.get("value", None)
    if value is None:
        value = _otsu(gray)
    return gray.point(lambda v: 255 if v >= int(value) else 0)


def _labels_to_image(labels, w, h, region_colors, base=None):
    """把标签矩阵渲染成彩色区域图（RGB）。region_colors: {label: (r,g,b)}。

    base 提供时，标签 0（背景/未分类）透传原图像素，
    让叠加层里只有真正的前景区域被着色。
    """
    base_px = base.load() if base is not None else None
    data = []
    for y in range(h):
        for x in range(w):
            lbl = labels[y][x]
            if not lbl and base_px is not None:
                data.append(base_px[x, y])
            else:
                data.append(region_colors.get(lbl, (0, 0, 0)))
    img = Image.new("RGB", (w, h))
    img.putdata(data)
    return img


def _components_from_mask(mask):
    w, h, rows = util.gray_matrix(mask)
    return util.connected_components(rows, w, h, threshold=128)


def _region_stats(components, w, h, orig_work):
    regions = []
    for label, pts in components.items():
        x0, y0, x1, y1 = util.points_bbox(pts)
        area = len(pts)
        mean = (0, 0, 0)
        try:
            crop = orig_work.crop((x0, y0, x1 + 1, y1 + 1)).resize((1, 1), Image.Resampling.BILINEAR)
            mean = crop.getpixel((0, 0))
        except Exception:
            pass
        regions.append({
            "id": label, "area": area, "coverage": round(area / float(w * h), 4),
            "box": [x0, y0, x1 - x0, y1 - y0], "mean_color": list(mean),
        })
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def segment(image, params):
    """执行分割，返回 overlay + 区域统计。"""
    method = params.get("method", "threshold")
    orig = util.ensure_rgb(image)
    work = util.downscale_to_max(orig, config.FEATURE_WORK_DIM)
    ratio = util.scale_ratio(orig.size, work.size)
    w, h = work.size

    bg_info = {"area": 0, "colors": []}
    if method == "color":
        labels, components, bg_info = _color_clustering(
            work, int(params.get("colors", 6)), bool(params.get("remove_bg", True)))
    else:
        gray = util.to_grayscale(work)
        mask = _binary_mask(gray, params)
        labels, components = _components_from_mask(mask)

    region_colors = {0: (0, 0, 0)}
    for i, label in enumerate(components.keys(), start=1):
        region_colors[label] = _PALETTE[i % len(_PALETTE)]

    # 标签 0（背景/未分类）透传原图，只有前景区域着色
    color_map = _labels_to_image(labels, w, h, region_colors, base=work)
    # 边界：区域图边缘检测
    boundaries = color_map.filter(ImageFilter.FIND_EDGES).point(lambda v: 0 if v < 30 else v)
    color_map = Image.blend(color_map, boundaries.convert("RGB"), 0.35)

    # 半透明叠加回原图
    overlay = Image.blend(work, color_map, float(params.get("alpha", 0.45)))
    overlay = overlay.resize(orig.size, Image.Resampling.BILINEAR)

    regions = _region_stats(components, w, h, work)
    for r in regions:
        r["box"] = [int(round(v * ratio)) for v in r["box"]]
        r["area"] = int(round(r["area"] * ratio * ratio))

    foreground = sum(r["area"] for r in regions)
    labeled = sum(len(pts) for pts in components.values())
    if method == "color":
        background = bg_info["area"]
    else:
        background = max(w * h - labeled, 0)  # 掩码外像素即背景
    return {
        "method": method,
        "region_count": len(regions),
        "coverage": round(foreground / float(orig.size[0] * orig.size[1]), 4),
        "background_coverage": round(background / float(w * h), 4),
        "background_colors": bg_info["colors"],
        "regions": regions,
        "image": overlay,
    }


def _border_background_colors(rows, w, h, min_share=0.25):
    """边界先验：统计画面四周边框上的颜色，占比 >= min_share 的判为背景色。

    纯色背景（白底/蓝底）几乎占满整个边框；纯色前景物体即使贴边，
    一般也只占边框的一小段，不会被误判为背景色。
    """
    counter = Counter()
    for x in range(w):
        counter[rows[0][x]] += 1
        counter[rows[h - 1][x]] += 1
    for y in range(1, h - 1):
        counter[rows[y][0]] += 1
        counter[rows[y][w - 1]] += 1
    frame = max(2 * w + 2 * h - 4, 1)
    return {c for c, n in counter.items() if n / float(frame) >= min_share}


def _touches_frame(pts, w, h):
    """连通域是否贴到画面四周边框。"""
    xm, ym = w - 1, h - 1
    for x, y in pts:
        if x == 0 or y == 0 or x == xm or y == ym:
            return True
    return False


def _color_clustering(rgb, n_colors, remove_bg=True):
    """颜色量化 + 每主色连通域，返回 (labels, components, bg_info)。

    remove_bg=True 时按边界先验剔除背景：主导四周边框的量化色视为背景色，
    贴边的背景色连通域不分配编号（保持标签 0，不计入区域/覆盖率）；
    被其他颜色包围、不贴边的同色连通域仍算前景，避免误删纯色前景物体
    （如蓝底上的白色产品：白色未主导边框，始终保留）。
    """
    quantized = rgb.quantize(colors=max(2, n_colors), method=Image.Quantize.MEDIANCUT).convert("RGB")
    w, h, qrows = util.rgb_matrix(quantized)
    # 统计出现频率最高的颜色
    counter = Counter(qrows[y][x] for y in range(h) for x in range(w))
    target_colors = [c for c, _ in counter.most_common(n_colors + 2)]
    bg_colors = _border_background_colors(qrows, w, h) if remove_bg else set()

    labels = [[0] * w for _ in range(h)]
    components = {}
    bg_area = 0
    next_label = 0
    for color in target_colors:
        # 该颜色的二值掩码
        mask_rows = [[255 if qrows[y][x] == color else 0 for x in range(w)] for y in range(h)]
        _, comps = util.connected_components(mask_rows, w, h, threshold=128)
        for _lbl, pts in comps.items():
            if len(pts) < (w * h) * 0.002:  # 过滤过小区域
                continue
            if color in bg_colors and _touches_frame(pts, w, h):
                bg_area += len(pts)  # 背景：保持标签 0，不占区域编号
                continue
            next_label += 1
            for px, py in pts:
                labels[py][px] = next_label
            components[next_label] = pts
    bg_info = {"area": bg_area, "colors": [list(c) for c in sorted(bg_colors)]}
    return labels, components, bg_info


def draw_region_outline(image, boxes, color=(255, 255, 255)):
    """在图上描出区域外接框（调试/展示用）。"""
    img = util.ensure_rgb(image).copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x, y, w, h = b
        draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
    return img

# -*- coding: utf-8 -*-
"""论文图导出兼容层。

项目原本依赖 math-modeling skill 的 ``setup_style`` / ``export_figure``
（位于 skill 目录 tools/figure/scripts 下）。该 skill 不在项目内时，
本模块提供同签名的降级实现，使绘图脚本仍可在本机直接运行。
"""
from pathlib import Path

import numpy as np


def setup_style(journal="general", lang="zh", serif_for_zh=True, **kwargs):
    """应用出版级 matplotlib 样式；返回 rcParams 字典。"""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                            "Source Han Sans SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })
    return dict(mpl.rcParams)


def export_figure(fig, basename, formats=("svg", "png"), size_inches=None,
                  dpi=300, grayscale_preview=True, **kwargs):
    """按原 skill 接口导出图形，并返回路径字典（键 svg / png / grayscale）。"""
    import matplotlib.image as image_io

    if dpi < 300:
        raise ValueError("论文图 PNG 的 dpi 不能低于 300")
    if size_inches is not None:
        fig.set_size_inches(*size_inches)

    stem = Path(basename)
    if stem.suffix.lower() in (".svg", ".png", ".pdf"):
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    outputs = {}
    if "svg" in formats:
        path = stem.with_suffix(".svg")
        fig.savefig(path)
        outputs["svg"] = str(path)
    if "png" in formats:
        path = stem.with_suffix(".png")
        fig.savefig(path, dpi=dpi)
        outputs["png"] = str(path)
        if grayscale_preview:
            pixels = image_io.imread(path)
            rgb = pixels[..., :3]
            if pixels.shape[-1] == 4:
                alpha = pixels[..., 3:4]
                rgb = rgb * alpha + (1 - alpha)
            gray = np.dot(rgb, (0.2126, 0.7152, 0.0722))
            gpath = stem.parent / (stem.name + "_grayscale.png")
            image_io.imsave(gpath, gray, cmap="gray", vmin=0, vmax=1, dpi=dpi)
            outputs["grayscale"] = str(gpath)
    return outputs


__all__ = ["setup_style", "export_figure"]

import io
import json
from html import escape
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from model.stage3_config import DEFAULT_STAGE3_CHECKPOINT, DEFAULT_STAGE3_DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
MAX_HISTORY_ITEMS = 12
SHADOW_TTA_SWEEP_PATH = Path("runs/stage2/shadow_tta_v6_20260422/shadow_tta_param_sweep.json")

APP_CSS = """
<style>
  :root {
    color-scheme: light;
    --ios-text: #1d1d1f;
    --ios-muted: #6e6e73;
    --ios-blue: #0a84ff;
    --ios-blue-soft: #64d2ff;
    --ios-indigo: #5e5ce6;
    --ios-success: #22c55e;
    --ios-warning: #f59e0b;
    --ios-slate: #475569;
    --ios-shadow: 0 18px 42px rgba(0, 0, 0, 0.06);
    --ios-shadow-soft: 0 10px 26px rgba(0, 0, 0, 0.045);
    --ios-border: rgba(217, 217, 223, 0.95);
    --ios-panel: rgba(255, 255, 255, 0.88);
    --ios-panel-strong: rgba(255, 255, 255, 0.97);
    --ios-blue-tint: rgba(10, 132, 255, 0.10);
    --ios-blue-tint-strong: rgba(10, 132, 255, 0.16);
    --ios-code-bg: #171923;
    --ios-code-text: #f5f5f7;
    --apple-page: #f5f5f7;
  }

  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "PingFang SC", "Helvetica Neue", sans-serif;
  }

  .stApp {
    color: var(--ios-text);
    background:
      radial-gradient(circle at 12% 8%, rgba(10, 132, 255, 0.07), transparent 24%),
      radial-gradient(circle at 88% 84%, rgba(94, 92, 230, 0.05), transparent 20%),
      linear-gradient(180deg, #fbfbfd 0%, var(--apple-page) 100%);
  }

  .main .block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 2.2rem;
  }

  .stApp,
  .stApp p,
  .stApp label,
  .stApp li,
  .stApp span,
  .stApp div,
  .stApp small,
  .stMarkdown,
  .stMarkdown p,
  .stMarkdown li,
  div[data-testid="stMarkdownContainer"],
  div[data-testid="stMarkdownContainer"] *,
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] *,
  [data-testid="stText"],
  [data-testid="stText"] *,
  [data-testid="stCaptionContainer"] *,
  [data-testid="stExpander"] *,
  [data-testid="stSidebar"] *,
  [data-testid="stSidebarNav"] *,
  [data-testid="stFileUploader"] *,
  [data-testid="stCameraInput"] *,
  [data-testid="stForm"] *,
  [data-testid="stHorizontalBlock"] *,
  [data-testid="column"] * {
    color: var(--ios-text) !important;
  }

  .stApp small,
  .stApp [data-testid="stCaptionContainer"] *,
  .stApp .muted-note {
    color: var(--ios-muted) !important;
  }

  h1, h2, h3 {
    color: var(--ios-text) !important;
    letter-spacing: -0.03em;
  }

  a,
  a * {
    color: var(--ios-blue) !important;
  }

  code:not(pre code) {
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    color: var(--ios-code-text) !important;
    background: linear-gradient(180deg, #222634, var(--ios-code-bg)) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
  }

  section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, rgba(251, 251, 253, 0.96), rgba(246, 246, 248, 0.94));
    border-right: 1px solid rgba(229, 229, 234, 0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
    padding: 0.45rem;
    border-radius: 1.3rem;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(229, 229, 234, 0.98);
    box-shadow: var(--ios-shadow-soft);
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    border-radius: 0.95rem;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
    background: var(--ios-blue-tint);
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(180deg, rgba(10, 132, 255, 0.14), rgba(100, 210, 255, 0.10));
    box-shadow: inset 0 0 0 1px rgba(10, 132, 255, 0.08);
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {
    color: var(--ios-text) !important;
    font-weight: 600;
  }

  .hero-card {
    padding: 1.55rem 1.65rem;
    border-radius: 1.8rem;
    color: var(--ios-text);
    border: 1px solid rgba(228, 228, 232, 0.96);
    background:
      radial-gradient(circle at top right, rgba(100, 210, 255, 0.18), transparent 24%),
      radial-gradient(circle at bottom left, rgba(10, 132, 255, 0.08), transparent 22%),
      linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 255, 0.96));
    box-shadow: 0 22px 48px rgba(0, 0, 0, 0.06);
    margin-bottom: 1rem;
  }

  .hero-card h1,
  .hero-card h2,
  .hero-card h3,
  .hero-card p {
    color: var(--ios-text) !important;
  }

  .hero-card p:first-child {
    color: var(--ios-blue) !important;
  }

  .hero-card p:last-child {
    color: var(--ios-muted) !important;
  }

  div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
  }

  [data-testid="column"] > div[data-testid="stVerticalBlock"] {
    gap: 0.95rem;
  }

  .section-header {
    margin: 0.3rem 0 0.35rem 0;
  }

  .section-eyebrow {
    margin: 0;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ios-blue) !important;
  }

  .section-title {
    margin: 0.28rem 0 0 0;
    font-size: 1.22rem;
    line-height: 1.2;
    letter-spacing: -0.03em;
    color: var(--ios-text) !important;
  }

  .section-copy {
    margin: 0.45rem 0 0 0;
    max-width: 46rem;
    line-height: 1.68;
    color: var(--ios-muted) !important;
  }

  .muted-note {
    color: var(--ios-muted) !important;
    line-height: 1.7;
  }

  .soft-card {
    padding: 1rem 1.1rem;
    border-radius: 1.3rem;
    border: 1px solid var(--ios-border);
    background: var(--ios-panel-strong);
    box-shadow: var(--ios-shadow);
    backdrop-filter: blur(14px) saturate(160%);
    -webkit-backdrop-filter: blur(14px) saturate(160%);
  }

  .panel-card {
    position: relative;
    padding: 1.1rem 1.15rem;
    border-radius: 1.45rem;
    border: 1px solid var(--ios-border);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.95));
    box-shadow: var(--ios-shadow);
    overflow: hidden;
  }

  .panel-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: linear-gradient(180deg, var(--ios-blue), var(--ios-blue-soft));
  }

  .panel-card.is-success::before {
    background: linear-gradient(180deg, #16a34a, #4ade80);
  }

  .panel-card.is-warn::before {
    background: linear-gradient(180deg, #d97706, #fbbf24);
  }

  .panel-card.is-violet::before {
    background: linear-gradient(180deg, #5e5ce6, #8b5cf6);
  }

  .panel-eyebrow {
    margin: 0 0 0.35rem 0;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ios-blue) !important;
  }

  .panel-title {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--ios-text) !important;
  }

  .panel-body {
    margin-top: 0.75rem;
    line-height: 1.75;
  }

  .panel-footer {
    margin-top: 0.9rem;
  }

  .mac-window {
    position: relative;
    margin: 0.2rem 0 1rem 0;
  }

  .mac-window-toggle {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  .mac-window-stage {
    max-height: 42rem;
    overflow: visible;
    perspective: 1400px;
    perspective-origin: 50% 100%;
    transition: max-height 0.74s cubic-bezier(0.22, 1, 0.36, 1), margin 0.42s ease;
  }

  .mac-window-shell {
    position: relative;
    overflow: hidden;
    border-radius: 1.45rem;
    border: 1px solid var(--ios-border);
    background:
      radial-gradient(circle at 96% 0%, rgba(100, 210, 255, 0.15), transparent 26%),
      linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(247, 250, 255, 0.96));
    box-shadow: var(--ios-shadow);
    transform-origin: 50% 100%;
    transform-style: preserve-3d;
    clip-path: inset(0 0 0 0 round 1.45rem);
    will-change: transform, clip-path, opacity, filter;
    transition:
      transform 0.74s cubic-bezier(0.22, 1, 0.36, 1),
      opacity 0.38s ease,
      filter 0.56s ease,
      clip-path 0.74s cubic-bezier(0.22, 1, 0.36, 1),
      box-shadow 0.56s ease;
  }

  .mac-window-shell::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.22), transparent 34%);
  }

  .mac-window-titlebar {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.72rem 0.85rem;
    border-bottom: 1px solid rgba(226, 232, 240, 0.92);
    background: linear-gradient(180deg, rgba(253, 253, 255, 0.98), rgba(244, 247, 251, 0.96));
  }

  .mac-window-traffic {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    flex: 0 0 auto;
  }

  .mac-window-dot {
    width: 0.78rem;
    height: 0.78rem;
    border-radius: 999px;
    display: inline-block;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
  }

  .mac-window-dot.red {
    cursor: pointer;
    background: linear-gradient(180deg, #ff7b72, #f04438);
  }

  .mac-window-dot.yellow {
    cursor: pointer;
    background: linear-gradient(180deg, #ffd666, #f5a524);
  }

  .mac-window-dot.green {
    cursor: pointer;
    background: linear-gradient(180deg, #7ee787, #22c55e);
  }

  .mac-window-title {
    flex: 1;
    text-align: center;
    font-size: 0.9rem;
    font-weight: 800;
    color: var(--ios-text) !important;
  }

  .mac-window-pill {
    flex: 0 0 auto;
    padding: 0.2rem 0.58rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    color: #1359b8 !important;
    background: rgba(10, 132, 255, 0.1);
    border: 1px solid rgba(10, 132, 255, 0.14);
  }

  .mac-window-body {
    padding: 1.05rem 1.08rem 1.1rem 1.08rem;
    transform-origin: 50% 100%;
    transition: padding 0.38s ease, transform 0.42s ease, filter 0.42s ease, opacity 0.3s ease;
  }

  .mac-window-copy {
    margin: 0.55rem 0 0 0;
    line-height: 1.7;
    color: var(--ios-muted) !important;
  }

  .mac-window-dock {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    width: fit-content;
    max-width: 100%;
    max-height: 0;
    margin: 0 auto;
    padding: 0;
    border-radius: 1.2rem;
    opacity: 0;
    pointer-events: none;
    cursor: pointer;
    transform: translateY(-0.45rem) scale(0.92);
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(226, 232, 240, 0.94);
    box-shadow: var(--ios-shadow-soft);
    backdrop-filter: blur(14px) saturate(160%);
    -webkit-backdrop-filter: blur(14px) saturate(160%);
    transition:
      opacity 0.28s ease 0.38s,
      transform 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.32s,
      max-height 0.28s ease 0.34s,
      padding 0.28s ease 0.34s,
      margin 0.28s ease 0.34s;
  }

  .mac-window-dock-icon {
    width: 1.86rem;
    height: 1.86rem;
    border-radius: 0.62rem;
    background:
      radial-gradient(circle at 70% 20%, rgba(255, 255, 255, 0.76), transparent 34%),
      linear-gradient(135deg, #64d2ff, #0a84ff);
    box-shadow: 0 8px 18px rgba(10, 132, 255, 0.22);
  }

  .mac-window-dock-text {
    font-size: 0.84rem;
    font-weight: 800;
    color: var(--ios-text) !important;
  }

  .mac-window-closed {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    width: fit-content;
    max-width: 100%;
    max-height: 0;
    margin: 0 auto;
    padding: 0;
    border-radius: 1.2rem;
    opacity: 0;
    pointer-events: none;
    cursor: pointer;
    transform: translateY(-0.35rem) scale(0.94);
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(248, 113, 113, 0.18);
    box-shadow: var(--ios-shadow-soft);
    backdrop-filter: blur(14px) saturate(160%);
    -webkit-backdrop-filter: blur(14px) saturate(160%);
    transition:
      opacity 0.24s ease 0.12s,
      transform 0.34s cubic-bezier(0.22, 1, 0.36, 1) 0.08s,
      max-height 0.24s ease 0.08s,
      padding 0.24s ease 0.08s,
      margin 0.24s ease 0.08s;
  }

  .mac-window-closed-icon {
    width: 1.74rem;
    height: 1.74rem;
    border-radius: 0.58rem;
    background:
      radial-gradient(circle at 70% 20%, rgba(255, 255, 255, 0.78), transparent 34%),
      linear-gradient(135deg, #fb7185, #ef4444);
    box-shadow: 0 8px 18px rgba(239, 68, 68, 0.2);
  }

  .mac-window-closed-text {
    font-size: 0.84rem;
    font-weight: 800;
    color: var(--ios-text) !important;
  }

  .mac-window-min-toggle:checked ~ .mac-window-stage,
  .mac-window-close-toggle:checked ~ .mac-window-stage {
    max-height: 0;
    margin-bottom: 0;
  }

  .mac-window-min-toggle:checked ~ .mac-window-stage .mac-window-shell {
    transform: translateY(2.35rem) scale(0.18, 0.08);
    clip-path: inset(44% 46% 44% 46% round 999px);
    opacity: 0;
    filter: blur(7px) saturate(114%);
    box-shadow: 0 0 0 rgba(0, 0, 0, 0);
  }

  .mac-window-close-toggle:checked ~ .mac-window-stage .mac-window-shell {
    transform: translateY(-0.65rem) scale(0.94, 0.92);
    clip-path: inset(0 0 100% 0 round 1.45rem);
    opacity: 0;
    filter: blur(1.4px) saturate(92%);
    box-shadow: 0 0 0 rgba(0, 0, 0, 0);
    animation: mac-window-close-fold 0.42s cubic-bezier(0.4, 0, 0.2, 1) both;
  }

  .mac-window-zoom-toggle:checked ~ .mac-window-stage {
    max-height: 56rem;
  }

  .mac-window-zoom-toggle:checked ~ .mac-window-stage .mac-window-shell {
    transform: translateY(-0.05rem) scale(1.018);
    box-shadow: 0 26px 56px rgba(10, 132, 255, 0.12);
    animation: mac-window-zoom-pop 0.44s cubic-bezier(0.22, 1, 0.36, 1);
  }

  .mac-window-zoom-toggle:checked ~ .mac-window-stage .mac-window-titlebar {
    background:
      radial-gradient(circle at 96% 0%, rgba(126, 231, 135, 0.16), transparent 28%),
      linear-gradient(180deg, rgba(253, 255, 253, 0.98), rgba(243, 249, 244, 0.96));
  }

  .mac-window-zoom-toggle:checked ~ .mac-window-stage .mac-window-body {
    padding: 1.2rem 1.25rem 1.28rem 1.25rem;
  }

  .mac-window-min-toggle:checked ~ .mac-window-dock {
    max-height: 3.6rem;
    margin: 0.72rem auto 1rem auto;
    padding: 0.55rem 0.75rem;
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0) scale(1);
  }

  .mac-window-min-toggle:checked ~ .mac-window-closed {
    max-height: 0;
    margin: 0 auto;
    padding: 0;
    opacity: 0;
    pointer-events: none;
  }

  .mac-window-close-toggle:checked ~ .mac-window-dock {
    max-height: 0;
    margin: 0 auto;
    padding: 0;
    opacity: 0;
    pointer-events: none;
    transform: translateY(-0.45rem) scale(0.92);
  }

  .mac-window-close-toggle:checked ~ .mac-window-closed {
    max-height: 3.4rem;
    margin: 0.72rem auto 1rem auto;
    padding: 0.52rem 0.78rem;
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0) scale(1);
  }

  .mac-window-min-toggle:checked ~ .mac-window-stage .mac-window-shell {
    animation: mac-window-genie-minimize 0.86s cubic-bezier(0.22, 1, 0.36, 1) both;
  }

  .mac-window-min-toggle:checked ~ .mac-window-stage .mac-window-body {
    animation: mac-window-body-genie 0.82s cubic-bezier(0.22, 1, 0.36, 1) both;
  }

  .mac-window-min-toggle:checked ~ .mac-window-stage .mac-window-titlebar {
    animation: mac-window-titlebar-genie 0.82s cubic-bezier(0.22, 1, 0.36, 1) both;
  }

  @keyframes mac-window-genie-minimize {
    0% {
      transform: translateY(0) scale(1, 1) rotateX(0deg);
      clip-path: inset(0 0 0 0 round 1.45rem);
      opacity: 1;
      filter: blur(0) saturate(100%);
    }
    18% {
      transform: translateY(0.1rem) scale(0.985, 0.985) rotateX(0.4deg);
      clip-path: polygon(0% 0%, 100% 0%, 100% 78%, 90% 86%, 73% 92%, 27% 92%, 10% 86%, 0% 78%);
      opacity: 1;
      filter: blur(0.1px) saturate(101%);
    }
    42% {
      transform: translateY(0.88rem) scale(0.9, 0.84) rotateX(1.5deg);
      clip-path: polygon(2% 2%, 98% 2%, 92% 24%, 82% 58%, 68% 86%, 32% 86%, 18% 58%, 8% 24%);
      opacity: 0.98;
      filter: blur(0.55px) saturate(106%);
    }
    62% {
      transform: translateY(1.58rem) scale(0.66, 0.47) rotateX(3.2deg);
      clip-path: polygon(14% 10%, 86% 10%, 78% 22%, 70% 48%, 60% 84%, 40% 84%, 30% 48%, 22% 22%);
      opacity: 0.9;
      filter: blur(1.2px) saturate(109%);
    }
    78% {
      transform: translateY(2.02rem) scale(0.42, 0.23) rotateX(5.6deg);
      clip-path: polygon(28% 18%, 72% 18%, 66% 28%, 60% 54%, 54% 88%, 46% 88%, 40% 54%, 34% 28%);
      opacity: 0.64;
      filter: blur(2.4px) saturate(112%);
    }
    100% {
      transform: translateY(2.35rem) scale(0.18, 0.08) rotateX(8deg);
      clip-path: polygon(48% 42%, 52% 42%, 54% 47%, 53% 56%, 50% 58%, 47% 56%, 46% 47%, 48% 42%);
      opacity: 0;
      filter: blur(7px) saturate(114%);
      box-shadow: 0 0 0 rgba(0, 0, 0, 0);
    }
  }

  @keyframes mac-window-body-genie {
    0% {
      transform: translateY(0) scale(1, 1);
      opacity: 1;
      filter: blur(0);
    }
    48% {
      transform: translateY(0.2rem) scale(0.92, 0.86);
      opacity: 0.74;
      filter: blur(0.45px);
    }
    100% {
      transform: translateY(0.85rem) scale(0.56, 0.38);
      opacity: 0;
      filter: blur(6px);
    }
  }

  @keyframes mac-window-titlebar-genie {
    0% {
      transform: translateY(0) scaleX(1);
      opacity: 1;
    }
    55% {
      transform: translateY(0.08rem) scaleX(0.86);
      opacity: 0.92;
    }
    100% {
      transform: translateY(0.34rem) scaleX(0.46);
      opacity: 0.2;
    }
  }

  @keyframes mac-window-close-fold {
    0% {
      transform: translateY(0) scale(1, 1);
      clip-path: inset(0 0 0 0 round 1.45rem);
      opacity: 1;
    }
    100% {
      transform: translateY(-0.65rem) scale(0.94, 0.92);
      clip-path: inset(0 0 100% 0 round 1.45rem);
      opacity: 0;
    }
  }

  @keyframes mac-window-zoom-pop {
    0% {
      transform: translateY(0) scale(1);
    }
    58% {
      transform: translateY(-0.08rem) scale(1.028);
    }
    100% {
      transform: translateY(-0.05rem) scale(1.018);
    }
  }

  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.85rem;
    margin: 0.45rem 0 1rem 0;
  }

  .stat-card {
    position: relative;
    padding: 1rem 1.05rem;
    border-radius: 1.35rem;
    border: 1px solid var(--ios-border);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.94));
    box-shadow: var(--ios-shadow-soft);
    overflow: hidden;
  }

  .stat-card::after {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(180deg, var(--ios-blue), var(--ios-blue-soft));
  }

  .stat-card.is-success::after {
    background: linear-gradient(180deg, #16a34a, #4ade80);
  }

  .stat-card.is-warn::after {
    background: linear-gradient(180deg, #d97706, #fbbf24);
  }

  .stat-card.is-violet::after {
    background: linear-gradient(180deg, #5e5ce6, #8b5cf6);
  }

  .stat-card.is-slate::after {
    background: linear-gradient(180deg, #475569, #94a3b8);
  }

  .stat-label {
    margin: 0;
    font-size: 0.83rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--ios-muted) !important;
  }

  .stat-value {
    margin: 0.4rem 0 0 0;
    font-size: 1.45rem;
    line-height: 1.15;
    letter-spacing: -0.04em;
    color: var(--ios-text) !important;
  }

  .stat-note {
    margin: 0.55rem 0 0 0;
    font-size: 0.88rem;
    line-height: 1.55;
    color: var(--ios-muted) !important;
  }

  .badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.72rem;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    background: rgba(10, 132, 255, 0.1);
    color: #1259b7 !important;
    border: 1px solid rgba(10, 132, 255, 0.14);
  }

  .status-badge.is-success {
    background: rgba(34, 197, 94, 0.11);
    border-color: rgba(34, 197, 94, 0.18);
    color: #137333 !important;
  }

  .status-badge.is-warn {
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.18);
    color: #b45309 !important;
  }

  .status-badge.is-violet {
    background: rgba(94, 92, 230, 0.12);
    border-color: rgba(94, 92, 230, 0.18);
    color: #4f46e5 !important;
  }

  .status-badge.is-slate {
    background: rgba(71, 85, 105, 0.11);
    border-color: rgba(71, 85, 105, 0.16);
    color: #334155 !important;
  }

  .kv-list {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }

  .kv-row {
    display: flex;
    justify-content: space-between;
    gap: 0.95rem;
    padding: 0.6rem 0.78rem;
    border-radius: 1rem;
    background: rgba(248, 250, 252, 0.9);
    border: 1px solid rgba(226, 232, 240, 0.92);
  }

  .kv-label {
    min-width: 4.5rem;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--ios-muted) !important;
  }

  .kv-value {
    flex: 1;
    text-align: right;
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--ios-text) !important;
    word-break: break-word;
  }

  .empty-state {
    padding: 1.35rem 1.2rem;
    border-radius: 1.45rem;
    border: 1px dashed rgba(148, 163, 184, 0.4);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 248, 252, 0.94));
    box-shadow: var(--ios-shadow-soft);
  }

  .empty-state h3 {
    margin: 0;
    font-size: 1.08rem;
  }

  .empty-state p {
    margin: 0.55rem 0 0 0;
    line-height: 1.7;
    color: var(--ios-muted) !important;
  }

  .history-stack {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .history-card {
    padding: 1rem 1.05rem;
    border-radius: 1.25rem;
    border: 1px solid var(--ios-border);
    background: var(--ios-panel-strong);
    box-shadow: var(--ios-shadow-soft);
  }

  .history-card strong {
    display: block;
    margin-bottom: 0.3rem;
  }

  .history-card .muted-note {
    display: inline-block;
    margin-top: 0.35rem;
  }

  .feature-list {
    margin: 0;
    padding-left: 1.35rem;
  }

  .feature-list li {
    margin: 0 0 0.72rem 0;
    line-height: 1.75;
    color: var(--ios-text) !important;
  }

  .feature-list li:last-child {
    margin-bottom: 0;
  }

  .inline-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.62rem;
    border-radius: 0.7rem;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    vertical-align: baseline;
    color: #1359b8 !important;
    background: linear-gradient(180deg, #eef6ff, #dfeeff);
    border: 1px solid rgba(10, 132, 255, 0.16);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72), 0 4px 10px rgba(10, 132, 255, 0.08);
  }

  .inline-chip *,
  .inline-chip span {
    color: #1359b8 !important;
    -webkit-text-fill-color: #1359b8 !important;
  }

  .command-block {
    margin: 0.25rem 0 0.25rem 0;
    padding: 1rem 1.25rem;
    border-radius: 1.45rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.98rem;
    line-height: 1.65;
    white-space: pre-wrap;
    overflow-x: auto;
    color: #1359b8 !important;
    background: linear-gradient(180deg, #f6fbff, #e7f1ff);
    border: 1px solid rgba(10, 132, 255, 0.14);
    box-shadow: var(--ios-shadow-soft);
  }

  .command-block * {
    color: #1359b8 !important;
    -webkit-text-fill-color: #1359b8 !important;
  }

  .input-section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.95rem 0 0.55rem 0;
  }

  .input-section-title h3 {
    margin: 0;
    font-size: 1.02rem;
    letter-spacing: -0.025em;
  }

  .input-section-title span {
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #1259b7 !important;
    background: rgba(10, 132, 255, 0.1);
    border: 1px solid rgba(10, 132, 255, 0.14);
  }

  .camera-studio,
  .upload-studio {
    position: relative;
    overflow: hidden;
    margin: 0.2rem 0 0.9rem 0;
    padding: 1.15rem 1.15rem 1.05rem 1.15rem;
    border-radius: 1.65rem;
    background:
      radial-gradient(circle at 100% 0%, rgba(100, 210, 255, 0.18), transparent 30%),
      radial-gradient(circle at 0% 100%, rgba(94, 92, 230, 0.08), transparent 24%),
      linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.96));
    border: 1px solid rgba(226, 232, 240, 0.95);
    box-shadow: var(--ios-shadow);
  }

  .studio-kicker {
    margin: 0;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ios-blue) !important;
  }

  .studio-title {
    margin: 0.42rem 0 0 0;
    font-size: 1.2rem;
    line-height: 1.2;
    letter-spacing: -0.04em;
    color: var(--ios-text) !important;
  }

  .studio-copy {
    margin: 0.58rem 0 0 0;
    max-width: 40rem;
    line-height: 1.68;
    color: var(--ios-muted) !important;
  }

  .studio-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.85rem;
  }

  .studio-pill {
    padding: 0.36rem 0.72rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #334155 !important;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(226, 232, 240, 0.92);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
  }

  .camera-guide {
    position: relative;
    margin: 0.2rem 0 0.75rem 0;
    padding: 1rem 1.05rem;
    border-radius: 1.45rem;
    background:
      radial-gradient(circle at 88% 0%, rgba(100, 210, 255, 0.18), transparent 30%),
      linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 250, 255, 0.96));
    border: 1px solid rgba(226, 232, 240, 0.95);
    box-shadow: var(--ios-shadow-soft);
  }

  .camera-guide h4 {
    margin: 0;
    font-size: 1rem;
    letter-spacing: -0.025em;
    color: var(--ios-text) !important;
  }

  .camera-guide p {
    margin: 0.42rem 0 0 0;
    line-height: 1.65;
    color: var(--ios-muted) !important;
  }

  .camera-steps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.45rem;
    margin-top: 0.8rem;
  }

  .camera-step {
    padding: 0.55rem 0.62rem;
    border-radius: 1rem;
    font-size: 0.83rem;
    font-weight: 700;
    text-align: center;
    color: #334155 !important;
    background: rgba(248, 250, 252, 0.9);
    border: 1px solid rgba(226, 232, 240, 0.92);
  }

  .capture-surface-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0 0 0.55rem 0;
    padding: 0.6rem 0.78rem;
    border-radius: 1rem;
    background: rgba(248, 250, 252, 0.88);
    border: 1px solid rgba(226, 232, 240, 0.92);
  }

  .capture-surface-label strong {
    font-size: 0.92rem;
    color: var(--ios-text) !important;
  }

  .capture-surface-label span {
    font-size: 0.82rem;
    color: var(--ios-muted) !important;
  }

  .camera-side-card {
    padding: 1rem 1.02rem;
    border-radius: 1.45rem;
    background:
      linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.95));
    border: 1px solid rgba(226, 232, 240, 0.95);
    box-shadow: var(--ios-shadow-soft);
    margin-bottom: 0.75rem;
  }

  .camera-side-eyebrow {
    margin: 0;
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ios-blue) !important;
  }

  .camera-side-card h4 {
    margin: 0.45rem 0 0 0;
    font-size: 1.03rem;
    letter-spacing: -0.03em;
    color: var(--ios-text) !important;
  }

  .camera-side-card p {
    margin: 0.48rem 0 0 0;
    line-height: 1.65;
    color: var(--ios-muted) !important;
  }

  .camera-status-card {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin: 0.7rem 0 0.7rem 0;
    padding: 0.9rem 0.95rem;
    border-radius: 1.2rem;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.92), rgba(241, 245, 249, 0.88));
    border: 1px solid rgba(226, 232, 240, 0.92);
  }

  .camera-status-card strong {
    font-size: 0.94rem;
    color: var(--ios-text) !important;
  }

  .camera-status-card span {
    font-size: 0.84rem;
    line-height: 1.55;
    color: var(--ios-muted) !important;
  }

  .camera-status-card.is-ready {
    background:
      linear-gradient(180deg, rgba(236, 253, 245, 0.98), rgba(220, 252, 231, 0.92));
    border-color: rgba(34, 197, 94, 0.2);
  }

  .action-hub {
    display: flex;
    flex-direction: column;
    gap: 0.22rem;
    margin: 0.75rem 0 0.7rem 0;
    padding: 0.9rem 0.98rem;
    border-radius: 1.25rem;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(246, 250, 255, 0.94));
    border: 1px solid rgba(226, 232, 240, 0.92);
    box-shadow: var(--ios-shadow-soft);
  }

  .action-hub strong {
    font-size: 0.95rem;
    color: var(--ios-text) !important;
  }

  .action-hub span {
    font-size: 0.84rem;
    line-height: 1.6;
    color: var(--ios-muted) !important;
  }

  [data-testid="stCameraInput"] {
    padding: 0.95rem;
    border-radius: 1.65rem;
    background:
      linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 249, 255, 0.95));
    border: 1px solid rgba(226, 232, 240, 0.96);
    box-shadow: var(--ios-shadow);
  }

  [data-testid="stCameraInput"] label {
    margin-bottom: 0.45rem;
    font-weight: 800;
    letter-spacing: -0.015em;
  }

  [data-testid="stCameraInput"] video,
  [data-testid="stCameraInput"] img,
  [data-testid="stCameraInput"] canvas {
    border-radius: 1.25rem !important;
    background: linear-gradient(135deg, #eef6ff, #f8fbff) !important;
    border: 1px solid rgba(203, 213, 225, 0.7);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78), 0 12px 26px rgba(15, 23, 42, 0.08);
  }

  [data-testid="stCameraInput"] button {
    border-radius: 999px !important;
    min-height: 2.7rem;
    font-weight: 800 !important;
    background: linear-gradient(180deg, #ffffff, #eef6ff) !important;
    border: 1px solid rgba(10, 132, 255, 0.18) !important;
    color: #1359b8 !important;
    box-shadow: 0 8px 18px rgba(10, 132, 255, 0.12) !important;
  }

  [data-testid="stCameraInput"] button * {
    color: #1359b8 !important;
    -webkit-text-fill-color: #1359b8 !important;
  }

  div[data-testid="stMetric"] {
    padding: 0.95rem 1rem;
    border-radius: 1.2rem;
    border: 1px solid var(--ios-border);
    background: var(--ios-panel-strong);
    box-shadow: var(--ios-shadow);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }

  div[data-testid="stMetricLabel"] p {
    color: var(--ios-muted) !important;
    font-weight: 600;
  }

  div[data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] * {
    color: var(--ios-text) !important;
  }

  div[data-testid="stMetricDelta"],
  div[data-testid="stMetricDelta"] * {
    color: var(--ios-blue) !important;
  }

  .stButton > button,
  .stFormSubmitButton > button {
    min-height: 2.9rem;
    border: none;
    border-radius: 999px;
    color: white;
    font-weight: 700;
    background: linear-gradient(180deg, #3a9bff, #0a84ff);
    box-shadow: 0 10px 22px rgba(10, 132, 255, 0.20);
    transition: transform 0.16s ease, filter 0.16s ease, box-shadow 0.16s ease;
  }

  .stButton > button:hover,
  .stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
    box-shadow: 0 14px 26px rgba(10, 132, 255, 0.24);
  }

  .stButton > button *,
  .stFormSubmitButton > button *,
  button[kind] * {
    color: white !important;
  }

  [data-testid="stAlert"],
  [data-testid="stCodeBlock"],
  [data-testid="stFileUploaderDropzone"] {
    border-radius: 1.2rem;
    border: 1px solid var(--ios-border);
    background: var(--ios-panel-strong);
    box-shadow: var(--ios-shadow-soft);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
  }

  [data-testid="stFileUploaderDropzone"] {
    border-style: dashed;
    border-width: 1.5px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(245, 249, 255, 0.92));
    padding-top: 1rem;
    padding-bottom: 1rem;
  }

  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(10, 132, 255, 0.26);
    box-shadow: 0 12px 24px rgba(10, 132, 255, 0.08);
  }

  [data-testid="stCheckbox"] {
    margin: 0.65rem 0 0.55rem 0;
    padding: 0.85rem 0.95rem;
    border-radius: 1.15rem;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(246, 249, 255, 0.94));
    border: 1px solid rgba(226, 232, 240, 0.92);
    box-shadow: var(--ios-shadow-soft);
  }

  [data-testid="stCheckbox"] label {
    font-weight: 700 !important;
  }

  [data-testid="stCheckbox"] p {
    color: var(--ios-text) !important;
  }

  [data-testid="stAlert"] {
    border-width: 1px !important;
    box-shadow: var(--ios-shadow-soft);
  }

  [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    line-height: 1.6;
  }

  div[data-baseweb="select"] > div,
  div[data-baseweb="base-input"] > div {
    border-radius: 1rem;
    border: 1px solid var(--ios-border);
    background: rgba(255, 255, 255, 0.95);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
  }

  div[data-baseweb="select"] *,
  div[data-baseweb="base-input"] * {
    color: var(--ios-text) !important;
  }

  [data-baseweb="popover"] *,
  [role="listbox"] *,
  [role="option"] * {
    color: var(--ios-text) !important;
  }

  div[data-baseweb="select"] input,
  div[data-baseweb="base-input"] input {
    color: var(--ios-text) !important;
    -webkit-text-fill-color: var(--ios-text) !important;
  }

  input,
  textarea,
  select {
    color: var(--ios-text) !important;
    -webkit-text-fill-color: var(--ios-text) !important;
  }

  [data-testid="stImage"] img {
    border-radius: 1.25rem;
    box-shadow: var(--ios-shadow);
  }

  [data-testid="stImage"],
  [data-testid="stPlotlyChart"] {
    padding: 0.88rem;
    border-radius: 1.35rem;
    border: 1px solid rgba(226, 232, 240, 0.95);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 250, 255, 0.95));
    box-shadow: var(--ios-shadow-soft);
  }

  [data-testid="stPlotlyChart"] > div {
    border-radius: 1rem;
    overflow: hidden;
  }

  [data-testid="stTabs"] [role="tablist"] {
    gap: 0.45rem;
    padding: 0.26rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid var(--ios-border);
    width: fit-content;
    box-shadow: var(--ios-shadow-soft);
  }

  [data-testid="stTabs"] [role="tab"] {
    height: auto;
    padding: 0.5rem 0.95rem;
    border-radius: 999px;
    color: var(--ios-muted);
    font-weight: 600;
    transition: background 0.16s ease, color 0.16s ease, box-shadow 0.16s ease;
  }

  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--ios-blue);
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05), inset 0 0 0 1px rgba(10, 132, 255, 0.08);
  }

  [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: transparent !important;
  }

  [data-testid="stTabs"] button[role="tab"]::after {
    display: none !important;
  }

  [data-testid="stTabs"] [role="tab"] * {
    color: inherit !important;
  }

  [data-testid="stAlert"] *,
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stCodeBlock"] * {
    color: var(--ios-text);
  }

  [data-testid="stCodeBlock"] {
    background: linear-gradient(180deg, #202433, var(--ios-code-bg)) !important;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  [data-testid="stCodeBlock"] pre,
  [data-testid="stCodeBlock"] pre code {
    background: var(--ios-code-bg) !important;
    color: var(--ios-code-text) !important;
  }

  [data-testid="stCodeBlock"] pre *,
  [data-testid="stCodeBlock"] pre code *,
  pre code,
  pre code * {
    color: var(--ios-code-text) !important;
  }

  pre {
    background: var(--ios-code-bg) !important;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 1.2rem !important;
  }

  @media (max-width: 900px) {
    .main .block-container {
      padding-top: 1rem;
      padding-left: 1rem;
      padding-right: 1rem;
    }

    .hero-card {
      padding: 1.25rem 1.1rem;
      border-radius: 1.45rem;
    }

    .camera-steps {
      grid-template-columns: 1fr;
    }

    .capture-surface-label {
      flex-direction: column;
      align-items: flex-start;
    }
  }
</style>
"""


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_stage3_metrics_path() -> Path:
    checkpoint_dir = ROOT / DEFAULT_STAGE3_CHECKPOINT.parent
    rechecked = checkpoint_dir / "test_metrics_rechecked.json"
    if rechecked.exists():
        return rechecked
    return checkpoint_dir / "test_metrics.json"


@lru_cache(maxsize=1)
def build_stage3_model_info() -> Dict[str, Any]:
    checkpoint_path = (ROOT / DEFAULT_STAGE3_CHECKPOINT).resolve()
    data_root = (ROOT / DEFAULT_STAGE3_DATA_ROOT).resolve()
    checkpoint_dir = checkpoint_path.parent
    metrics_path = resolve_stage3_metrics_path()
    metrics = load_json_file(metrics_path)
    model_summary = load_json_file(checkpoint_dir / "model_summary.json")
    verification_summary = load_json_file(ROOT / "runs/stage3/verification_summary.json")
    stage2_summary = verification_summary.get("stage2", {})
    summary_config = model_summary.get("config", {})
    shadow_tta_results = load_json_file(ROOT / SHADOW_TTA_SWEEP_PATH)
    best_shadow_tta = shadow_tta_results[0] if isinstance(shadow_tta_results, list) and shadow_tta_results else {}

    class_names = stage2_summary.get("class_names")
    if not class_names:
        split_report = load_json_file(data_root / "split_report.json")
        class_names = sorted(split_report.get("train", {}).keys())
    device = stage2_summary.get("device") or summary_config.get("device") or "未记录"

    accuracy = metrics.get("accuracy")
    if accuracy is None:
        accuracy = metrics.get("test_accuracy")

    loss = metrics.get("loss")
    if loss is None:
        loss = metrics.get("test_loss")

    return {
        "checkpoint_name": checkpoint_path.parent.name,
        "checkpoint_path": str(checkpoint_path),
        "data_root": str(data_root),
        "device": device,
        "class_names": list(class_names or []),
        "class_count": len(class_names or []),
        "image_size": summary_config.get("image_size", "未记录"),
        "auto_crop": summary_config.get("auto_crop", False),
        "accuracy": accuracy,
        "accuracy_pct": None if accuracy is None else round(float(accuracy) * 100, 2),
        "loss": None if loss is None else round(float(loss), 4),
        "best_epoch": metrics.get("best_epoch"),
        "shadow_tta_available": bool(best_shadow_tta),
        "shadow_tta_accuracy": best_shadow_tta.get("accuracy", {}).get("avg_50_50"),
        "shadow_tta_accuracy_pct": None
        if best_shadow_tta.get("accuracy", {}).get("avg_50_50") is None
        else round(float(best_shadow_tta["accuracy"]["avg_50_50"]) * 100, 2),
        "shadow_tta_params": best_shadow_tta.get("params", {}),
    }


def _escape_text(value: Any) -> str:
    return escape("" if value is None else str(value))


def build_badge_row(items: Sequence[Dict[str, Any]]) -> str:
    badges: List[str] = []
    for item in items:
        label = _escape_text(item.get("label"))
        if not label:
            continue
        tone = _escape_text(item.get("tone", "blue"))
        badges.append(f"<span class='status-badge is-{tone}'>{label}</span>")
    if not badges:
        return ""
    return "<div class='badge-row'>" + "".join(badges) + "</div>"


def build_stat_grid(items: Sequence[Dict[str, Any]]) -> str:
    cards: List[str] = []
    for item in items:
        label = _escape_text(item.get("label"))
        value = _escape_text(item.get("value"))
        note = item.get("note")
        note_html = f"<p class='stat-note'>{_escape_text(note)}</p>" if note else ""
        tone = _escape_text(item.get("tone", "blue"))
        cards.append(
            (
                f"<div class='stat-card is-{tone}'>"
                f"<p class='stat-label'>{label}</p>"
                f"<h3 class='stat-value'>{value}</h3>"
                f"{note_html}"
                "</div>"
            )
        )
    if not cards:
        return ""
    return "<div class='stat-grid'>" + "".join(cards) + "</div>"


def build_key_value_list(rows: Sequence[Sequence[Any]]) -> str:
    entries: List[str] = []
    for row in rows:
        if len(row) < 2:
            continue
        label = _escape_text(row[0])
        value = _escape_text(row[1])
        entries.append(
            (
                "<div class='kv-row'>"
                f"<span class='kv-label'>{label}</span>"
                f"<span class='kv-value'>{value}</span>"
                "</div>"
            )
        )
    if not entries:
        return ""
    return "<div class='kv-list'>" + "".join(entries) + "</div>"


def build_panel_card(
    title: str,
    body_html: str,
    eyebrow: str = "",
    footer_html: str = "",
    tone: str = "blue",
) -> str:
    eyebrow_html = f"<p class='panel-eyebrow'>{_escape_text(eyebrow)}</p>" if eyebrow else ""
    footer = f"<div class='panel-footer'>{footer_html}</div>" if footer_html else ""
    return (
        f"<div class='panel-card is-{_escape_text(tone)}'>"
        f"{eyebrow_html}"
        f"<h3 class='panel-title'>{_escape_text(title)}</h3>"
        f"<div class='panel-body'>{body_html}</div>"
        f"{footer}"
        "</div>"
    )


def build_mac_window_card(
    window_id: str,
    title: str,
    body_html: str,
    eyebrow: str = "Actual Window",
    subtitle: str = "红点关闭，黄点最小化，绿点放大或还原；下方卡片可恢复窗口。",
    footer_html: str = "",
) -> str:
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in window_id).strip("-_") or "mac-window"
    close_id = f"{safe_id}-close"
    min_id = f"{safe_id}-min"
    zoom_id = f"{safe_id}-zoom"
    footer = f"<div class='panel-footer'>{footer_html}</div>" if footer_html else ""
    subtitle_html = f"<p class='mac-window-copy'>{_escape_text(subtitle)}</p>" if subtitle else ""
    return (
        "<div class='mac-window'>"
        f"<input id='{close_id}' class='mac-window-toggle mac-window-close-toggle' type='checkbox' />"
        f"<input id='{min_id}' class='mac-window-toggle mac-window-min-toggle' type='checkbox' />"
        f"<input id='{zoom_id}' class='mac-window-toggle mac-window-zoom-toggle' type='checkbox' />"
        "<div class='mac-window-stage'>"
        "<div class='mac-window-shell'>"
        "<div class='mac-window-titlebar'>"
        "<span class='mac-window-traffic'>"
        f"<label class='mac-window-dot red' for='{close_id}' title='关闭'></label>"
        f"<label class='mac-window-dot yellow' for='{min_id}' title='最小化'></label>"
        f"<label class='mac-window-dot green' for='{zoom_id}' title='放大或还原'></label>"
        "</span>"
        f"<span class='mac-window-title'>{_escape_text(title)}</span>"
        f"<span class='mac-window-pill'>{_escape_text(eyebrow)}</span>"
        "</div>"
        "<div class='mac-window-body'>"
        f"{body_html}"
        f"{subtitle_html}"
        f"{footer}"
        "</div>"
        "</div>"
        "</div>"
        f"<label class='mac-window-dock' for='{min_id}' title='恢复 { _escape_text(title) }'>"
        "<span class='mac-window-dock-icon'></span>"
        f"<span class='mac-window-dock-text'>{_escape_text(title)} 已最小化，点击恢复</span>"
        "</label>"
        f"<label class='mac-window-closed' for='{close_id}' title='重新打开 { _escape_text(title) }'>"
        "<span class='mac-window-closed-icon'></span>"
        f"<span class='mac-window-closed-text'>{_escape_text(title)} 已关闭，点击重新打开</span>"
        "</label>"
        "</div>"
    )


def build_section_header(title: str, body: str = "", eyebrow: str = "") -> str:
    eyebrow_html = f"<p class='section-eyebrow'>{_escape_text(eyebrow)}</p>" if eyebrow else ""
    body_html = f"<p class='section-copy'>{_escape_text(body)}</p>" if body else ""
    return (
        "<div class='section-header'>"
        f"{eyebrow_html}"
        f"<h3 class='section-title'>{_escape_text(title)}</h3>"
        f"{body_html}"
        "</div>"
    )


def build_empty_state(title: str, body: str) -> str:
    return (
        "<div class='empty-state'>"
        f"<h3>{_escape_text(title)}</h3>"
        f"<p>{_escape_text(body)}</p>"
        "</div>"
    )


def image_bytes_to_pil(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("上传文件为空。")
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise ValueError("无法读取上传图片。") from exc


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def serialize_prediction(prediction: Any) -> Dict[str, Any]:
    return {
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "confidence_pct": round(prediction.confidence * 100, 2),
        "image_size": prediction.image_size,
        "top_probabilities": [
            {
                "class_name": item.class_name,
                "probability": item.probability,
                "probability_pct": round(item.probability * 100, 2),
            }
            for item in prediction.top_probabilities
        ],
    }


def classify_uploaded_bytes(content: bytes, filename: str, use_shadow_tta: bool = False) -> Dict[str, Any]:
    return classify_uploaded_batch([(filename, content)], use_shadow_tta=use_shadow_tta)[0]


def classify_uploaded_batch(items: Sequence[tuple[str, bytes]], use_shadow_tta: bool = False) -> List[Dict[str, Any]]:
    from model.inference import load_default_classifier, predict_pil_images

    if not items:
        return []

    classifier = load_default_classifier()
    filenames: List[str] = []
    images: List[Image.Image] = []
    for filename, content in items:
        filenames.append(filename)
        images.append(image_bytes_to_pil(content))

    predictions = predict_pil_images(images, classifier=classifier, top_k=5, use_shadow_tta=use_shadow_tta)
    results: List[Dict[str, Any]] = []
    for filename, image, prediction in zip(filenames, images, predictions):
        item = serialize_prediction(prediction)
        item.update(
            {
                "filename": filename,
                "image_bytes": pil_to_png_bytes(image),
                "inference_mode": "抗阴影双路融合" if use_shadow_tta else "标准单路推理",
            }
        )
        results.append(item)
    return results


@lru_cache(maxsize=1)
def build_example_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    verification_summary = load_json_file(ROOT / "runs/stage3/verification_summary.json")
    sample_predictions = verification_summary.get("stage3", {}).get("sample_predictions", [])

    for item in sample_predictions:
        sample_path = Path(item["path"])
        if not sample_path.exists():
            continue

        with Image.open(sample_path) as image:
            preview_bytes = pil_to_png_bytes(image.convert("RGB"))

        scores = [
            {
                "class_name": score["class_name"],
                "probability": score["probability"],
                "probability_pct": round(float(score["probability"]) * 100, 2),
            }
            for score in item.get("top_probabilities", [])
        ]
        cards.append(
            {
                "filename": sample_path.name,
                "expected_class": item.get("expected_class", "未记录"),
                "predicted_class": item.get("predicted_class", "未记录"),
                "is_correct": item.get("predicted_class", "未记录") == item.get("expected_class", "未记录"),
                "confidence": float(item.get("confidence", 0.0)),
                "confidence_pct": round(float(item.get("confidence", 0.0)) * 100, 2),
                "top_probabilities": scores,
                "image_bytes": preview_bytes,
                "image_path": str(sample_path),
            }
        )

    if cards:
        return cards

    from model.inference import load_default_classifier, predict_image_file

    classifier = load_default_classifier()
    test_root = ROOT / DEFAULT_STAGE3_DATA_ROOT / "test"
    for class_name in classifier.class_names:
        class_dir = test_root / class_name
        if not class_dir.exists():
            continue
        sample_path = next((path for path in sorted(class_dir.iterdir()) if path.is_file()), None)
        if sample_path is None:
            continue
        with Image.open(sample_path) as image:
            preview_bytes = pil_to_png_bytes(image.convert("RGB"))
        prediction = predict_image_file(sample_path, classifier=classifier, top_k=3)
        card = serialize_prediction(prediction)
        card.update(
            {
                "filename": sample_path.name,
                "expected_class": class_name,
                "is_correct": card["predicted_class"] == class_name,
                "image_bytes": preview_bytes,
                "image_path": str(sample_path),
            }
        )
        cards.append(card)
    return cards


def build_about_context() -> Dict[str, Any]:
    quality_report = load_json_file(ROOT / "data/quality_report.json")
    stage1_split_report = load_json_file(ROOT / "data/splits/split_report.json")
    stage2_split_report = load_json_file(ROOT / DEFAULT_STAGE3_DATA_ROOT / "split_report.json")
    model_info = build_stage3_model_info()

    stage1_split_totals = {
        split_name: sum(stage1_split_report.get(split_name, {}).values())
        for split_name in ("train", "val", "test")
    }
    stage2_split_totals = {
        split_name: sum(stage2_split_report.get(split_name, {}).values())
        for split_name in ("train", "val", "test")
    }

    return {
        "stage1": {
            "class_count": quality_report.get("class_count", 0),
            "total_images": quality_report.get("dimension_summary", {}).get("total_images", 0),
            "per_class_counts": quality_report.get("per_class_counts", {}),
            "format_distribution": quality_report.get("format_distribution", {}),
            "split_totals": stage1_split_totals,
        },
        "stage2": {
            **model_info,
            "split_totals": stage2_split_totals,
        },
        "stage3": {
            "page_count": 4,
            "features": [
                {
                    "title": "Streamlit 多页面应用",
                    "body": "使用 Streamlit 主入口加 pages 目录构建多页面应用，支持首页、分类识别、示例体验和关于页面。",
                },
                {
                    "title": "深度学习模型推理链路",
                    "body": "复用统一的 checkpoint 加载和推理模块，完成图片预处理、前向推理、Softmax 概率解析和结果展示。",
                },
                {
                    "title": "Plotly 概率可视化",
                    "body": "分类页面和示例页面都提供 Plotly 概率分布条形图，并配合双栏布局提升展示效果。",
                },
                {
                    "title": "批量识别与拍照入口",
                    "body": "分类页面支持批量上传、移动端拍照识别和历史记录展示，更适合课堂演示和实际试用。",
                },
            ],
        },
    }


def build_probability_figure(scores: Sequence[Dict[str, Any]], title: str = "类别概率分布"):
    import plotly.graph_objects as go

    labels = [item["class_name"] for item in scores]
    values = [item["probability_pct"] for item in scores]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0.0, "#bfdbfe"], [0.5, "#5ac8fa"], [1.0, "#0a84ff"]],
                line=dict(color="#0f172a", width=0),
            ),
            text=[f"{value:.2f}%" for value in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        font=dict(color="#0f172a"),
        height=max(260, 80 + len(scores) * 56),
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(color="#0f172a", size=20),
        xaxis=dict(
            title="概率 (%)",
            title_font=dict(color="#0f172a"),
            tickfont=dict(color="#0f172a"),
            range=[0, 100],
            gridcolor="rgba(148, 163, 184, 0.25)",
            zerolinecolor="rgba(148, 163, 184, 0.25)",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#0f172a"),
        ),
    )
    return figure


def run_geometry_detection(image: np.ndarray, mode: str) -> Dict[str, Any]:
    if mode == "generated_multi":
        from detectors.lego_generated_detector import detect_generated_objects, draw_generated_result

        objects = detect_generated_objects(image)
        annotated = draw_generated_result(image, objects)
        return {
            "annotated_image": annotated,
            "summary": [f"检测到主体数：{len(objects)}"],
            "objects": [
                {
                    "name": f"对象 {item.index}",
                    "shape": f"{item.dims[0]} x {item.dims[1]} x {item.height}",
                    "confidence": f"{item.confidence:.2f}",
                }
                for item in objects
            ],
        }

    if mode == "multi_stack":
        from detectors.lego_multi_stack_detector import detect_multi_stack_objects, draw_result as draw_multi_stack_result

        objects = detect_multi_stack_objects(image)
        annotated = draw_multi_stack_result(image, objects)
        summary = [f"检测到主体数：{len(objects)}"]
        if not objects:
            summary.append("没有检测到可分离的主体。")
        return {
            "annotated_image": annotated,
            "summary": summary,
            "objects": [
                {
                    "name": f"对象 {item.index}",
                    "shape": f"1 x 1 x {item.layers}",
                    "confidence": f"{item.confidence:.2f}",
                }
                for item in objects
            ],
        }

    from detectors.lego_size_detector import detect_lego_size, draw_result as draw_single_result

    result = detect_lego_size(image)
    annotated = draw_single_result(
        image,
        result,
        height_only=mode == "height_only",
        size_only=mode == "size_only",
    )

    summary: List[str] = []
    if mode != "height_only":
        if result.dims != (0, 0):
            summary.append(f"尺寸：{result.dims[0]} x {result.dims[1]}")
            summary.append(f"尺寸置信度：{result.size_confidence:.2f}")
        else:
            summary.append("尺寸：未识别")

    if mode != "size_only":
        if result.height > 0:
            summary.append(f"层高：{result.height}")
            summary.append(f"层高置信度：{result.height_confidence:.2f}")
        else:
            summary.append("层高：未识别")

    if result.dims != (0, 0) and result.height > 0 and mode == "single":
        summary.append(f"三维形态：{result.dims[0]} x {result.dims[1]} x {result.height}")

    return {
        "annotated_image": annotated,
        "summary": summary,
        "objects": [],
    }

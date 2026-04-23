import streamlit as st
import streamlit.components.v1 as components

from utils.stage3_streamlit import (
    APP_CSS,
    bgr_to_rgb,
    build_about_context,
    build_badge_row,
    build_empty_state,
    build_key_value_list,
    build_mac_window_card,
    build_panel_card,
    build_stage3_model_info,
    build_stat_grid,
    image_bytes_to_pil,
    pil_to_bgr,
    run_geometry_detection,
)


DETECT_MODE_OPTIONS = {
    "single": "单主体：尺寸 + 层高",
    "size_only": "单主体：只看尺寸",
    "height_only": "单主体：只看层高",
    "multi_stack": "多主体：堆叠层数",
}


st.set_page_config(
    page_title="乐高检测与识别应用",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)

MAC_MINIMIZE_CSS = """
<style>
  .mac-minimize-demo {
    position: relative;
    width: 100%;
    margin-top: 0.9rem;
  }

  .mac-demo-toggle {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  .mac-demo-scene {
    position: relative;
    min-height: 19rem;
    padding-bottom: 4.2rem;
  }

  .mac-demo-window {
    position: absolute;
    top: 0;
    left: 50%;
    width: min(100%, 21rem);
    transform: translateX(-50%) translateY(0) scale(1);
    transform-origin: 50% 100%;
    clip-path: inset(0 0 0 0 round 1.3rem);
    will-change: transform, clip-path, opacity, filter;
    border-radius: 1.3rem;
    overflow: hidden;
    border: 1px solid rgba(214, 220, 228, 0.96);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 249, 255, 0.96));
    box-shadow: 0 24px 50px rgba(15, 23, 42, 0.12);
    transition:
      transform 0.76s cubic-bezier(0.22, 1, 0.36, 1),
      opacity 0.46s ease,
      filter 0.56s ease,
      clip-path 0.76s cubic-bezier(0.22, 1, 0.36, 1),
      box-shadow 0.56s ease;
  }

  .mac-demo-window::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.32), transparent 38%);
    pointer-events: none;
  }

  .mac-demo-toggle:checked + .mac-demo-scene .mac-demo-window {
    transform: translateX(-50%) translateY(12rem) scale(0.16);
    clip-path: inset(46% 47% 46% 47% round 999px);
    opacity: 0.18;
    filter: blur(8px) saturate(112%);
    box-shadow: 0 0 0 rgba(0, 0, 0, 0);
    animation: mac-genie-minimize 0.82s cubic-bezier(0.22, 1, 0.36, 1) forwards;
  }

  @keyframes mac-genie-minimize {
    0% {
      transform: translateX(-50%) translateY(0) scale(1, 1);
      clip-path: inset(0 0 0 0 round 1.3rem);
      opacity: 1;
      filter: blur(0) saturate(100%);
    }
    32% {
      transform: translateX(-50%) translateY(2.8rem) scale(0.96, 0.86);
      clip-path: inset(4% 5% 8% 5% round 1.1rem);
      opacity: 0.98;
      filter: blur(0.5px) saturate(104%);
    }
    62% {
      transform: translateX(-50%) translateY(7.6rem) scale(0.62, 0.28);
      clip-path: inset(26% 30% 34% 30% round 999px);
      opacity: 0.74;
      filter: blur(2px) saturate(108%);
    }
    100% {
      transform: translateX(-50%) translateY(12rem) scale(0.16, 0.08);
      clip-path: inset(46% 47% 46% 47% round 999px);
      opacity: 0.14;
      filter: blur(8px) saturate(112%);
      box-shadow: 0 0 0 rgba(0, 0, 0, 0);
    }
  }

  .mac-demo-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    padding: 0.7rem 0.82rem;
    border-bottom: 1px solid rgba(226, 232, 240, 0.9);
    background: linear-gradient(180deg, rgba(250, 251, 253, 0.98), rgba(244, 246, 250, 0.96));
  }

  .mac-demo-traffic {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
  }

  .mac-dot {
    width: 0.72rem;
    height: 0.72rem;
    border-radius: 999px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.68);
  }

  .mac-dot.red {
    background: linear-gradient(180deg, #ff7b72, #f04438);
  }

  .mac-dot.yellow {
    display: inline-block;
    cursor: pointer;
    background: linear-gradient(180deg, #ffd666, #f5a524);
  }

  .mac-dot.green {
    background: linear-gradient(180deg, #7ee787, #22c55e);
  }

  .mac-demo-title {
    flex: 1;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #475569 !important;
  }

  .mac-demo-pill {
    padding: 0.2rem 0.52rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #1259b7 !important;
    background: rgba(10, 132, 255, 0.1);
    border: 1px solid rgba(10, 132, 255, 0.12);
  }

  .mac-demo-body {
    padding: 0.95rem;
    background:
      radial-gradient(circle at top right, rgba(100, 210, 255, 0.14), transparent 22%),
      linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.96));
  }

  .mac-demo-hero {
    height: 5.9rem;
    border-radius: 1rem;
    background:
      radial-gradient(circle at 75% 24%, rgba(100, 210, 255, 0.28), transparent 24%),
      radial-gradient(circle at 18% 82%, rgba(94, 92, 230, 0.16), transparent 26%),
      linear-gradient(135deg, #f8fbff 0%, #eaf3ff 50%, #f5f8ff 100%);
    border: 1px solid rgba(215, 226, 240, 0.92);
  }

  .mac-demo-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.62rem;
    margin-top: 0.75rem;
  }

  .mac-demo-card {
    padding: 0.72rem 0.6rem;
    border-radius: 0.95rem;
    border: 1px solid rgba(224, 232, 240, 0.92);
    background: rgba(255, 255, 255, 0.92);
  }

  .mac-demo-card strong {
    display: block;
    font-size: 0.75rem;
    font-weight: 800;
    color: #1d1d1f !important;
  }

  .mac-demo-card span {
    display: block;
    margin-top: 0.26rem;
    font-size: 0.72rem;
    line-height: 1.45;
    color: #6e6e73 !important;
  }

  .mac-demo-line-group {
    display: grid;
    gap: 0.45rem;
    margin-top: 0.8rem;
  }

  .mac-demo-line {
    height: 0.58rem;
    border-radius: 999px;
    background: linear-gradient(180deg, rgba(214, 228, 248, 0.9), rgba(235, 242, 252, 0.95));
  }

  .mac-demo-line.short {
    width: 64%;
  }

  .mac-demo-line.mid {
    width: 83%;
  }

  .mac-demo-dock {
    position: absolute;
    left: 50%;
    bottom: 0.2rem;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 0.62rem;
    width: min(100%, 15rem);
    padding: 0.6rem 0.75rem;
    transform: translateX(-50%);
    border-radius: 1.3rem;
    border: 1px solid rgba(221, 228, 236, 0.96);
    background: rgba(255, 255, 255, 0.7);
    box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
  }

  .dock-item {
    position: relative;
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 0.95rem;
    border: 1px solid rgba(210, 220, 232, 0.9);
    background: linear-gradient(180deg, #f7fbff, #dfeeff);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82), 0 8px 18px rgba(15, 23, 42, 0.08);
    transition: transform 0.4s ease, box-shadow 0.4s ease, background 0.4s ease;
  }

  .dock-item::before {
    content: '';
    position: absolute;
    inset: 0.52rem;
    border-radius: 0.68rem;
    background: linear-gradient(180deg, rgba(94, 92, 230, 0.24), rgba(10, 132, 255, 0.16));
  }

  .dock-item.primary {
    cursor: pointer;
    background: linear-gradient(180deg, #fdfefe, #eaf5ff);
  }

  .dock-item.primary::after {
    content: '';
    position: absolute;
    left: 50%;
    bottom: -0.58rem;
    width: 0.34rem;
    height: 0.34rem;
    border-radius: 999px;
    background: rgba(18, 89, 183, 0.22);
    transform: translateX(-50%);
    transition: background 0.35s ease, transform 0.35s ease, opacity 0.35s ease;
    opacity: 0.56;
  }

  .dock-item.secondary::before {
    background: linear-gradient(180deg, rgba(34, 197, 94, 0.24), rgba(16, 185, 129, 0.16));
  }

  .dock-item.tertiary::before {
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.24), rgba(251, 191, 36, 0.16));
  }

  .mac-demo-toggle:checked + .mac-demo-scene .dock-item.primary {
    transform: translateY(-0.38rem) scale(1.08);
    background: linear-gradient(180deg, #ffffff, #d9ebff);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 16px 28px rgba(10, 132, 255, 0.2);
    animation: dock-bounce 0.68s cubic-bezier(0.2, 0.9, 0.2, 1) both;
  }

  .mac-demo-toggle:checked + .mac-demo-scene .dock-item.primary::after {
    background: rgba(10, 132, 255, 0.72);
    transform: translateX(-50%) scale(1.12);
    opacity: 1;
  }

  @keyframes dock-bounce {
    0% {
      transform: translateY(0) scale(1);
    }
    45% {
      transform: translateY(-0.56rem) scale(1.12);
    }
    72% {
      transform: translateY(-0.24rem) scale(1.05);
    }
    100% {
      transform: translateY(-0.38rem) scale(1.08);
    }
  }

  .mac-demo-hint {
    margin: 0.8rem 0 0 0;
    text-align: center;
    font-size: 0.8rem;
    line-height: 1.6;
    color: #6e6e73 !important;
  }

  .mac-demo-toggle:checked + .mac-demo-scene .mac-demo-hint {
    color: #1259b7 !important;
  }
</style>
"""

st.markdown(MAC_MINIMIZE_CSS, unsafe_allow_html=True)


def render_summary_chips(items):
    if not items:
        st.markdown(build_empty_state("当前没有可展示的识别摘要", "请重新上传图片并执行一次检测。"), unsafe_allow_html=True)
        return
    cards = []
    for item in items:
        if "：" in item:
            label, value = item.split("：", 1)
        else:
            label, value = "结果", item
        cards.append(
            {
                "label": label,
                "value": value,
                "tone": "violet" if "置信度" in label else "blue",
            }
        )
    st.markdown(build_stat_grid(cards), unsafe_allow_html=True)


def build_mac_minimize_demo(demo_id: str = "home-mac-demo") -> str:
    safe_id = "".join(ch if ch.isalnum() else "-" for ch in demo_id).strip("-") or "home-mac-demo"
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8" />
      <style>
        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: "SF Pro Display", "SF Pro Text", -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif;
          color: #1d1d1f;
        }

        .genie-demo {
          padding: 0.25rem 0;
        }

        .genie-shell {
          position: relative;
          overflow: hidden;
          border-radius: 28px;
          border: 1px solid rgba(222, 228, 238, 0.92);
          background:
            radial-gradient(circle at top right, rgba(120, 200, 255, 0.18), transparent 24%),
            radial-gradient(circle at bottom left, rgba(10, 132, 255, 0.08), transparent 28%),
            linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 249, 255, 0.96));
          box-shadow: 0 26px 60px rgba(15, 23, 42, 0.1);
        }

        .genie-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: 1rem 1.1rem 0.4rem;
        }

        .genie-kicker {
          display: inline-flex;
          align-items: center;
          padding: 0.28rem 0.58rem;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: #1259b7;
          background: rgba(10, 132, 255, 0.1);
          border: 1px solid rgba(10, 132, 255, 0.12);
        }

        .genie-status {
          font-size: 0.78rem;
          color: #6e6e73;
        }

        .genie-copy {
          padding: 0 1.1rem 0.6rem;
        }

        .genie-copy h3 {
          margin: 0;
          font-size: 1.08rem;
          font-weight: 700;
          color: #1d1d1f;
        }

        .genie-copy p {
          margin: 0.42rem 0 0;
          font-size: 0.84rem;
          line-height: 1.65;
          color: #6e6e73;
        }

        .genie-stage {
          position: relative;
          min-height: 318px;
          padding: 0 0.85rem 0.95rem;
        }

        .genie-stage::before {
          content: "";
          position: absolute;
          inset: 0.55rem 0.85rem 1rem;
          border-radius: 24px;
          background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.68), rgba(248, 251, 255, 0.8)),
            repeating-linear-gradient(
              135deg,
              rgba(255, 255, 255, 0.16) 0,
              rgba(255, 255, 255, 0.16) 8px,
              rgba(235, 243, 252, 0.12) 8px,
              rgba(235, 243, 252, 0.12) 16px
            );
          border: 1px solid rgba(230, 236, 244, 0.88);
          pointer-events: none;
        }

        .genie-canvas {
          position: absolute;
          inset: 0.55rem 0.85rem 1rem;
          width: calc(100% - 1.7rem);
          height: calc(100% - 1.55rem);
          display: block;
        }

        .window-controls {
          position: absolute;
          top: 1.35rem;
          left: 50%;
          width: min(calc(100% - 3.4rem), 440px);
          transform: translateX(-50%);
          pointer-events: none;
        }

        .traffic-lights {
          display: inline-flex;
          align-items: center;
          gap: 0.42rem;
          padding-left: 0.42rem;
          pointer-events: auto;
        }

        .traffic-dot {
          width: 0.78rem;
          height: 0.78rem;
          border: 0;
          border-radius: 999px;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
        }

        .traffic-dot.red {
          background: linear-gradient(180deg, #ff7b72, #f04438);
        }

        .traffic-dot.yellow {
          cursor: pointer;
          background: linear-gradient(180deg, #ffd666, #f5a524);
        }

        .traffic-dot.green {
          background: linear-gradient(180deg, #7ee787, #22c55e);
        }

        .genie-dock {
          position: absolute;
          left: 50%;
          bottom: 0.9rem;
          display: inline-flex;
          align-items: flex-end;
          gap: 0.65rem;
          padding: 0.72rem 0.85rem;
          border-radius: 1.35rem;
          transform: translateX(-50%);
          border: 1px solid rgba(221, 228, 236, 0.96);
          background: rgba(255, 255, 255, 0.74);
          box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
          backdrop-filter: blur(18px) saturate(165%);
          -webkit-backdrop-filter: blur(18px) saturate(165%);
        }

        .dock-item {
          position: relative;
          width: 2.8rem;
          height: 2.8rem;
          border-radius: 1rem;
          border: 1px solid rgba(210, 220, 232, 0.92);
          background: linear-gradient(180deg, #f7fbff, #dfeeff);
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82), 0 9px 18px rgba(15, 23, 42, 0.08);
        }

        .dock-item::before {
          content: "";
          position: absolute;
          inset: 0.56rem;
          border-radius: 0.72rem;
          background: linear-gradient(180deg, rgba(94, 92, 230, 0.24), rgba(10, 132, 255, 0.16));
        }

        .dock-item.soft::before {
          background: linear-gradient(180deg, rgba(34, 197, 94, 0.24), rgba(16, 185, 129, 0.16));
        }

        .dock-item.warm::before {
          background: linear-gradient(180deg, rgba(245, 158, 11, 0.24), rgba(251, 191, 36, 0.16));
        }

        button.dock-item {
          cursor: pointer;
          background: linear-gradient(180deg, #fdfefe, #eaf5ff);
        }

        button.dock-item::after {
          content: "";
          position: absolute;
          left: 50%;
          bottom: -0.6rem;
          width: 0.35rem;
          height: 0.35rem;
          border-radius: 999px;
          background: rgba(18, 89, 183, 0.22);
          transform: translateX(-50%);
          transition: transform 0.35s ease, opacity 0.35s ease, background 0.35s ease;
          opacity: 0.55;
        }

        .genie-demo.is-minimized button.dock-item {
          animation: dock-bounce 0.68s cubic-bezier(0.2, 0.9, 0.2, 1) both;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 16px 28px rgba(10, 132, 255, 0.2);
          background: linear-gradient(180deg, #ffffff, #d9ebff);
        }

        .genie-demo.is-minimized button.dock-item::after {
          background: rgba(10, 132, 255, 0.72);
          transform: translateX(-50%) scale(1.12);
          opacity: 1;
        }

        .genie-hint {
          padding: 0 1.1rem 1rem;
          text-align: center;
          font-size: 0.8rem;
          line-height: 1.6;
          color: #6e6e73;
          transition: color 0.35s ease;
        }

        .genie-demo.is-minimized .genie-hint {
          color: #1259b7;
        }

        @keyframes dock-bounce {
          0% {
            transform: translateY(0) scale(1);
          }
          45% {
            transform: translateY(-0.56rem) scale(1.12);
          }
          72% {
            transform: translateY(-0.24rem) scale(1.05);
          }
          100% {
            transform: translateY(-0.38rem) scale(1.08);
          }
        }

        @media (max-width: 640px) {
          .genie-shell {
            border-radius: 24px;
          }

          .genie-copy h3 {
            font-size: 1rem;
          }

          .genie-stage {
            min-height: 292px;
          }

          .dock-item {
            width: 2.45rem;
            height: 2.45rem;
          }
        }
      </style>
    </head>
    <body>
      <div id="__DEMO_ID__" class="genie-demo">
        <div class="genie-shell">
          <div class="genie-head">
            <span class="genie-kicker">Genie Effect</span>
            <span class="genie-status">Stage 3 Preview</span>
          </div>
          <div class="genie-copy">
            <h3>更接近 macOS 的窗口最小化演示</h3>
            <p>参考你提供的 OpenGL 方案思路，这里改成了逐条扫描线插值与横向扭曲，让窗口收缩时更像 macOS 的 Genie Effect，而不是普通缩放。</p>
          </div>
          <div class="genie-stage">
            <canvas class="genie-canvas"></canvas>
            <div class="window-controls">
              <div class="traffic-lights">
                <span class="traffic-dot red"></span>
                <button class="traffic-dot yellow" type="button" title="最小化窗口" aria-label="最小化窗口"></button>
                <span class="traffic-dot green"></span>
              </div>
            </div>
            <div class="genie-dock">
              <div class="dock-item soft"></div>
              <button class="dock-item primary" type="button" title="从 Dock 恢复" aria-label="从 Dock 恢复"></button>
              <div class="dock-item warm"></div>
            </div>
          </div>
          <div class="genie-hint">点黄色按钮最小化，点 Dock 中间图标恢复。</div>
        </div>
      </div>

      <script>
        (() => {
          const root = document.getElementById("__DEMO_ID__");
          if (!root) {
            return;
          }

          const stage = root.querySelector(".genie-stage");
          const canvas = root.querySelector(".genie-canvas");
          const ctx = canvas.getContext("2d");
          const minimizeButton = root.querySelector(".traffic-dot.yellow");
          const restoreButton = root.querySelector(".dock-item.primary");
          const offscreen = document.createElement("canvas");
          const sourceWidth = 920;
          const sourceHeight = 560;
          offscreen.width = sourceWidth;
          offscreen.height = sourceHeight;
          const offCtx = offscreen.getContext("2d");

          let progress = 0;
          let target = 0;
          let rafId = null;

          function clamp(value, min, max) {
            return Math.min(max, Math.max(min, value));
          }

          function lerp(start, end, amount) {
            return start + (end - start) * amount;
          }

          function easeInOutCubic(value) {
            return value < 0.5
              ? 4 * value * value * value
              : 1 - Math.pow(-2 * value + 2, 3) / 2;
          }

          function roundRectPath(context, x, y, width, height, radius) {
            const r = Math.min(radius, width / 2, height / 2);
            context.beginPath();
            context.moveTo(x + r, y);
            context.arcTo(x + width, y, x + width, y + height, r);
            context.arcTo(x + width, y + height, x, y + height, r);
            context.arcTo(x, y + height, x, y, r);
            context.arcTo(x, y, x + width, y, r);
            context.closePath();
          }

          function fillRoundRect(context, x, y, width, height, radius, fillStyle) {
            context.save();
            roundRectPath(context, x, y, width, height, radius);
            context.fillStyle = fillStyle;
            context.fill();
            context.restore();
          }

          function strokeRoundRect(context, x, y, width, height, radius, strokeStyle, lineWidth = 1) {
            context.save();
            roundRectPath(context, x, y, width, height, radius);
            context.lineWidth = lineWidth;
            context.strokeStyle = strokeStyle;
            context.stroke();
            context.restore();
          }

          function drawSourceWindow() {
            offCtx.clearRect(0, 0, sourceWidth, sourceHeight);
            const x = 32;
            const y = 18;
            const width = sourceWidth - 64;
            const height = sourceHeight - 36;
            const radius = 34;

            offCtx.save();
            offCtx.shadowColor = "rgba(15, 23, 42, 0.18)";
            offCtx.shadowBlur = 42;
            offCtx.shadowOffsetY = 26;
            fillRoundRect(offCtx, x, y + 6, width, height - 4, radius, "rgba(242, 247, 255, 0.98)");
            offCtx.restore();

            const shellGradient = offCtx.createLinearGradient(0, y, 0, y + height);
            shellGradient.addColorStop(0, "rgba(255, 255, 255, 0.99)");
            shellGradient.addColorStop(1, "rgba(245, 249, 255, 0.97)");
            fillRoundRect(offCtx, x, y, width, height, radius, shellGradient);
            strokeRoundRect(offCtx, x, y, width, height, radius, "rgba(214, 220, 228, 0.98)", 2);

            const toolbarHeight = 64;
            const toolbarGradient = offCtx.createLinearGradient(0, y, 0, y + toolbarHeight);
            toolbarGradient.addColorStop(0, "rgba(250, 251, 253, 0.98)");
            toolbarGradient.addColorStop(1, "rgba(244, 246, 250, 0.96)");
            fillRoundRect(offCtx, x, y, width, toolbarHeight, radius, toolbarGradient);
            offCtx.save();
            offCtx.beginPath();
            offCtx.rect(x, y + 34, width, toolbarHeight);
            offCtx.clip();
            offCtx.clearRect(x, y + 34, width, toolbarHeight);
            offCtx.restore();

            offCtx.strokeStyle = "rgba(226, 232, 240, 0.92)";
            offCtx.lineWidth = 2;
            offCtx.beginPath();
            offCtx.moveTo(x, y + toolbarHeight);
            offCtx.lineTo(x + width, y + toolbarHeight);
            offCtx.stroke();

            const trafficY = y + 26;
            const trafficX = x + 28;
            const dotColors = [
              ["#ff7b72", "#f04438"],
              ["#ffd666", "#f5a524"],
              ["#7ee787", "#22c55e"],
            ];

            dotColors.forEach((pair, index) => {
              const gradient = offCtx.createLinearGradient(0, trafficY - 8, 0, trafficY + 8);
              gradient.addColorStop(0, pair[0]);
              gradient.addColorStop(1, pair[1]);
              offCtx.beginPath();
              offCtx.fillStyle = gradient;
              offCtx.arc(trafficX + index * 22, trafficY, 8, 0, Math.PI * 2);
              offCtx.fill();
            });

            offCtx.fillStyle = "#475569";
            offCtx.font = "700 22px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif";
            offCtx.textAlign = "center";
            offCtx.fillText("Stage 3 Preview", x + width / 2, y + 33);

            fillRoundRect(offCtx, x + width - 98, y + 16, 66, 30, 15, "rgba(10, 132, 255, 0.1)");
            strokeRoundRect(offCtx, x + width - 98, y + 16, 66, 30, 15, "rgba(10, 132, 255, 0.16)", 1.2);
            offCtx.fillStyle = "#1259b7";
            offCtx.font = "700 16px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif";
            offCtx.fillText("Live", x + width - 65, y + 36);

            const bodyY = y + toolbarHeight;
            const heroX = x + 28;
            const heroY = bodyY + 28;
            const heroW = width - 56;
            const heroH = 188;
            const heroGradient = offCtx.createLinearGradient(heroX, heroY, heroX + heroW, heroY + heroH);
            heroGradient.addColorStop(0, "#f8fbff");
            heroGradient.addColorStop(0.5, "#eaf3ff");
            heroGradient.addColorStop(1, "#f5f8ff");
            fillRoundRect(offCtx, heroX, heroY, heroW, heroH, 28, heroGradient);
            strokeRoundRect(offCtx, heroX, heroY, heroW, heroH, 28, "rgba(215, 226, 240, 0.94)", 1.4);

            const glowA = offCtx.createRadialGradient(heroX + heroW * 0.76, heroY + 42, 0, heroX + heroW * 0.76, heroY + 42, 80);
            glowA.addColorStop(0, "rgba(100, 210, 255, 0.32)");
            glowA.addColorStop(1, "rgba(100, 210, 255, 0)");
            offCtx.fillStyle = glowA;
            offCtx.fillRect(heroX, heroY, heroW, heroH);

            const glowB = offCtx.createRadialGradient(heroX + heroW * 0.2, heroY + heroH * 0.8, 0, heroX + heroW * 0.2, heroY + heroH * 0.8, 88);
            glowB.addColorStop(0, "rgba(94, 92, 230, 0.18)");
            glowB.addColorStop(1, "rgba(94, 92, 230, 0)");
            offCtx.fillStyle = glowB;
            offCtx.fillRect(heroX, heroY, heroW, heroH);

            const cardY = heroY + heroH + 24;
            const cardGap = 16;
            const cardW = (heroW - cardGap * 2) / 3;
            const cardH = 92;
            const cardData = [
              ["Detect", "几何分析面板"],
              ["Classify", "7 类积木识别"],
              ["About", "阶段成果总览"],
            ];

            cardData.forEach((card, index) => {
              const cardX = heroX + index * (cardW + cardGap);
              fillRoundRect(offCtx, cardX, cardY, cardW, cardH, 22, "rgba(255, 255, 255, 0.92)");
              strokeRoundRect(offCtx, cardX, cardY, cardW, cardH, 22, "rgba(224, 232, 240, 0.96)", 1.4);
              offCtx.fillStyle = "#1d1d1f";
              offCtx.font = "800 18px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif";
              offCtx.textAlign = "left";
              offCtx.fillText(card[0], cardX + 18, cardY + 34);
              offCtx.fillStyle = "#6e6e73";
              offCtx.font = "16px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif";
              offCtx.fillText(card[1], cardX + 18, cardY + 60);
            });

            const lineY = cardY + cardH + 24;
            const lineWidths = [heroW, heroW * 0.83, heroW * 0.64];
            lineWidths.forEach((lineWidth, index) => {
              const gradient = offCtx.createLinearGradient(heroX, 0, heroX + lineWidth, 0);
              gradient.addColorStop(0, "rgba(214, 228, 248, 0.92)");
              gradient.addColorStop(1, "rgba(235, 242, 252, 0.96)");
              fillRoundRect(offCtx, heroX, lineY + index * 20, lineWidth, 12, 999, gradient);
            });
          }

          function drawStaticWindow(width, height, windowX, windowY, windowWidth, windowHeight) {
            ctx.drawImage(offscreen, 0, 0, sourceWidth, sourceHeight, windowX, windowY, windowWidth, windowHeight);
          }

          function drawGenieWindow(width, height, windowX, windowY, windowWidth, windowHeight, progressValue) {
            const eased = easeInOutCubic(progressValue);
            const strips = 120;
            const dockCenterX = width / 2;
            const dockCenterY = height - 34;
            const dockWidth = clamp(windowWidth * 0.16, 52, 74);
            const dockHeight = 22;

            ctx.save();
            for (let strip = 0; strip < strips; strip += 1) {
              const y0 = strip / strips;
              const y1 = (strip + 1) / strips;
              const yMid = (y0 + y1) * 0.5;
              const curve = Math.pow(yMid, 1.55);
              const stripProgress = 1 - Math.pow(1 - eased, 0.58 + curve * 1.55);
              const widthProgress = 1 - Math.pow(1 - eased, 0.94 + curve * 0.42);
              const localCenterX = lerp(windowX + windowWidth / 2, dockCenterX, stripProgress);
              const targetWidth = dockWidth * (1.06 + 0.14 * Math.sin(yMid * Math.PI));
              const localWidth = Math.max(1.2, lerp(windowWidth, targetWidth, widthProgress));
              const bend = Math.sin(Math.PI * Math.min(stripProgress, 0.999)) * (1 - curve) * 34 * (1 - eased * 0.14);
              const drawX = localCenterX - localWidth / 2 - bend * 0.46;
              const targetY = dockCenterY - dockHeight / 2 + y0 * dockHeight;
              const drawY = lerp(windowY + y0 * windowHeight, targetY, stripProgress);
              const drawHeight = Math.max(1.0, lerp(windowHeight / strips, dockHeight / strips, stripProgress));
              const drawWidth = localWidth + bend;

              ctx.globalAlpha = lerp(1, 0.08, stripProgress * (0.54 + 0.46 * curve));
              ctx.drawImage(
                offscreen,
                0,
                y0 * sourceHeight,
                sourceWidth,
                Math.ceil((y1 - y0) * sourceHeight) + 1,
                drawX,
                drawY,
                drawWidth,
                drawHeight
              );
            }
            ctx.restore();
          }

          function render() {
            const bounds = stage.getBoundingClientRect();
            const width = Math.max(bounds.width, 280);
            const height = Math.max(bounds.height, 260);
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            canvas.style.width = width + "px";
            canvas.style.height = height + "px";
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, width, height);

            const windowWidth = Math.min(width - 42, 440);
            const windowHeight = Math.min(height - 110, 274);
            const windowX = (width - windowWidth) / 2;
            const windowY = 18;

            ctx.save();
            ctx.globalAlpha = 0.17 * (1 - progress);
            ctx.shadowColor = "rgba(15, 23, 42, 0.24)";
            ctx.shadowBlur = 28;
            ctx.shadowOffsetY = 18;
            fillRoundRect(ctx, windowX + 10, windowY + 22, windowWidth - 20, windowHeight - 18, 30, "rgba(15, 23, 42, 0.18)");
            ctx.restore();

            if (progress < 0.001) {
              drawStaticWindow(width, height, windowX, windowY, windowWidth, windowHeight);
            } else {
              drawGenieWindow(width, height, windowX, windowY, windowWidth, windowHeight, progress);
            }

            root.classList.toggle("is-minimized", progress > 0.72);
          }

          function tick() {
            const delta = target - progress;
            if (Math.abs(delta) < 0.002) {
              progress = target;
              render();
              rafId = null;
              return;
            }

            progress += delta * 0.16;
            progress = clamp(progress, 0, 1);
            render();
            rafId = window.requestAnimationFrame(tick);
          }

          function animateTo(nextTarget) {
            target = clamp(nextTarget, 0, 1);
            if (!rafId) {
              rafId = window.requestAnimationFrame(tick);
            }
          }

          drawSourceWindow();
          render();

          minimizeButton.addEventListener("click", () => animateTo(1));
          restoreButton.addEventListener("click", () => animateTo(0));

          if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(() => render());
            observer.observe(stage);
          } else {
            window.addEventListener("resize", render);
          }
        })();
      </script>
    </body>
    </html>
    """.replace("__DEMO_ID__", safe_id)


def main() -> None:
    model_info = build_stage3_model_info()
    about_context = build_about_context()
    stage1 = about_context["stage1"]
    stage2 = about_context["stage2"]

    st.markdown(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.08em; text-transform:uppercase; opacity:0.84; font-weight:700;">Stage 3 / Streamlit Application</p>
          <h1 style="margin:0.35rem 0 0.7rem 0;">乐高检测与识别应用</h1>
          <p style="margin:0;">首页保留几何检测能力，分类模型、示例体验和阶段总览已迁移到 Streamlit 多页面结构中。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        build_badge_row(
            [
                {"label": "多页面导航", "tone": "blue"},
                {"label": "几何检测", "tone": "violet"},
                {"label": "7 类分类模型", "tone": "success"},
                {"label": "课堂演示友好", "tone": "slate"},
            ]
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.success("当前已切换为 Streamlit 多页面应用。")
    st.sidebar.caption("左侧侧边栏可切换到分类识别、示例体验和关于阶段页面。")
    st.sidebar.metric("当前模型准确率", f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录")
    st.sidebar.metric("当前类别数", model_info["class_count"])
    st.sidebar.metric("阶段 1 图片数", about_context["stage1"]["total_images"])

    st.markdown(
        build_stat_grid(
            [
                {"label": "应用框架", "value": "Streamlit", "note": "多页面结构", "tone": "blue"},
                {
                    "label": "当前分类模型",
                    "value": model_info["checkpoint_name"],
                    "note": f"设备 {model_info['device']}",
                    "tone": "violet",
                },
                {
                    "label": "测试准确率",
                    "value": f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录",
                    "note": f"Loss {model_info['loss']}" if model_info["loss"] is not None else "评估文件未记录",
                    "tone": "success",
                },
                {
                    "label": "当前类别数",
                    "value": model_info["class_count"],
                    "note": "默认接入 7 类 Brick",
                    "tone": "slate",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    overview_tab, geometry_tab = st.tabs(["项目概览", "几何检测"])

    with overview_tab:
        left_col, right_col = st.columns([1.1, 0.9], gap="large")
        with left_col:
            st.markdown(
                build_panel_card(
                    "当前阶段能力",
                    """
                    <ul class="feature-list">
                      <li>首页继续保留单主体尺寸/层高检测和多主体堆叠检测。</li>
                      <li><span class="inline-chip">Classification</span> 页面接入第二阶段 7 类 CNN 模型做图片识别。</li>
                      <li><span class="inline-chip">Examples</span> 页面直接展示测试集示例和概率分布图。</li>
                      <li><span class="inline-chip">About</span> 页面汇总第一、二、三阶段关键结果，方便展示。</li>
                    </ul>
                    """,
                    eyebrow="Overview",
                    footer_html=build_badge_row(
                        [
                            {"label": "检测 + 识别", "tone": "blue"},
                            {"label": "示例 + 汇总", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )

        with right_col:
            st.markdown(
                build_panel_card(
                    "模型与数据快照",
                    build_key_value_list(
                        [
                            ("当前 checkpoint", model_info["checkpoint_name"]),
                            ("数据根目录", model_info["data_root"]),
                            ("自动裁剪", "开启" if model_info["auto_crop"] else "关闭"),
                            ("阶段 1 总图片", stage1["total_images"]),
                        ]
                    ),
                    eyebrow="Live Snapshot",
                    footer_html=build_badge_row(
                        [
                            {"label": f"Train {stage1['split_totals']['train']}", "tone": "blue"},
                            {"label": f"Val {stage1['split_totals']['val']}", "tone": "slate"},
                            {"label": f"Test {stage1['split_totals']['test']}", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )

        quick_col, stage_col = st.columns([0.95, 1.05], gap="large")
        with quick_col:
            st.markdown(
                build_mac_window_card(
                    "home-quick-start-window",
                    "运行方式",
                    (
                        "<div class='command-block'>streamlit run app.py</div>"
                        "<p class='muted-note'>分类识别、示例体验和关于阶段页面都可以通过左侧侧边栏直接切换。</p>"
                    ),
                    eyebrow="Quick Start",
                    subtitle="这里已经不是演示窗口，而是首页真实使用的可收起模块。",
                    footer_html=build_badge_row(
                        [
                            {"label": "真实可收起模块", "tone": "success"},
                            {"label": "红黄绿都可交互", "tone": "slate"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
        with stage_col:
            st.markdown(
                build_mac_window_card(
                    "home-stage-summary-window",
                    "阶段摘要",
                    build_key_value_list(
                        [
                            ("阶段 1 划分", f"{stage1['split_totals']['train']} / {stage1['split_totals']['val']} / {stage1['split_totals']['test']}"),
                            ("阶段 2 划分", f"{stage2['split_totals']['train']} / {stage2['split_totals']['val']} / {stage2['split_totals']['test']}"),
                            ("格式分布", str(stage1["format_distribution"])),
                            ("当前设备", model_info["device"]),
                        ]
                    ),
                    eyebrow="Stage Summary",
                    subtitle="阶段摘要也可以像 macOS 窗口一样收起，再从下方 Dock 样式卡片恢复。",
                    footer_html=build_badge_row(
                        [
                            {"label": "已应用真实动效", "tone": "violet"},
                            {"label": "不是演示组件", "tone": "blue"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )

    with geometry_tab:
        intro_col, mode_col = st.columns([1.05, 0.95], gap="large")
        with intro_col:
            st.markdown(
                build_panel_card(
                    "图片几何检测",
                    "<p class='muted-note'>这个页面保留原来的几何检测能力，适合展示尺寸、层高和多主体堆叠分析结果。</p>",
                    eyebrow="Geometry",
                    footer_html=build_badge_row(
                        [
                            {"label": "尺寸检测", "tone": "blue"},
                            {"label": "层高检测", "tone": "violet"},
                            {"label": "多主体堆叠", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
        with mode_col:
            st.markdown(
                build_panel_card(
                    "检测模式说明",
                    build_key_value_list(
                        [
                            ("single", "同时输出尺寸和层高"),
                            ("size_only", "只关注平面尺寸"),
                            ("height_only", "只判断层高"),
                            ("multi_stack", "识别多主体堆叠层数"),
                        ]
                    ),
                    eyebrow="Modes",
                    tone="slate",
                ),
                unsafe_allow_html=True,
            )

        with st.form("geometry_detection_form"):
            upload_col, mode_col = st.columns([1.05, 0.95], gap="large")
            with upload_col:
                uploaded_file = st.file_uploader(
                    "上传待检测图片",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    help="支持真实乐高照片或实验演示图片。",
                )
            with mode_col:
                selected_mode = st.selectbox(
                    "选择检测模式",
                    options=list(DETECT_MODE_OPTIONS.keys()),
                    format_func=lambda key: DETECT_MODE_OPTIONS[key],
                )
            submitted = st.form_submit_button("开始检测", use_container_width=True)

        if submitted:
            if uploaded_file is None:
                st.warning("请先上传一张图片。")
            else:
                try:
                    image = image_bytes_to_pil(uploaded_file.getvalue())
                    original_bgr = pil_to_bgr(image)
                    result = run_geometry_detection(original_bgr, selected_mode)

                    original_col, annotated_col = st.columns(2, gap="large")
                    with original_col:
                        st.markdown("#### 原图")
                        st.image(image, use_container_width=True)
                    with annotated_col:
                        st.markdown("#### 标注图")
                        st.image(bgr_to_rgb(result["annotated_image"]), use_container_width=True)

                    st.markdown("#### 识别摘要")
                    render_summary_chips(result["summary"])

                    if result["objects"]:
                        st.markdown("#### 对象明细")
                        st.markdown(
                            build_stat_grid(
                                [
                                    {
                                        "label": item["name"],
                                        "value": item["shape"],
                                        "note": f"置信度 {item['confidence']}",
                                        "tone": "success",
                                    }
                                    for item in result["objects"]
                                ]
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            build_panel_card(
                                "检测说明",
                                "<p class='muted-note'>当前模式主要返回摘要信息；如果没有对象明细，这是正常结果。</p>",
                                eyebrow="Result Note",
                                tone="slate",
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception as exc:
                    st.error(str(exc))


main()

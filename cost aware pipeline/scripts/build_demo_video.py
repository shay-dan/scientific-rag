from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "docs" / "demo" / "captures"
OUTPUT = ROOT / "docs" / "demo" / "scientific-rag-demo.mp4"
PREVIEW = ROOT / "docs" / "demo" / "demo-contact-sheet.png"

WIDTH, HEIGHT, FPS = 1280, 720, 15
BG = "#07111f"
PANEL = "#0e1d30"
WHITE = "#f5f8fc"
MUTED = "#a9b8cb"
TEAL = "#43d6c5"
BLUE = "#58a6ff"
GOLD = "#f4c95d"


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def wrap(draw, text, width, fnt):
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def base_frame(kicker, title, body, step=None):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    d.rounded_rectangle((28, 28, 444, 692), radius=26, fill=PANEL)
    d.text((58, 62), kicker.upper(), font=font(18, True), fill=TEAL)
    y = 105
    for line in wrap(d, title, 330, font(39, True)):
        d.text((58, y), line, font=font(39, True), fill=WHITE)
        y += 47
    y += 18
    for line in wrap(d, body, 330, font(23)):
        d.text((58, y), line, font=font(23), fill=MUTED)
        y += 33
    if step:
        d.text((58, 642), step, font=font(17, True), fill=BLUE)
    d.rounded_rectangle((470, 28, 1252, 692), radius=26, fill="#f8fafc")
    return frame


def paste_crop(frame, source, box, *, contain=False):
    crop = source.crop(box)
    area = (488, 46, 1234, 674)
    aw, ah = area[2] - area[0], area[3] - area[1]
    scale = min(aw / crop.width, ah / crop.height) if contain else max(aw / crop.width, ah / crop.height)
    resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
    x = area[0] + (aw - resized.width) // 2
    y = area[1] + (ah - resized.height) // 2
    frame.paste(resized, (x, y))


def title_scene(progress):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    d.ellipse((825, -80, 1390, 485), fill="#0d2d42")
    d.ellipse((950, 345, 1320, 715), fill="#12324c")
    d.text((72, 92), "SCIENTIFIC RAG", font=font(22, True), fill=TEAL)
    d.text((72, 174), "Evidence-first", font=font(61, True), fill=WHITE)
    d.text((72, 245), "research assistant", font=font(61, True), fill=WHITE)
    d.text((76, 355), "FastAPI + PostgreSQL/pgvector + Redis", font=font(28), fill=MUTED)
    d.rounded_rectangle((72, 452, 553, 518), radius=18, fill="#12324c")
    d.text((97, 470), "Saved-run walkthrough • no live generation", font=font(20, True), fill=BLUE)
    d.text((76, 632), "Docker-ready • Source-grounded • Evaluated", font=font(21), fill=GOLD)
    return frame


def screenshot_scene(source, box, kicker, title, body, step, contain=False):
    frame = base_frame(kicker, title, body, step)
    paste_crop(frame, source, box, contain=contain)
    return frame


def metrics_scene(source):
    frame = base_frame(
        "QUALITY EVALUATION",
        "Measured on 10 research cases",
        "Internal benchmark across biomedical and technical topics. Nine cases used Ragas; one used the documented fallback judge.",
        "04 / 05  •  Benchmark results",
    )
    d = ImageDraw.Draw(frame)
    metrics = [
        ("Faithfulness", "0.9672", TEAL),
        ("Answer relevancy", "0.8641", BLUE),
        ("Context precision", "0.5047", GOLD),
        ("Context recall", "0.5933", "#f18f6d"),
    ]
    x0, y0 = 500, 112
    for i, (label, value, color) in enumerate(metrics):
        x = x0 + (i % 2) * 355
        y = y0 + (i // 2) * 232
        d.rounded_rectangle((x, y, x + 320, y + 190), radius=22, fill="#eaf0f7", outline="#d3deea", width=2)
        d.text((x + 24, y + 28), label, font=font(22, True), fill="#21354d")
        d.text((x + 24, y + 82), value, font=font(50, True), fill=color)
    d.text((500, 584), "Aggregate scores • not a claim of clinical accuracy", font=font(19, True), fill="#52677f")
    return frame


def evaluation_mode_scene(mode):
    descriptions = {
        "claim": (
            "CLAIM-AWARE VIEW",
            "Useful evidence, measured per claim",
            "Recommended: an excerpt counts when it supports at least one material reference claim. Partial support receives half credit.",
            "Claim-aware (Recommended)",
        ),
        "ragas": (
            "ORIGINAL RAGAS",
            "Keep the third-party score visible",
            "The installed Ragas metric compares each excerpt with the complete reference answer, so useful partial evidence may still receive zero.",
            "Original Ragas",
        ),
        "compare": (
            "COMPARE BOTH",
            "Different metrics answer different questions",
            "The gap separates complete-answer sufficiency from claim-level usefulness. Neither score is overwritten or hidden.",
            "Compare both",
        ),
    }
    kicker, title, body, selected = descriptions[mode]
    step = {"claim": "02 / 05  •  Recommended view", "ragas": "03 / 05  •  Original metric", "compare": "04 / 05  •  Side-by-side"}[mode]
    frame = base_frame(kicker, title, body, step)
    d = ImageDraw.Draw(frame)
    x, y = 500, 58
    d.text((x, y), "COMPLETED", font=font(15, True), fill="#0c7059")
    d.text((x, y + 31), "Ragas results", font=font(31, True), fill="#17201e")
    d.text((x, y + 73), "Saved Standard run · no new API call", font=font(16), fill="#62716c")
    d.text((x, y + 113), "Context precision view", font=font(16, True), fill="#17201e")
    d.rounded_rectangle((x, y + 139, 1205, y + 188), radius=9, fill="#ffffff", outline="#9fb2a8", width=2)
    d.text((x + 17, y + 152), selected, font=font(18, True), fill="#17201e")
    d.polygon([(1172, y + 158), (1188, y + 158), (1180, y + 168)], fill="#62716c")

    help_text = {
        "claim": "Counts excerpts that support at least one material claim; rank still matters.",
        "ragas": "Unmodified Ragas 0.2.x score; partial evidence can be judged as insufficient.",
        "compare": "Shows original Ragas and claim-aware precision together for interpretation.",
    }[mode]
    d.rounded_rectangle((x, y + 204, 1205, y + 263), radius=10, fill="#f0f4ef")
    for line_index, line in enumerate(wrap(d, help_text, 660, font(16))):
        d.text((x + 16, y + 216 + line_index * 22), line, font=font(16), fill="#52635c")

    if mode == "claim":
        metrics = [("Claim-aware partial", "1.000", TEAL), ("Claim-aware strict", "1.000", BLUE), ("Relevant excerpts", "3 / 3", "#0c7059"), ("Ragas recall", "0.333", GOLD)]
    elif mode == "ragas":
        metrics = [("Original Ragas precision", "0.000", "#d2674d"), ("Faithfulness", "0.000", "#d2674d"), ("Answer relevancy", "0.525", BLUE), ("Ragas recall", "0.333", GOLD)]
    else:
        metrics = [("Original Ragas", "0.000", "#d2674d"), ("Claim-aware partial", "1.000", TEAL), ("Claim-aware strict", "1.000", BLUE), ("Relevant excerpts", "3 / 3", "#0c7059")]

    for index, (label, value, color) in enumerate(metrics):
        column, row = index % 2, index // 2
        bx, by = x + column * 354, y + 282 + row * 133
        d.rounded_rectangle((bx, by, bx + 330, by + 112), radius=14, fill="#eef9f4", outline="#dce2dc")
        d.text((bx + 16, by + 15), label, font=font(15, True), fill="#62716c")
        d.text((bx + 16, by + 46), value, font=font(32, True), fill=color)
    d.text((x, 638), "Neural operators for partial differential equations · saved evaluation", font=font(15), fill="#62716c")
    return frame


def retrieval_audit_scene(progress):
    frame = base_frame(
        "AUDITABLE EVALUATION",
        "See why each excerpt counted",
        "The claim-aware audit is persisted with the run. Review the verdict, supported claims, rationale, and rank contribution.",
        "05 / 05  •  Per-excerpt audit",
    )
    d = ImageDraw.Draw(frame)
    d.text((500, 62), "Per-excerpt retrieval audit", font=font(29, True), fill="#17201e")
    d.text((500, 104), "3 relevant · 0 partial · 0 irrelevant", font=font(18, True), fill="#0c7059")
    d.rounded_rectangle((500, 143, 1215, 205), radius=12, fill="#eef9f4")
    d.text((520, 158), "Strict ranked precision: 1.000", font=font(18, True), fill="#17201e")
    d.text((865, 158), "Partial-credit: 1.000", font=font(18, True), fill="#17201e")
    rows = [
        ("#1 · RELEVANT", "Supports a material reference claim directly."),
        ("#2 · RELEVANT", "Adds complementary evidence for the answer."),
        ("#3 · RELEVANT", "On-topic evidence retained at a lower rank."),
    ]
    for index, (heading, reason) in enumerate(rows):
        ry = 226 + index * 124
        d.rounded_rectangle((500, ry, 1215, ry + 103), radius=13, fill="#f6f8f5", outline="#dce2dc")
        d.text((520, ry + 16), heading, font=font(18, True), fill="#0c7059")
        d.text((520, ry + 49), reason, font=font(17), fill="#52635c")
        d.text((1043, ry + 17), f"rank {index + 1}", font=font(14, True), fill="#58a6ff")
    return frame


def end_scene(progress):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    d.text((73, 86), "TRY IT LOCALLY", font=font(21, True), fill=TEAL)
    d.text((73, 150), "Run with Docker", font=font(58, True), fill=WHITE)
    d.text((76, 238), "Clone the repository, copy .env.example to .env,", font=font(25), fill=MUTED)
    d.text((76, 278), "add provider keys, then start the stack.", font=font(25), fill=MUTED)
    d.rounded_rectangle((74, 367, 1040, 502), radius=22, fill="#0e1d30")
    d.text((108, 397), "docker compose up --build", font=font(34, True), fill=BLUE)
    d.text((108, 452), "Open http://localhost:8000", font=font(24), fill=WHITE)
    d.text((76, 620), "Setup instructions and evaluation details are in README.md", font=font(22, True), fill=GOLD)
    return frame


def fade(frame, local_t, duration, fade_time=0.45):
    alpha = min(1.0, local_t / fade_time, (duration - local_t) / fade_time)
    if alpha >= 1:
        return frame
    black = Image.new("RGB", frame.size, BG)
    return Image.blend(black, frame, max(0.0, alpha))


def main():
    benchmark = Image.open(CAPTURES / "benchmark.png").convert("RGB")
    scenes = [
        (5.0, lambda p: title_scene(p)),
        (7.0, lambda p: screenshot_scene(
            benchmark, (0, 1100, 652, 2250),
            "SAVED EVALUATION", "Inspect quality without rerunning models",
            "Open a completed Fast or Standard run. The answer, retrieved excerpts, sources, and judge output remain available.",
            "01 / 05  •  Evaluation history", contain=True)),
        (9.0, lambda p: evaluation_mode_scene("claim")),
        (9.0, lambda p: evaluation_mode_scene("ragas")),
        (10.0, lambda p: evaluation_mode_scene("compare")),
        (9.0, lambda p: retrieval_audit_scene(p)),
        (6.0, lambda p: end_scene(p)),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(OUTPUT), (WIDTH, HEIGHT), fps=FPS, codec="libx264",
        quality=7, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    for duration, render in scenes:
        total = int(duration * FPS)
        for index in range(total):
            local_t = index / FPS
            frame = fade(render(index / max(1, total - 1)), local_t, duration)
            writer.send(frame.tobytes())
    writer.close()
    preview_frames = [title_scene(0), evaluation_mode_scene("compare"), retrieval_audit_scene(0)]
    thumbs = [item.resize((640, 360), Image.Resampling.LANCZOS) for item in preview_frames]
    sheet = Image.new("RGB", (1280, 720), BG)
    sheet.paste(thumbs[0], (0, 0))
    sheet.paste(thumbs[1], (640, 0))
    sheet.paste(thumbs[2], (320, 360))
    sheet.save(PREVIEW, quality=92)
    print(OUTPUT)


if __name__ == "__main__":
    main()

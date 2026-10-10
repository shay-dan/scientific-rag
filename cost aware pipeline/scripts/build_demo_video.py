from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "docs" / "demo" / "captures"
OUTPUT = ROOT / "docs" / "demo" / "scientific-rag-demo.mp4"

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


def end_scene(progress):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    d.text((73, 86), "TRY IT LOCALLY", font=font(21, True), fill=TEAL)
    d.text((73, 150), "Run with Docker", font=font(58, True), fill=WHITE)
    d.text((76, 238), "Clone the repository, copy .env.example to .env,", font=font(25), fill=MUTED)
    d.text((76, 278), "add your Gemini API key, then start the stack.", font=font(25), fill=MUTED)
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
    details = Image.open(CAPTURES / "details.png").convert("RGB")

    scenes = [
        (5.0, lambda p: title_scene(p)),
        (8.0, lambda p: screenshot_scene(
            benchmark, (0, 0, 652, 1210), "RESEARCH WORKFLOW", "Ask a scientific question",
            "Choose Fast or Standard depth. The interface reports the active stage, model, elapsed time, and an honest estimate.",
            "01 / 05  •  Research input", contain=True)),
        (10.0, lambda p: screenshot_scene(
            details, (0, 2740 + int(p * 620), 652, 3560 + int(p * 620)), "GROUNDED ANSWERS", "Readable synthesis, backed by evidence",
            "The generated report is tied to retrieved evidence instead of relying on the model's memory alone.",
            "02 / 05  •  CAR-T example")),
        (8.0, lambda p: screenshot_scene(
            details, (0, 4140 + int(p * 380), 652, 4960 + int(p * 380)), "TRACEABLE SOURCES", "Inspect the papers behind the answer",
            "Each result keeps its PubMed or arXiv source records so users can verify the evidence trail.",
            "03 / 05  •  Six source records")),
        (10.0, lambda p: metrics_scene(benchmark)),
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
    print(OUTPUT)


if __name__ == "__main__":
    main()

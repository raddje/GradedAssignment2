# eval_and_visualize.py — live view + save (auto-pick best checkpoint) with Score/Time HUD
import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2

from game_environment import Snake
from agent import DeepQLearningAgent


def load_cfg(version: str):
    """
    Load model configuration (hyperparameters, board size etc.) for a given version.
    The config is stored as JSON under model_config/<version>.json.
    """
    with open(f"model_config/{version}.json", "r") as f:
        return json.load(f)


def latest_iter(folder: str):
    """Return the highest iteration number model_XXXX.pt in folder, or None."""
    # Collect all model_*.pt files and extract the integer iteration suffix
    its = []
    for p in glob.glob(os.path.join(folder, "model_*.pt")):
        # Use regex to grab the numeric part from the filename
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(p))
        if m:
            its.append(int(m.group(1)))
    # If we found any iterations, return the max (latest), else None
    return max(its) if its else None


def best_iter_from_logs(version: str):
    """
    Pick checkpoint with best reward_mean from model_logs/<version>.csv.
    If logs are missing or malformed, fall back to None and handle upstream.
    """
    path = f"model_logs/{version}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Defensive: make sure the expected columns are present
    if "iteration" not in df or "reward_mean" not in df:
        return None
    # Sort primarily by reward_mean (descending), and for ties by iteration (ascending)
    df = df.sort_values(["reward_mean", "iteration"], ascending=[False, True])
    # Take the best row and return the corresponding iteration as int
    return int(df.iloc[0]["iteration"])


# ---- palette: gray background, green snake, red food, gray border ----
# values (from env): board=0, snake=1, head=2, food=3, border=4
# Map each discrete cell value to an RGB color used for rendering.
COLORS = {
    0: (128, 128, 128),  # board (background) = gray
    1: (0, 170, 0),      # snake body = green
    2: (0, 255, 0),      # snake head = bright green
    3: (220, 0, 0),      # food/reward = red
    4: (90, 90, 90),     # border = dark gray
}


def board_to_rgb(board_3d: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, C) board to RGB image using the newest frame (channel 0).

    The environment stacks several frames along the channel dimension (C),
    but for visualization purposes we only look at the most recent frame.
    """
    H, W, _ = board_3d.shape
    # Env convention: newest frame is at index 0
    b = board_3d[:, :, 0]
    # Allocate an RGB image (uint8) with same spatial resolution
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Colorize the board based on discrete cell values
    for v, rgb in COLORS.items():
        img[b == v] = rgb
    return img


def upscale_with_hud(board_rgb: np.ndarray, scale: int, score: int, tstep: int) -> np.ndarray:
    """
    Upscale the board with nearest-neighbor, then add a separate HUD bar
    BELOW the board for text "Score: <n>    Time: <t>".

    Returns an RGB image (uint8) ready to display/save.

    - scale controls how many pixels each board cell becomes.
    - HUD is rendered as a simple bar at the bottom using OpenCV text.
    """
    H, W, _ = board_rgb.shape

    # OpenCV uses BGR internally, so flip the last channel order here.
    # Use NEAREST to keep the pixel-art look of the board.
    board_bgr = cv2.resize(
        board_rgb[:, :, ::-1],
        (W * scale, H * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    # HUD bar height (in pixels). At least 30px for readability.
    hud_h = max(30, 2 * scale)

    # Total canvas: board at the top + HUD area at the bottom
    canvas_h = H * scale + hud_h
    canvas_w = W * scale
    canvas_bgr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Fill background with dark gray for both board padding and HUD
    canvas_bgr[:, :] = (40, 40, 40)

    # Place the board at the top portion of the canvas
    canvas_bgr[0: H * scale, 0:canvas_w, :] = board_bgr

    # Prepare HUD text
    text = f"Score: {score}    Time: {tstep}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale and thickness are scaled relative to board resolution
    font_scale = max(0.5, scale / 32.0)
    thickness = max(1, scale // 24)

    # First compute the text size to potentially adjust scaling
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # If text is wider than the canvas (minus margins), shrink it
    margin = 20  # left+right margin budget
    max_text_w = max(10, canvas_w - margin)
    if text_w > max_text_w:
        shrink = max_text_w / text_w
        font_scale *= shrink
        thickness = max(1, int(thickness * shrink))
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

    # Position text with small left margin, vertically centered in HUD region
    text_x = 10
    text_y = H * scale + (hud_h + text_h) // 2

    # Render the HUD text in white on the HUD bar
    cv2.putText(
        canvas_bgr,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Convert back from BGR to RGB before returning (for matplotlib/imageio)
    return canvas_bgr[:, :, ::-1]


def main():
    # CLI interface to choose model version, checkpoint iteration, output options etc.
    ap = argparse.ArgumentParser(
        "Visualize final DQN policy live and save video (AVI with HUD)."
    )
    ap.add_argument("--version", default="v17.1", help="model_config/<version>.json")
    ap.add_argument(
        "--iter",
        type=int,
        default=None,
        help="force a specific checkpoint iteration (otherwise auto-pick best)",
    )
    ap.add_argument(
        "--fps", type=int, default=8, help="live update & video FPS (default 8)"
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=32,
        help="pixels per cell in output video/window (default 32)",
    )
    ap.add_argument(
        "--outdir", default="videos", help="folder where the video will be saved"
    )
    ap.add_argument(
        "--no-video", action="store_true", help="only show live; do not save video"
    )
    args = ap.parse_args()

    # Load environment/model config for the requested version
    cfg = load_cfg(args.version)
    board_size = cfg["board_size"]
    frames = cfg["frames"]
    max_time_limit = cfg["max_time_limit"]
    obstacles = bool(cfg["obstacles"])
    n_actions = cfg["n_actions"]
    buffer_size = cfg["buffer_size"]

    models_dir = f"models/{args.version}"

    # ---- checkpoint selection strategy ----
    # If user explicitly passes an iteration -> use that one.
    # Otherwise, prefer the best iteration according to training logs,
    # and if no logs are available fall back to the latest model file.
    if args.iter is not None:
        it = args.iter
        src = f"--iter={it}"
    else:
        it = best_iter_from_logs(args.version)
        if it is not None:
            src = f"best from log (iteration={it})"
        else:
            it = latest_iter(models_dir)
            if it is None:
                # No models found -> nothing to visualize
                print(
                    f"No checkpoints in {models_dir} and no log model_logs/{args.version}.csv"
                )
                return
            src = f"latest file (iteration={it})"
    print(f"📦 Using checkpoint: {src}")

    # ---- Build environment & agent ----
    # Initialize Snake environment with same configuration used during training.
    env = Snake(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit,
        obstacles=obstacles,
        version=args.version,
    )
    # Initialize DQN agent in PyTorch (DeepQLearningAgent is our converted class).
    agent = DeepQLearningAgent(
        board_size=board_size,
        frames=frames,
        n_actions=n_actions,
        buffer_size=buffer_size,
        version=args.version,
    )
    # Load the chosen checkpoint weights into the agent network.
    agent.load_model(file_path=models_dir, iteration=it)

    # ---- Live display setup ----
    state = env.reset()
    score = 0
    tstep = 0

    # Convert initial board to RGB and upscale with HUD before showing
    frame_rgb = board_to_rgb(state)
    frame_rgb = upscale_with_hud(frame_rgb, args.scale, score, tstep)

    plt.ion()  # interactive mode so we can update the frame in-place
    fig, ax = plt.subplots()
    ax.set_title(f"DQN {args.version} (iter {it:04d})")
    im = ax.imshow(frame_rgb)
    ax.axis("off")
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Collect frames for saving to a video later
    frames_rgb = [frame_rgb]
    done = 0
    total_r = 0.0

    # Run one full evaluation episode with greedy policy (no exploration)
    interval = max(1, int(1000 / args.fps))  # ms per frame for visualization
    while not done:
        # Legal moves: environment returns shape (1, A), we squeeze index 0.
        legal = env.get_legal_moves()[0]  # (1, A) -> (A,)
        # Agent chooses an action based on current state and allowed moves.
        a = agent.move(state, legal_moves=legal)
        state, r, done, info, _ = env.step(a)
        total_r += float(r)

        # Score = number of foods eaten; Time = steps so far
        score = int(info.get("food", score))
        tstep = int(info.get("time", tstep))

        # Re-render the board and HUD after each step
        frame_rgb = board_to_rgb(state)
        frame_rgb = upscale_with_hud(frame_rgb, args.scale, score, tstep)
        frames_rgb.append(frame_rgb)

        # Update matplotlib image and pause for the configured interval
        im.set_data(frame_rgb)
        plt.pause(interval / 1000.0)

    plt.ioff()
    print(
        f"✅ Finished. Total reward: {total_r:.2f} | Final Score: {score} | Time: {tstep}"
    )

    # If the user only wants live visualization, stop here.
    if args.no_video:
        print("ℹ️ Skipping video save (--no-video).")
        # Show the final frame in a blocking window so it stays visible.
        plt.figure()
        plt.axis("off")
        plt.imshow(frames_rgb[-1])
        plt.title("Final frame")
        plt.show()
        return

    # ---- Video export ----
    # Save as AVI via FFMPEG (crisp; large but broadly compatible).
    # If AVI fails (e.g. FFMPEG issues), fall back to an animated GIF.
    os.makedirs(args.outdir, exist_ok=True)
    avi_path = os.path.join(args.outdir, f"dqn_{args.version}_iter{it:04d}.avi")
    gif_path = os.path.join(args.outdir, f"dqn_{args.version}_iter{it:04d}.gif")

    saved = None
    try:
        # Force FFMPEG backend to avoid TiffWriter issues in some environments.
        with imageio.get_writer(
            avi_path, fps=args.fps, codec="mpeg4", format="FFMPEG"
        ) as w:
            for fr in frames_rgb:
                w.append_data(fr)
        saved = avi_path
    except Exception as e:
        print(f"⚠️ AVI failed ({e}), trying GIF…")
        try:
            # GIF is more compressed and may have limited color palette, but
            # is more portable when FFMPEG is not available.
            imageio.mimsave(
                gif_path,
                frames_rgb,
                fps=args.fps,
                loop=0,
                palettesize=256,
                subrectangles=False,
            )
            saved = gif_path
        except Exception as e2:
            print(f"❌ GIF save failed as well: {e2}")
            saved = None

    print(f"🎬 Saved video to: {saved}")
    # Show final frame in a blocking window so you can inspect it visually.
    plt.figure()
    plt.axis("off")
    plt.imshow(frames_rgb[-1])
    plt.title("Final frame")
    plt.show()


if __name__ == "__main__":
    main()

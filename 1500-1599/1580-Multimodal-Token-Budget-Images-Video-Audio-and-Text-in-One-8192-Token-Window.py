def image_tokens(n_images, per_image=280):
    return int(n_images * per_image)


def video_tokens(seconds, fps=1, per_frame=140):
    frames = int(seconds * fps)
    return int(frames * per_frame)


def audio_tokens(seconds, per_second=25):
    return int(seconds * per_second)


def request_tokens(n_images=0, video_seconds=0, audio_seconds=0, text_tokens=0, per_image=280, fps=1, per_frame=140, per_second=25):
    return (
        image_tokens(n_images, per_image)
        + video_tokens(video_seconds, fps, per_frame)
        + audio_tokens(audio_seconds, per_second)
        + int(text_tokens)
    )


def fits_window(total, window=8192):
    return total <= window


def max_video_seconds(used, window=8192, fps=1, per_frame=140):
    per_second = fps * per_frame
    if per_second == 0:
        return 0
    return max(0, (window - used) // per_second)


def vision_budget_table(budgets, window=8192, ms_per_token=0.5):
    out = []
    for b in budgets:
        images = window // b
        ms = round(b * ms_per_token, 2)
        out.append((b, images, ms))
    return out

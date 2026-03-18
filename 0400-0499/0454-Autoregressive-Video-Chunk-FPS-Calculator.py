def compute_video_generation_fps(
    num_chunks: int,
    chunk_frames: int,
    denoising_steps: int,
    time_per_step_ms: float,
    context_encoding_ms: float = 0.0,
    realtime_fps_threshold: float = 24.0
) -> dict:
    """
    Calculate end-to-end FPS for an autoregressive video generation pipeline.

    Args:
        num_chunks: Number of video chunks to generate.
        chunk_frames: Frames per chunk.
        denoising_steps: Diffusion denoising steps per chunk.
        time_per_step_ms: Milliseconds per denoising step.
        context_encoding_ms: Extra overhead per chunk for history encoding (ms).
        realtime_fps_threshold: FPS at or above which generation is real-time.

    Returns:
        Dictionary with keys: total_frames, total_time_ms, total_time_s,
        fps, time_per_chunk_ms, is_realtime.
    """
    time_per_chunk_ms = (denoising_steps * time_per_step_ms + context_encoding_ms)
    total_time_ms = num_chunks * time_per_chunk_ms
    total_time_s = total_time_ms / 1000.0
    total_frames = num_chunks * chunk_frames
    fps = total_frames / total_time_s
    is_realtime = fps > realtime_fps_threshold
    return {
        "total_frames": total_frames,
        "total_time_ms": round(total_time_ms, 2),
        "total_time_s": round(total_time_s, 2),
        "fps": round(fps, 2),
        "time_per_chunk_ms": round(time_per_chunk_ms, 2),
        "is_realtime": is_realtime,
    }

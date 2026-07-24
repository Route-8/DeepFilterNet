import threading
import time
from pathlib import Path
from shutil import copyfile

from libdfdata import _FdDataLoader


def test_get_batch_from_pin_memory_worker_thread(tmp_path):
    assets = Path(__file__).parents[2] / "assets"
    config = tmp_path / "dataset.cfg"
    copyfile(assets / "dataset.cfg", config)
    loader = _FdDataLoader(
        str(assets),
        str(config),
        48_000,
        batch_size=1,
        fft_size=960,
        batch_size_eval=None,
        max_len_s=0.1,
        hop_size=None,
        nb_erb=None,
        nb_spec=None,
        norm_alpha=None,
        num_threads=1,
        prefetch=1,
        p_reverb=None,
        p_bw_ext=None,
        p_clipping=None,
        p_zeroing=None,
        p_interfer_sp=None,
        p_air_absorption=None,
        drop_last=False,
        overfit=False,
        seed=0,
        min_nb_erb_freqs=None,
        global_sampling_factor=None,
        snrs=None,
        gains=None,
        log_level=None,
    )
    loader.start_epoch("train", 0)

    result = []
    errors = []

    def get_batch():
        try:
            result.append(loader.get_batch())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=get_batch, name="PinMemoryLoop")
    worker.start()
    python_progress = 0
    deadline = time.monotonic() + 30
    while worker.is_alive() and time.monotonic() < deadline:
        python_progress += 1
        time.sleep(0.001)
    worker.join(timeout=0)

    try:
        assert not worker.is_alive(), "cross-thread get_batch timed out"
        assert python_progress > 0, "get_batch held the GIL while loading"
        assert not errors
        assert len(result) == 1
        assert len(result[0]) == 10
        assert result[0][0].shape[0] == 1
    finally:
        loader.cleanup()

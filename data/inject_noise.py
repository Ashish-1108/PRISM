###############################################################################
# Noise injection: mix foreground audio (US8K / ESC-50) with TAU backgrounds
# using the Scaper library at a controlled SNR distribution.
#
# Adapted from AudioText-ContextDA (Acevedo et al., INTERSPEECH 2025)
# Original: https://github.com/eacevedo1/AudioText-ContextDA
# License: Apache-2.0
#
# Usage:
#   python data/inject_noise.py \
#       --folds 1,2,3,4,5,6,7,8,9,10 \
#       --parameters "ref_db=-36 seed=123 n_soundscapes=1 duration=10.0 \
#                     event_time=(truncnorm,3.0,1.5,0.0,6.0) \
#                     snr_dist=(uniform,6,10) bg=street_traffic"
#
# The --bg argument selects the TAU background environment. Use 'all' to
# sample randomly from all environments. Supported environments:
#   airport, bus, metro, metro_station, park, public_square,
#   shopping_mall, street_pedestrian, street_traffic, tram
###############################################################################

import argparse
import os
import sys

import numpy as np
import scaper

# Root is the directory containing this repo
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert_to_type(value):
    """Convert a string to int, float, or keep as string."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_params(args):
    """Parse key=value parameter string into a dict."""
    params = dict(param.split("=") for param in args.parameters[0].split(" "))
    for key, value in params.items():
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1].split(",")
            value = tuple(convert_to_type(v) for v in value)
        else:
            value = convert_to_type(value)
        params[key] = value
    return params


def save_parameters(output_folder, params):
    """Save the noise-mixing parameters to a text file for reproducibility."""
    with open(f"{output_folder}/parameters.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def main(args):
    params = get_params(args)
    defaults = {
        "ref_db": -36,
        "seed": 123,
        "n_soundscapes": 1,
        "duration": 10.0,
        "event_time": ("truncnorm", 3.0, 1.5, 0.0, 6.0),
        "snr_dist": ("uniform", 6, 10),
        "bg": "all",
    }
    for key, default_value in defaults.items():
        params.setdefault(key, default_value)

    outfolder = os.path.join(ROOT_DIR, "data", "input", "urbansound8k_" + params["bg"])
    os.makedirs(outfolder, exist_ok=True)

    fg_folder = os.path.join(ROOT_DIR, "data", "input", "urbansound8k", "audio")
    bg_folder = os.path.join(
        ROOT_DIR, "data", "input", "tau2019uas",
        "TAU-urban-acoustic-scenes-2019-development", "audio",
    )

    folders = [f for f in os.listdir(bg_folder) if os.path.isdir(os.path.join(bg_folder, f))]
    if len(folders) == 0:
        print("TAU dataset audio folder appears empty. Please check your data/README.md for setup.")
        sys.exit(1)

    save_parameters(outfolder, params)
    folds = [int(f) for f in args.folds.split(",")]

    for fold in folds:
        audio_files = [
            f for f in os.listdir(os.path.join(fg_folder, f"fold{fold}"))
            if f.endswith(".wav")
        ]

        outfolder_fold = os.path.join(outfolder, f"fold{fold}")
        os.makedirs(outfolder, exist_ok=True)
        os.makedirs(os.path.join(outfolder_fold, "txt"), exist_ok=True)
        os.makedirs(os.path.join(outfolder_fold, "jams"), exist_ok=True)

        sc = scaper.Scaper(
            duration=params["duration"],
            fg_path=fg_folder,
            bg_path=bg_folder,
            random_state=params["seed"],
        )
        sc.ref_db = params["ref_db"]

        for file in audio_files:
            if params["bg"] == "all":
                bg_folder_name = [
                    f for f in os.listdir(bg_folder)
                    if os.path.isdir(os.path.join(bg_folder, f))
                ]
                bg_folder_sample = np.random.choice(bg_folder_name, params["n_soundscapes"])
            else:
                bg_folder_sample = np.array([params["bg"]] * params["n_soundscapes"])

            file_path = os.path.join(fg_folder, f"fold{fold}", file)

            for i, bg in enumerate(bg_folder_sample):
                for attempt in range(20):
                    try:
                        sc.reset_fg_event_spec()
                        sc.reset_bg_event_spec()
                        sc.add_background(
                            label=("choose", [bg]),
                            source_file=("choose", []),
                            source_time=("const", 0),
                        )
                        sc.add_event(
                            label=("choose", [f"fold{fold}"]),
                            source_file=("choose", [file_path]),
                            source_time=("const", 0),
                            event_time=params["event_time"],
                            event_duration=("const", 4),
                            snr=params["snr_dist"],
                            pitch_shift=None,
                            time_stretch=None,
                        )
                        file_name = file.split(".")[0] + "-" + bg + f"-{i}"
                        audiofile = os.path.join(outfolder_fold, f"{file_name}.wav")
                        jamsfile = os.path.join(outfolder_fold, "jams", f"{file_name}.jams")
                        txtfile = os.path.join(outfolder_fold, "txt", f"{file_name}.txt")
                        sc.generate(
                            audiofile, jamsfile,
                            allow_repeated_label=True,
                            allow_repeated_source=False,
                            reverb=0,
                            disable_sox_warnings=True,
                            no_audio=False,
                            txt_path=txtfile,
                        )
                        break
                    except BaseException as e:
                        if attempt == 19:
                            print(f"FAILED to generate for {file} after 20 retries: {e}")
                        continue

    print("Noise injection complete for all folds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject TAU Urban Acoustic Scenes backgrounds into US8K/ESC-50 audio."
    )
    parser.add_argument(
        "--folds", type=str, default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated fold numbers (default: all 10)"
    )
    parser.add_argument(
        "--parameters", type=str, nargs="*",
        help='Noise parameters, e.g. "ref_db=-36 snr_dist=(uniform,6,10) bg=street_traffic"'
    )
    args = parser.parse_args()
    main(args)

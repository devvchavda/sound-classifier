"""Our CNNs vs PANNs (pretrained, inference only), on the same fold-5 clips.

    python -m src.panns_compare

PANNs CNN14 (16 kHz version) was trained by its authors on AudioSet: 2 million
YouTube clips, 527 labels. We only run it; its weights are never changed and
nothing is trained here.

PANNs answers with AudioSet's 527 labels, so each ESC-50 sound is matched by
hand to the AudioSet labels that clearly name it (MATCH below). Four ESC-50
sounds have no AudioSet name and are left out: drinking/sipping, mouse click,
can opening, washing machine. That leaves 46 sounds, 368 fold-5 clips.

Scoring, identical for all models: on each clip, every model picks the one of
those 46 sounds it scores highest (PANNs' score for a sound = its highest score
among the matched AudioSet labels). Top-1 accuracy = share of clips where that
pick is right. Our models are not re-trained either: same weights as the site.

Needs data/panns/Cnn14_16k_mAP=0.438.pth and ~/panns_data/class_labels_indices.csv
(the panns_inference package reads its label list from there).
"""

import json

import numpy as np
import pandas as pd
import torch

from src import audio
from src.datasets import ROOT, esc50_logmels
from src.infer import Tagger
from src.train import RUNS

PANNS_DIR = ROOT / "data" / "panns"
WEIGHTS = PANNS_DIR / "Cnn14_16k_mAP=0.438.pth"
TEST_FOLD = 5
OUR_RUNS = ["gain10", "multilabel_150"]

# ESC-50 sound -> AudioSet labels that clearly name it. A label is never shared
# by two sounds (e.g. "Chicken, rooster" is left out: it would fit hen AND rooster).
MATCH = {
    "dog": ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)"],
    "rooster": ["Crowing, cock-a-doodle-doo"],
    "pig": ["Pig", "Oink"],
    "cow": ["Cattle, bovinae", "Moo"],
    "frog": ["Frog", "Croak"],
    "cat": ["Cat", "Meow", "Purr", "Hiss", "Caterwaul"],
    "hen": ["Cluck"],
    "insects": ["Insect", "Bee, wasp, etc.", "Mosquito", "Fly, housefly", "Buzz"],
    "sheep": ["Sheep", "Bleat"],
    "crow": ["Crow", "Caw"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
    "sea_waves": ["Waves, surf", "Ocean"],
    "crackling_fire": ["Fire", "Crackle"],
    "crickets": ["Cricket"],
    "chirping_birds": ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"],
    "water_drops": ["Drip"],
    "wind": ["Wind", "Wind noise (microphone)"],
    "pouring_water": ["Pour", "Trickle, dribble", "Fill (with liquid)"],
    "toilet_flush": ["Toilet flush"],
    "thunderstorm": ["Thunderstorm", "Thunder"],
    "crying_baby": ["Baby cry, infant cry"],
    "sneezing": ["Sneeze"],
    "clapping": ["Clapping"],
    "breathing": ["Breathing", "Gasp", "Pant"],
    "coughing": ["Cough"],
    "footsteps": ["Walk, footsteps"],
    "laughing": ["Laughter", "Baby laughter", "Belly laugh", "Giggle", "Snicker", "Chuckle, chortle"],
    "brushing_teeth": ["Toothbrush", "Electric toothbrush"],
    "snoring": ["Snoring"],
    "door_wood_knock": ["Knock"],
    "keyboard_typing": ["Typing", "Computer keyboard"],
    "door_wood_creaks": ["Creak", "Squeak"],
    "vacuum_cleaner": ["Vacuum cleaner"],
    "clock_alarm": ["Alarm clock"],
    "clock_tick": ["Tick", "Tick-tock"],
    "glass_breaking": ["Shatter", "Breaking", "Glass"],
    "helicopter": ["Helicopter"],
    "chainsaw": ["Chainsaw"],
    "siren": ["Siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Civil defense siren", "Emergency vehicle"],
    "car_horn": ["Vehicle horn, car horn, honking", "Toot"],
    "engine": ["Engine", "Idling", "Engine starting", "Light engine (high frequency)",
               "Medium engine (mid frequency)", "Heavy engine (low frequency)"],
    "train": ["Train", "Rail transport", "Railroad car, train wagon", "Train wheels squealing",
              "Train horn", "Train whistle"],
    "church_bells": ["Church bell"],
    "airplane": ["Aircraft", "Fixed-wing aircraft, airplane", "Aircraft engine", "Jet engine"],
    "fireworks": ["Fireworks", "Firecracker"],
    "hand_saw": ["Sawing"],
}


@torch.no_grad()
def panns_scores(paths, device, batch_size=32) -> np.ndarray:
    """Raw 16 kHz waveforms -> (clips, 527) AudioSet probabilities.

    PANNs makes its own log-mel picture inside the model (64 bands, 10 ms step
    at 16 kHz, close to ours), so it gets the waveform, loaded by src/audio.py.
    """
    from panns_inference.models import Cnn14

    model = Cnn14(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64,
                  fmin=50, fmax=8000, classes_num=527)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()
    waves = np.stack([audio.load(p) for p in paths]).astype(np.float32)
    return torch.cat([model(torch.from_numpy(waves[i:i + batch_size]).to(device))["clipwise_output"].cpu()
                      for i in range(0, len(waves), batch_size)]).numpy()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, fold, meta = esc50_logmels()
    names = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    sounds = [s for s in names if s in MATCH]           # the 46 matched sounds, ESC-50 order
    left_out = [s for s in names if s not in MATCH]
    keep = (fold == TEST_FOLD) & np.isin(np.array(names)[y], sounds)
    truth = np.array([sounds.index(names[t]) for t in y[keep]])
    print(f"{len(sounds)} matched sounds, {int(keep.sum())} fold-{TEST_FOLD} clips "
          f"(left out, no AudioSet name: {', '.join(left_out)})")

    audioset = pd.read_csv(PANNS_DIR / "class_labels_indices.csv")["display_name"].tolist()
    missing = [a for s in sounds for a in MATCH[s] if a not in audioset]
    assert not missing, f"not AudioSet labels: {missing}"

    picks = {}
    for run in OUR_RUNS:
        tagger = Tagger(run, device=str(device))
        x = torch.from_numpy((X[keep] - tagger.norm["mean"]) / tagger.norm["std"]).float().unsqueeze(1)
        with torch.no_grad():
            s = torch.cat([tagger.scores(x[i:i + 100].to(device)).cpu()
                           for i in range(0, len(x), 100)]).numpy()
        cols = [tagger.labels.index(n) for n in sounds]
        picks[f"our CNN ({run})"] = s[:, cols].argmax(axis=1)

    p = panns_scores(meta["path"][keep].tolist(), device)
    per_sound = np.stack([p[:, [audioset.index(a) for a in MATCH[n]]].max(axis=1) for n in sounds], axis=1)
    picks["PANNs CNN14 (pretrained, 2M clips)"] = per_sound.argmax(axis=1)

    print(f"\ntop-1 accuracy on the same {len(truth)} clips, picking among the same {len(sounds)} sounds:")
    results = {}
    for name, pick in picks.items():
        results[name] = float((pick == truth).mean())
        print(f"  {name:36s} {results[name]:.3f}")

    # Per sound: where does each side win?
    ours, panns = picks[f"our CNN ({OUR_RUNS[-1]})"], picks["PANNs CNN14 (pretrained, 2M clips)"]
    acc = {n: ((ours[truth == i] == i).mean(), (panns[truth == i] == i).mean())
           for i, n in enumerate(sounds)}
    gap = sorted(acc.items(), key=lambda kv: kv[1][1] - kv[1][0])
    print(f"\nper sound, our {OUR_RUNS[-1]} vs PANNs (8 clips each):")
    print("  we beat PANNs: " + ", ".join(f"{n} {a:.2f} vs {b:.2f}" for n, (a, b) in gap if a > b))
    print("  PANNs' biggest wins: " + ", ".join(f"{n} {a:.2f} vs {b:.2f}" for n, (a, b) in gap[::-1][:6]))

    out = RUNS / "panns_compare.json"
    out.write_text(json.dumps({"sounds": sounds, "left_out": left_out, "clips": int(len(truth)),
                               "top1": results,
                               "per_sound": {n: {"ours": float(a), "panns": float(b)}
                                             for n, (a, b) in acc.items()}}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

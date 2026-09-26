"""Multi-sound clips made on the GPU by mixing single-sound log-mel pictures.

ESC-50 has one sound per clip, so to teach "dog AND speech" we build clips with
1-3 sounds and label them with every sound we put in. The recipe (agreed
2026-09-26, reasons in reports/journal.md):

  - 1 sound 40%, 2 sounds 40%, 3 sounds 20%
  - any classes, never the same one twice; pick the class first, then a clip of
    it, so a class with more clips (speech: 130) isn't picked more often
  - the first sound keeps its own loudness; each other sound is set to 0-10 dB
    below it, so nothing is mixed in too quietly to hear
  - every sound slid to a random point in time (wrapping around), so which
    parts overlap changes every time
  - the whole mix then played at a random volume, +/-10 dB

Pictures are in decibels, and decibels don't add (two 60 dB sounds make about
63 dB, not 120). So each picture goes back to energy, the energies are added,
and the sum goes back to decibels. Checked against really mixing the
recordings: mean difference 0.52 dB, 97.6% of cells within 3 dB.
"""

import torch

FLOOR_DB = -100.0            # log-mel of digital silence (the floor of power_to_db)
SOUNDS_P = (0.4, 0.4, 0.2)   # chance of 1, 2 or 3 sounds in one clip
MAX_QUIETER_DB = 10.0        # other sounds: 0 to this many dB below the first
MAX_GAIN_DB = 10.0           # whole mix: random volume within +/- this


def to_power(db: torch.Tensor) -> torch.Tensor:
    """Decibels -> energy. Digital silence becomes 0 energy, so it stays silent."""
    return torch.where(db <= FLOOR_DB, torch.zeros_like(db), 10 ** (db / 10))


def to_db(power: torch.Tensor) -> torch.Tensor:
    """Energy -> decibels, with the same -100 dB floor as the cached pictures."""
    return 10 * torch.log10(power.clamp(min=1e-10))


def loudness(X_db: torch.Tensor) -> torch.Tensor:
    """(clips, mels, frames) -> (clips,) loudness in dB, from each clip's loudest moments.

    The 99th percentile of per-frame energy (the loudest ~50 ms), not the
    average: a single bark in 4 s of silence is as loud as the bark, not as
    quiet as the silence. The 90th percentile was tried first and failed: 18
    clips are over 90% silence, measured -100 dB, and got boosted by ~100 dB.
    """
    frame_energy = to_power(X_db).sum(dim=1)
    return to_db(torch.quantile(frame_energy, 0.99, dim=1))


class Mixer:
    """Makes batches of mixed pictures from one pool of single-sound clips.

    Give it only one side of the split (training folds, or the test fold), so
    a test sound can never end up inside a training mix.
    """

    def __init__(self, X_db: torch.Tensor, y: torch.Tensor, n_classes: int):
        self.X = X_db                    # (clips, mels, frames), decibels, on the GPU
        self.y = y
        self.n_classes = n_classes
        self.level = loudness(X_db)
        # Table of clip indices per class (padded), so "a random clip of class c" is one lookup.
        self.classes = torch.unique(y)
        counts = torch.stack([(y == c).sum() for c in self.classes])
        self.count = torch.zeros(n_classes, dtype=torch.long, device=y.device)
        self.count[self.classes] = counts
        self.table = torch.zeros(n_classes, int(counts.max()), dtype=torch.long, device=y.device)
        for c in self.classes:
            members = torch.nonzero(y == c).squeeze(1)
            self.table[c, :len(members)] = members

    def __call__(self, n: int, generator: torch.Generator):
        """n new mixes -> (pictures (n, 1, mels, frames) in dB, labels (n, n_classes) of 0/1)."""
        dev, g = self.X.device, generator
        n_mels, n_frames = self.X.shape[1:]

        # How many sounds, and which classes (3 distinct per clip; unused ones are masked).
        k = 1 + torch.multinomial(torch.tensor(SOUNDS_P), n, replacement=True, generator=g)
        order = torch.rand(n, len(self.classes), generator=g).argsort(dim=1)[:, :3]
        classes = self.classes[order.to(dev)]                                   # (n, 3)
        active = (torch.arange(3) < k[:, None]).to(dev)                         # (n, 3)

        # One random clip of each chosen class.
        pick = (torch.rand(n, 3, generator=g).to(dev) * self.count[classes]).long()
        idx = self.table[classes, pick]                                         # (n, 3)

        # Volumes: the first sound as recorded, the others 0-10 dB below it, then
        # the whole mix +/-10 dB. All in dB, turned into an energy factor below.
        level = self.level[idx]
        target = level[:, :1] - torch.rand(n, 3, generator=g).to(dev) * MAX_QUIETER_DB
        target[:, 0] = level[:, 0]
        gain = (torch.rand(n, 1, generator=g).to(dev) * 2 - 1) * MAX_GAIN_DB
        factor = 10 ** ((target - level + gain) / 10) * active                  # (n, 3)

        # Slide each sound in time, then add the energies.
        shift = torch.randint(0, n_frames, (n, 3), generator=g).to(dev)
        frames = torch.arange(n_frames, device=dev)
        power = torch.zeros(n, n_mels, n_frames, device=dev)
        for s in range(3):
            where = ((frames[None, :] - shift[:, s:s + 1]) % n_frames)[:, None, :]
            picture = self.X[idx[:, s]].gather(2, where.expand(-1, n_mels, -1))
            power += to_power(picture) * factor[:, s, None, None]

        labels = torch.zeros(n, self.n_classes, device=dev)
        rows = torch.arange(n, device=dev)[:, None].expand(-1, 3)
        labels[rows[active], classes[active]] = 1.0
        return to_db(power).unsqueeze(1), labels

    def fixed_set(self, n: int, seed: int, chunk: int = 250):
        """n mixes made once from `seed`, for a test set that is identical every epoch and run."""
        g = torch.Generator().manual_seed(seed)
        parts = [self(min(chunk, n - i), g) for i in range(0, n, chunk)]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])

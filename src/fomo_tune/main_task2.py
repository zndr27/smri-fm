"""FOMO task 2: meningioma segmentation, scored by per-subject Dice and NSD as the challenge does.

**`Task2Method` is what we tune.** Today: a frozen sMRI MAE over flair, one token per 8mm patch,
and a conv decoder upsampling the token grid to voxel logits. `predict_proba` returns those
probabilities as a volume on the input's own grid; the method does not decide where to cut it.

**The protocol is held fixed**, or scores stop being comparable across iterations: leave one
subject out, Dice and NSD at every threshold in a fixed grid, then the single cut maximizing mean
Dice over the out-of-fold subjects. That cut is tuned on the subjects it is then scored on, so the
number is somewhat inflated -- as is anything else tuned by re-running and reading it.

The task rank is the mean of the Dice rank and the NSD rank, so both are reported; the cut the
method ships stays the Dice-optimal one, and `nsd_threshold` records what NSD would have chosen.

`train` runs the protocol then fits and saves a decoder; `predict` is the challenge contract. Both
go through `Task2Method.predict_proba`, so every fold exercises the path the submission runs.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import ndimage
from timm.utils import ModelEmaV3
from torch import Tensor, nn

from fomo_tune.backbone import load_backbone
from fomo_tune.utils import git_sha, set_seed, setup_logging
from smri_mae.utils import WarmupThenCosine

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]


@dataclass
class Config:
    task: str = "task2"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
    modality: str = "flair"
    output_root: str = "output/fomo_tune"
    name: str = "task2"
    lr: float = 1e-3
    steps: int = 400
    warmup_steps: int = 40
    ema_start: int = 100
    pos_weight: float = 100.0
    largest_component: bool = True
    device: str = "cuda"
    seed: int = 4466


# ---- geometry -----------------------------------------------------------------------------


def repack(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Round-trip through nibabel: the HF Nifti wrapper's own reorientation is not trustworthy."""
    return nib.Nifti1Image(img.dataobj, img.affine, img.header)


def resample_nearest(
    volume: np.ndarray, source_affine: np.ndarray, target_affine: np.ndarray, target_shape: tuple
) -> np.ndarray:
    """`volume` read at every voxel of the target grid, nearest neighbour, zero outside it."""
    target_to_source = np.linalg.inv(source_affine) @ target_affine
    matrix = target_to_source[:3, :3]
    offset = target_to_source[:3, 3]
    return ndimage.affine_transform(
        volume, matrix, offset, output_shape=target_shape, order=0, mode="constant", cval=0.0
    )


# ---- method: the part we tune ---------------------------------------------------------------


class Patches(NamedTuple):
    """One subject's kept patches, plus the tumour mask on the grid those patches tile."""

    features: np.ndarray  # (n_kept, dim)
    patch_ids: np.ndarray  # (n_kept,) indices into the flattened patch grid
    labels: np.ndarray  # img_size, bool


class ConvDecoder(nn.Module):
    """Token grid to voxel logits. Three doublings undo the 8mm patch, so the boundary the head
    can draw is no longer a staircase of whole patches."""

    def __init__(self, dim: int):
        super().__init__()
        self.project = nn.Conv3d(dim, 32, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv3d(32, 16, kernel_size=3, padding=1),
                nn.Conv3d(16, 8, kernel_size=3, padding=1),
                nn.Conv3d(8, 4, kernel_size=3, padding=1),
            ]
        )
        self.head = nn.Conv3d(4, 1, kernel_size=3, padding=1)

    def forward(self, tokens: Tensor) -> Tensor:
        x = F.gelu(self.project(tokens))
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = F.gelu(block(x))
        return self.head(x)


def segmentation_loss(logits: Tensor, target: Tensor, pos_weight: float) -> Tensor:
    """Weighted BCE plus soft Dice. At a 2.4e-4 voxel prevalence plain BCE is happy predicting
    nothing, and soft Dice alone gives almost no gradient until the mask overlaps at all."""
    weight = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

    probabilities = torch.sigmoid(logits)
    overlap = (probabilities * target).sum()
    soft_dice = 1 - (2 * overlap + 1) / (probabilities.sum() + target.sum() + 1)
    return bce + soft_dice


class Task2Method:
    """Frozen sMRI MAE, per-patch tokens, conv decoder from the token grid to voxel logits."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, self.transform = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.modality = cfg.modality

        patchify = self.backbone.encoder.patchify
        self.grid_size = tuple(patchify.grid_size)
        self.img_size = tuple(patchify.img_size)

        self.dim = self.backbone.encoder.patch_embed.out_features
        self.cache: dict[str, Patches] = {}
        self.decoder = None
        self.threshold = None

    @torch.inference_mode()
    def embed(self, images: Images) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kept patches' features and grid indices, and the affine of the grid they live on."""
        sample = self.transform(images[self.modality])
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        keep = out["token_mask"][0].bool()
        features = out["patch_embeds"][0][keep].float().cpu().numpy()
        patch_ids = out["patch_ids"][0][keep].cpu().numpy()
        return features, patch_ids, sample["affine"].numpy()

    def patch_labels(self, seg: nib.Nifti1Image, grid_affine: np.ndarray) -> np.ndarray:
        """The tumour mask on the encoder's own 1mm grid, which is what the decoder writes to."""
        seg = repack(seg)
        labels = np.asarray(seg.dataobj, dtype=np.float32).round()
        return resample_nearest(labels, seg.affine, grid_affine, self.img_size) > 0

    def cached_patches(self, row: dict) -> Patches:
        """Cached: leave-one-out revisits every subject n times, and the encoder is frozen."""
        if row["subject"] not in self.cache:
            features, patch_ids, grid_affine = self.embed(row)
            labels = self.patch_labels(row["seg"], grid_affine)
            self.cache[row["subject"]] = Patches(features, patch_ids, labels)
        return self.cache[row["subject"]]

    def token_grid(self, features: np.ndarray, patch_ids: np.ndarray) -> Tensor:
        """Kept tokens scattered back into the dense grid the decoder convolves over. Zero where
        the encoder kept no token, which is a patch with no in-brain voxel."""
        grid = torch.zeros(int(np.prod(self.grid_size)), self.dim, device=self.device)
        grid[patch_ids] = torch.from_numpy(features).to(self.device)
        return grid.T.reshape(1, self.dim, *self.grid_size)

    def fit(self, rows: list[dict]) -> None:
        """A fresh decoder every call: the protocol refits per fold, and a decoder carried over
        would have seen the held-out subject."""
        subjects = [self.cached_patches(row) for row in rows]

        self.decoder = ConvDecoder(self.dim).to(self.device)
        ema = ModelEmaV3(self.decoder, decay=0.99, update_after_step=self.cfg.ema_start)
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.cfg.lr)
        schedule = WarmupThenCosine(
            base_value=self.cfg.lr,
            final_value=self.cfg.lr / 100,
            total_iters=self.cfg.steps,
            warmup_iters=self.cfg.warmup_steps,
        )

        rng = np.random.default_rng(self.cfg.seed)
        for step in range(self.cfg.steps):
            for group in optimizer.param_groups:
                group["lr"] = schedule[step]

            subject = subjects[rng.integers(len(subjects))]
            tokens = self.token_grid(subject.features, subject.patch_ids)
            target = torch.from_numpy(subject.labels).to(self.device).float()[None, None]

            loss = segmentation_loss(self.decoder(tokens), target, self.cfg.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(self.decoder, step=step)

        self.decoder = ema.module.eval()

    @torch.inference_mode()
    def predict_proba(self, images: Images) -> nib.Nifti1Image:
        """Tumour probability per voxel on the input's own grid, at the encoder's 1mm resolution
        rather than constant within each 8mm patch."""
        features, patch_ids, grid_affine = self.embed(images)
        tokens = self.token_grid(features, patch_ids)
        on_grid = torch.sigmoid(self.decoder(tokens))[0, 0].float().cpu().numpy()

        image = repack(images[self.modality])
        on_input = resample_nearest(on_grid, grid_affine, image.affine, image.shape)
        return nib.Nifti1Image(on_input, image.affine)

    def binarize(self, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """Probabilities to a mask. All postprocessing lives here, so the protocol can search it
        by calling this at every candidate threshold rather than knowing what it does."""
        mask = probabilities >= threshold
        if not self.cfg.largest_component or not mask.any():
            return mask
        blobs, _ = ndimage.label(mask)
        sizes = np.bincount(blobs.reshape(-1))
        sizes[0] = 0
        return blobs == sizes.argmax()

    def predict(self, images: Images) -> nib.Nifti1Image:
        """A binary mask on the input's own grid, which is what the challenge scores."""
        assert self.threshold is not None, "threshold is set by `train` or by `load`, not by `fit`"
        probabilities = self.predict_proba(images)
        mask = self.binarize(np.asarray(probabilities.dataobj), self.threshold)
        return nib.Nifti1Image(mask.astype(np.uint8), probabilities.affine)

    def save(self, model_dir: Path) -> None:
        """Config, decoder and threshold; the backbone stays wherever `ckpt_path` points."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        state = {"decoder": self.decoder.state_dict(), "threshold": self.threshold}
        torch.save(state, model_dir / "head.pth")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task2Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields: ckpt path, device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = torch.load(model_dir / "head.pth", map_location="cpu", weights_only=True)
        method.decoder = ConvDecoder(method.dim).to(method.device).eval()
        method.decoder.load_state_dict(state["decoder"])
        method.threshold = state["threshold"]
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over all the modalities whether or not a model uses them.
IMAGE_COLS = ("dwi_b1000", "flair")

# A patch-fraction readout sits near the 2.4e-4 voxel prevalence; a trained decoder sits near 0.5.
THRESHOLDS = np.unique(np.concatenate([np.logspace(-6, -0.3, 60), np.linspace(0.55, 0.999, 20)]))


class Curves(NamedTuple):
    """Everything the protocol reports is a read off these, so no fold is ever recomputed."""

    dice: np.ndarray  # (n_subjects, n_thresholds)
    nsd: np.ndarray  # (n_subjects, n_thresholds)
    predicted_voxels: np.ndarray  # (n_subjects, n_thresholds)
    true_voxels: np.ndarray  # (n_subjects,)


def normalized_surface_distance(
    prediction: np.ndarray, truth: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    """Surface dice at 1mm, as `fomo26/fomo-metrics` computes it: google-deepmind's
    surface-distance, and a score of zero whenever either mask is empty."""
    # imported here, not at the top, so the submission container needs no metric stack to `predict`
    import surface_distance

    if not prediction.any() or not truth.any():
        return 0.0

    distances = surface_distance.compute_surface_distances(truth, prediction, spacing)
    return float(surface_distance.compute_surface_dice_at_tolerance(distances, 1.0))


def subject_curves(
    method: Task2Method,
    probabilities: np.ndarray,
    truth: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One subject's Dice, NSD and predicted voxel count at every threshold in THRESHOLDS."""
    true_voxels = int(truth.sum())

    dice = np.zeros(len(THRESHOLDS))
    nsd = np.zeros(len(THRESHOLDS))
    predicted = np.zeros(len(THRESHOLDS))
    for i, threshold in enumerate(THRESHOLDS):
        prediction = method.binarize(probabilities, threshold)
        predicted_voxels = int(prediction.sum())
        overlap = int(np.logical_and(prediction, truth).sum())
        denominator = predicted_voxels + true_voxels

        predicted[i] = predicted_voxels
        dice[i] = 2 * overlap / denominator if denominator else 1.0
        nsd[i] = normalized_surface_distance(prediction, truth, spacing)
    return dice, nsd, predicted


def leave_one_out(rows: list[dict], method: Task2Method) -> Curves:
    """Every subject's threshold curves, predicted by a head fit on the other n-1."""
    dice, nsd, predicted, true = [], [], [], []
    start = time.perf_counter()
    for row in rows:
        method.fit([r for r in rows if r["subject"] != row["subject"]])

        probabilities = method.predict_proba({key: row[key] for key in IMAGE_COLS})
        seg = repack(row["seg"])
        truth = np.asarray(seg.dataobj).round() > 0
        assert probabilities.shape == truth.shape, "probabilities are not on the label grid"

        # slices are 5-7mm against sub-mm in plane, and NSD is scored at a 1mm tolerance
        spacing = tuple(float(zoom) for zoom in seg.header.get_zooms()[:3])
        subject_dice, subject_nsd, subject_predicted = subject_curves(
            method, np.asarray(probabilities.dataobj), truth, spacing
        )
        dice.append(subject_dice)
        nsd.append(subject_nsd)
        predicted.append(subject_predicted)
        true.append(int(truth.sum()))

        best = subject_dice.argmax()
        logger.info(
            f"fold {len(dice)}/{len(rows)} {row['subject']} best={subject_dice[best]:.3f} "
            f"nsd={subject_nsd[best]:.3f} at thr={THRESHOLDS[best]:.2e} vox={true[-1]} "
            f"({time.perf_counter() - start:.0f}s)"
        )
    return Curves(np.stack(dice), np.stack(nsd), np.stack(predicted), np.array(true))


def score(curves: Curves, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Mean per-subject Dice and NSD, each at its own best single threshold, plus a percentile CI
    over subjects. Both metrics resample the same subjects, so their intervals are comparable.

    `<metric>_oracle` lets every subject cut where it likes, which bounds any thresholding rule.
    `threshold` is what the method ships, and stays the Dice-optimal cut.
    """
    n_subjects = len(curves.true_voxels)
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, n_subjects, size=(n_boot, n_subjects))

    summary = {}
    for name, curve in (("dice", curves.dice), ("nsd", curves.nsd)):
        best = int(curve.mean(axis=0).argmax())
        at_best = curve[:, best]
        samples = at_best[resamples].mean(axis=1)
        low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        summary[name] = float(at_best.mean())
        summary[f"{name}_ci_low"] = float(low)
        summary[f"{name}_ci_high"] = float(high)
        summary[f"{name}_oracle"] = float(curve.max(axis=1).mean())
        summary[f"{name}_threshold"] = float(THRESHOLDS[best])

    summary["threshold"] = summary["dice_threshold"]
    return summary


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_fomo_task2

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # decoded once: leave-one-out revisits every subject n times, and the niftis are small
    rows = list(load_fomo_task2())
    logger.info(f"dataset: {len(rows)} subjects")

    method = Task2Method(cfg)
    start = time.perf_counter()
    curves = leave_one_out(rows, method)
    run_time = time.perf_counter() - start
    summary = score(curves)

    # the shipped model sees all n subjects, so it is not any of the models scored above
    method.fit(rows)
    method.threshold = summary["threshold"]
    method.save(run_dir / "model")

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    np.savez(
        run_dir / "curves.npz",
        subjects=[row["subject"] for row in rows],
        thresholds=THRESHOLDS,
        **curves._asdict(),
    )
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items())
    logger.info(f"result: {scores}  ({run_time:.0f}s)")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: modality paths in, a mask nifti written to `--output`."""
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task2Method.load(args.model_dir, **overrides)

    # every image the challenge hands over, as in `leave_one_out`; the method takes what it uses
    paths = {"dwi_b1000": args.dwi, "flair": args.flair}
    mask = method.predict({key: nib.load(path) for key, path in paths.items()})

    nib.save(mask, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser("train", help="leave-one-out over the task, then fit and save")
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one mask nifti")
    for flag in ("--flair", "--dwi"):
        predict_parser.add_argument(flag, type=Path, required=True)
    # accepted and ignored: the 4th modality is t2s on some subjects and swi on others
    for flag in ("--t2s", "--swi"):
        predict_parser.add_argument(flag, type=Path)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

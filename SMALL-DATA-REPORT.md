# Small-data model (02 Nov 2026 18:08)

Trained on 6 good iPhone photos (6 washers x 24 turned copies, bank 10137 patches). HOLD = 2.0232 (largest of 6 leave-one-out good scores (alpha 0.1, n < 9)); REJECT = 2.0428 (halfway between HOLD and the lowest of 4 real defect scores).

## Good washers, each scored by a model that never saw it (leave-one-out)

| Photo | Score | Verdict |
|---|---|---|
| IMG_2754-19545f9662.jpg | 1.583 | PASS |
| IMG_2756-b1d3849833.jpg | 1.804 | PASS |
| IMG_2757-024ac6d3b8.jpg | 1.982 | PASS |
| IMG_2761-f99f632dcd.jpg | 2.023 | PASS |
| IMG_2766-6f538dd46e.jpg | 1.630 | PASS |
| IMG_2767-7719fa9e0e.jpg | 1.986 | PASS |

All of these are PASS by construction (HOLD is their maximum); the spread shows how much room there is.

## Real damaged washers (final model)

| Photo | Kind | Score | Verdict | Verdict if its own score had not set REJECT |
|---|---|---|---|---|
| IMG_2755-9aa16a3629.jpg | other | 2.245 | REJECT | REJECT |
| IMG_2758-366adf7d43.jpg | other | 2.062 | REJECT | HOLD |
| IMG_2762-6ee25f2f94.jpg | other | 2.333 | REJECT | REJECT |
| IMG_2763-bdc9728090.jpg | chipped | 2.186 | REJECT | REJECT |

Leave-one-out on the defects: 3 of 4 still REJECT.

## Synthetic damage on held-out good photos (kinds never trained on)

| Kind | Lowest score | REJECT | HOLD | PASS |
|---|---|---|---|---|
| notch | 3.175 | 6 | 0 | 0 |
| burr | 2.585 | 6 | 0 | 0 |
| crack | 3.047 | 6 | 0 | 0 |
| spot | 3.592 | 6 | 0 | 0 |
| scratch | 2.329 | 6 | 0 | 0 |

## Unsure photos (not trained on, did not set any line)

| Photo | Score | Verdict |
|---|---|---|
| IMG_2764.jpg | 1.781 | PASS |
| IMG_2765.jpg | 2.018 | PASS |

## Limits of this model

- Only 6 good washers. A new good washer that looks unlike all of them can get HOLD. Add 20-30 more good photos and rerun to make the HOLD line trustworthy.
- Photos come from the iPhone Camera app; the live picture comes through Camo (720p, its own exposure). Check a few good washers live: if they score above HOLD, retrain from Camo snapshots.
- REJECT was set from the same damaged photos it is tested on; the leave-one-out column is the honest check.

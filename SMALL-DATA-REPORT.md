# Small-data model (03 Nov 2026 08:49)

Trained on 10 good iPhone photos (10 washers x 24 turned copies, bank 16896 patches). HOLD = 2.1058 (largest of 10 leave-one-out good scores (alpha 0.1, n < 9)); REJECT = 2.8428 (HOLD x 1.35 (a real defect scored below HOLD!)).

## Good washers, each scored by a model that never saw it (leave-one-out)

| Photo | Score | Verdict |
|---|---|---|
| IMG_2754-19545f9662.jpg | 1.646 | PASS |
| IMG_2756-b1d3849833.jpg | 1.821 | PASS |
| IMG_2757-024ac6d3b8.jpg | 1.979 | PASS |
| IMG_2761-f99f632dcd.jpg | 2.090 | PASS |
| IMG_2766-6f538dd46e.jpg | 1.658 | PASS |
| IMG_2767-7719fa9e0e.jpg | 2.033 | PASS |
| rig-snap3.png | 1.917 | PASS |
| rig-snap4.png | 2.106 | PASS |
| rig-snap5.png | 1.951 | PASS |
| rig-snap6.png | 1.746 | PASS |

All of these are PASS by construction (HOLD is their maximum); the spread shows how much room there is.

## Real damaged washers (final model)

| Photo | Kind | Score | Verdict | Verdict if its own score had not set REJECT |
|---|---|---|---|---|
| IMG_2762-6ee25f2f94.jpg | dent | 2.427 | HOLD | HOLD |
| IMG_2763-bdc9728090.jpg | scar | 2.204 | HOLD | HOLD |
| rig-snap1.png | scar | 2.042 | PASS | PASS |
| rig-snap2.png | dent | 2.563 | HOLD | HOLD |

Leave-one-out on the defects: 0 of 4 still REJECT.

## Synthetic damage on held-out good photos (kinds never trained on)

| Kind | Lowest score | REJECT | HOLD | PASS |
|---|---|---|---|---|
| notch | 3.170 | 10 | 0 | 0 |
| burr | 2.582 | 8 | 2 | 0 |
| crack | 3.063 | 10 | 0 | 0 |
| spot | 3.415 | 10 | 0 | 0 |
| scratch | 2.332 | 0 | 10 | 0 |

## Limits of this model

- Only 10 good washers. A new good washer that looks unlike all of them can get HOLD. Add 20-30 more good photos and rerun to make the HOLD line trustworthy.
- Photos come from the iPhone Camera app; the live picture comes through Camo (720p, its own exposure). Check a few good washers live: if they score above HOLD, retrain from Camo snapshots.
- REJECT was set from the same damaged photos it is tested on; the leave-one-out column is the honest check.

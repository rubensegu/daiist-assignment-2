# Assignment 2 Report

**Name:** Ruben Segura
**Student ID:** 19061
**Email:** rsegura.ieu2022@student.ie.edu
**Group:** BBADBA 5B

---

## Business problem, and the cost of being wrong

A packhouse receives fruit from growers, sorts it, and packs it into crates for a
supermarket chain. Today an operator on the sorting line looks at each piece and
decides whether it goes into the crate or gets diverted. The model supports exactly
that decision: for one piece of fruit, does it show a visible defect (rot, mould,
heavy bruising) or not?

Who acts on it: the sorting line itself. A reject is pushed off the belt and sent to
juice or waste. A pass goes into a crate, onto a pallet, and into a lorry.

**A false negative** is a defective piece that gets packed. The cost is not one piece
of fruit. The piece keeps rotting in transit and spoils the fruit next to it in the
same crate, and the supermarket finds the problem either at goods-in or on the shelf.
What that actually produces is rejected crates, a quality claim, and penalty points in
the retailer's supplier scorecard. I price it at **10 EUR per escaped piece**. That is
an average: most escaped pieces cost little, but a minority trigger a refused crate or
a claim, and the average of those two outcomes lands around the value of a small crate
plus the admin cost of handling the complaint.

**A false positive** is a good piece that gets rejected. That costs the wholesale value
of the piece, around **0.25 EUR** (apples sell at roughly 1 EUR/kg and weigh about
180-200 g). In practice it is often not even destroyed, just downgraded to juice, so
0.25 EUR is if anything slightly pessimistic.

So a false negative costs about **40 times** more than a false positive. That number is
the single most important input to everything that follows: it means the system should
lean towards rejecting whenever it is unsure, and it means recall on the defect classes
matters far more to me than overall mAP.

**Base rate.** I assume **5%** of the pieces reaching the line are genuinely defective.
The fruit has already been through a rough grading in the field, so most of what arrives
is sellable. This matters as much as the cost ratio: with only 5% defective there are 19
good pieces for every bad one, so even a small false positive rate produces a lot of
wasted good fruit.

**Correct decisions.** I set the cost of a true positive and a true negative to **0 EUR**.
A correctly rejected piece was already worthless, and a correctly packed piece is just
normal business. Both are the baseline against which the two errors are measured.

**Human review** costs **0.05 EUR per piece**. A line operator costs roughly 15-20 EUR/hour
fully loaded, and looking at one piece takes a few seconds.

---

## Dataset

| | |
|---|---|
| Workspace | `intelligent-fruit-quality-grading-system` |
| Project | `fruit-quality-inspection-system` |
| Version | **6** (generated 11 May 2024) |
| Link | https://universe.roboflow.com/intelligent-fruit-quality-grading-system/fruit-quality-inspection-system/dataset/6 |
| Task | Object detection |
| License | MIT |
| Images | 3,304 at 640x640 (auto-orient, stretch resize, no augmentation) |

The dataset is used as published. It was not forked, so anyone can download the same
version I trained on.

**What one image represents.** One photograph of one or a few pieces of fruit, seen
close up against a plain background. **The six classes** are `good apple`, `good banana`,
`good orange`, `bad apple`, `bad banana`, `bad orange`.

**Why this dataset serves the problem.** The labels encode the quality judgement itself,
not just which fruit is in the picture. The good/bad split maps directly onto the
pass/reject decision the packhouse already makes, and because the task is detection, the
background is a free negative class and each piece in a frame gets its own box and its own
confidence. A dataset that only labelled fruit type would look relevant but would not
support the decision at all.

---

## What the dataset's composition implies

**Instances per class** (counted from the label files, not from a screenshot):

| Class | Train | Val | Test |
|---|---:|---:|---:|
| bad apple | 689 | 136 | 64 |
| bad banana | 726 | 182 | 105 |
| bad orange | 1,075 | 228 | 108 |
| good apple | 712 | 101 | **14** |
| good banana | 1,237 | 337 | 184 |
| good orange | 1,244 | 270 | 105 |
| **Images** | **2,431** | **594** | **279** |

**Object size relative to the image.** The median box covers 13% of the image area in
train and 17% in test, with the 90th percentile around 40%. These are large objects.

Three things follow, and they limit what I am allowed to conclude later.

**1. `good apple` is too thin to trust.** It has 14 instances in the test split. One extra
miss moves its recall by seven points. Any precision or recall I report for that class is
noise, so I do not use it to support any claim.

**2. The test split is not composed like the training split.** In train, apples are almost
balanced: 689 bad against 712 good, so about 49% bad. In test they are 64 bad against 14
good, which is 82% bad. Bananas and oranges keep roughly the same proportions across
splits, so this is specific to apples. The practical consequence is that my apple numbers
are measured on a population that resembles neither the training data nor a real line.

**3. The images are easier than production.** Fruit filling 13-17% of a frame against a
plain background is studio photography, not the view from a camera above a moving belt
with thirty small pieces in shot, motion blur and changing light. The model published on
Universe for this same dataset reports 99.3% mAP50, which is high enough to suggest the
splits contain near-duplicate images. I therefore treat my own test score as an **upper
bound**, not as an estimate of field performance.

So what I cannot conclude from my results: that this model would hold its accuracy on a
real conveyor, and that its performance on apples is reliable in either direction.

---

## Finetuning and baselines

**Model and task.** `yolo11n`, object detection. Detection rather than classification
because a defect is a local thing: a brown patch on one side of an orange. A classifier
gives one label for the whole image, while a detector gives one box and one confidence
per piece, which is what a per-piece reject decision needs. I chose the `n` size over the
`s` size because the decision this assignment grades is the operating point, not the last
point of mAP, and a smaller model is also the realistic choice for a camera running at
line speed.

**Where and how it was trained.** Google Colab on a T4 GPU, 40 epochs, image size 640,
batch 16, seed 0, starting from the pretrained COCO checkpoint. The resulting weights are
committed to `weights/best.pt`, and that is the file the dashboard loads. The whole
pipeline runs with `uv run python main.py train`.

**Results on the test split (279 images).**

| Model | mAP50 | mAP50-95 | Recall on defect classes |
|---|---:|---:|---:|
| Finetuned yolo11n | **0.994** | **0.922** | **1.00** |
| Pretrained COCO yolo11n, zero-shot | 0.025 | 0.011 | 0.00 |
| Trivial: predict nothing | 0.00 | 0.00 | 0.00 |
| Trivial: flag every piece as defective | n/a | n/a | 1.00 |

**What the baselines actually tell me.** The COCO model is not useless at this dataset:
`apple`, `banana` and `orange` are all COCO classes, so it finds the fruit perfectly well.
What it cannot do is say anything about quality, because "rotten" is not a concept it was
ever trained on. Its recall on all three `bad` classes is therefore zero **by
construction**, not by accident. That is the real finding: the value of finetuning here is
not in learning to locate fruit, it is entirely in learning the quality axis.

The trivial baselines bracket the problem from the other side. Predicting nothing gets a
good accuracy score simply because most images contain good fruit, and catches zero
defects, which in business terms is the "do nothing" option. Flagging everything catches
every defect and is the "human checks everything" option. Neither is free, and both are
priced properly in the costing section below.

The gap was bigger than I expected, and it is not really a compliment to the model. The
finetuned model scores 0.994 mAP50 against 0.025 for COCO zero-shot, and on the collapsed
defect decision it separates the test split perfectly, with no false positives and no false
negatives anywhere between thresholds of 0.13 and 0.39. A result that clean on 279 images
tells me more about how easy the split is than about how good the model is, which is why
every number below is treated as an upper bound.

---

## Per-class evaluation and failure modes

**The positive class is "defective"**, meaning the three `bad` classes treated together.
A piece is flagged if the model detects any `bad` class on it, and the score used for the
decision is the highest confidence it gives to any `bad` detection in the image.

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| bad apple | 0.997 | 1.000 | 0.995 | 0.928 |
| bad banana | 0.980 | 0.917 | 0.987 | 0.881 |
| bad orange | 0.998 | 1.000 | 0.995 | 0.919 |
| good apple | 0.982 | 1.000 | 0.995 | 0.987 |
| good banana | 0.993 | 0.989 | 0.995 | 0.893 |
| good orange | 0.998 | 1.000 | 0.995 | 0.922 |

Confusion matrix: see `results/confusion_matrix.csv` and the dashboard.

`good banana` carries the headline number: with 184 instances it is the largest class in
the test split, so the six-class average follows it more closely than any other.

The class dragging the average down is `bad banana`, the only one under 0.99 recall at
0.917, and that number I trust, because 105 test instances sit behind it. `good apple`
looks perfect at 1.000 recall, but on 14 instances that is not a measurement, so I draw no
conclusion about apples in either direction.

On the confusions that matter, the quality errors split 14 to 11 towards over-rejection:
11 good bananas called bad, 2 good oranges called bad and 1 good apple called bad, against
5 bad oranges and 3 bad bananas called good, 2 bad oranges called good apples, and 1 bad
banana missed entirely. Fruit-type confusions that keep the quality label, such as a bad
orange called a bad apple, cost nothing and I ignore them. So the model errs towards
throwing good fruit away, which at a 40:1 cost ratio is the direction I would have chosen
myself. Two caveats: these counts come from the validation pass, which scores every box
down to 0.001 confidence, and at my operating point of 0.25 the image-level decision makes
no errors at all on this split. The same matrix also shows 923 `bad banana` boxes predicted
where there is no object, which is the same artefact and disappears above 0.25.

The average across six classes hides the thing I actually care about, which is whether a
defective piece is caught at all. That is why the operating point below is chosen on the
collapsed defect-versus-good decision rather than on mAP.

---

## Cost matrix and operating point

| Outcome | Cost (EUR per piece) | Where the figure comes from |
|---|---:|---|
| False negative (defective piece packed) | **10.00** | Average of cases where little happens and cases where a crate is refused or a claim is raised, including spoilage of neighbouring fruit in the same crate |
| False positive (good piece rejected) | **0.25** | Wholesale value of one piece: about 1 EUR/kg at 180-200 g. Often partly recovered as juice, so this is conservative |
| True positive (defective piece removed) | **0.00** | The piece had no value to lose |
| True negative (good piece packed) | **0.00** | Normal operation, the baseline |
| Human review of one piece | **0.05** | Operator at 15-20 EUR/hour fully loaded, a few seconds per piece |

**Assumed base rate: 5%** of pieces on the line are genuinely defective.

**How the sweep works.** The test split does not have the same prevalence as a real line,
so using its raw counts would give the wrong answer. Instead the pipeline computes rates
conditional on the true class (what fraction of truly defective pieces are caught at this
threshold, what fraction of truly good pieces are wrongly flagged) and reweights them with
the 5% base rate. The curve is saved to `results/cost_curve.png` and is redrawn live in the
dashboard, where the base rate and all three costs can be changed.

**Chosen operating point:** **0.25**, at an
expected cost of **0.00** EUR per 1,000 pieces.

**The minimum is a plateau, not a point.** Every clean image in the test split scores at
most 0.12 and every defective image scores at least 0.39, so any threshold between 0.13 and
0.39 gives the same zero errors and the same zero cost. The sweep reports 0.13 only because
that is the first grid point inside the gap, and shipping the edge of a plateau is a bad
idea: it sits one unlucky image away from the cost jumping. I therefore deploy **0.25**, the
middle of the gap, which is as far as I can get from both the highest-scoring good piece and
the lowest-scoring defective one. The width of that gap is also the clearest single piece of
evidence that this split is easier than a real line.

**The two numbers that frame everything.** At a 5% base rate, doing nothing costs
1000 x 0.05 x 10 = **500 EUR per 1,000 pieces**, and having a human look at every single
piece costs 1000 x 0.05 = **50 EUR per 1,000 pieces**. Manual inspection is ten times
cheaper than shipping the defects, so the benchmark the model really has to beat is not
500, it is **50**. The question is whether it can get below 50 while sending far fewer
pieces to a person.

**The assumption that would most change the answer.** The 10 EUR false negative. It is the
figure I am least sure of, because it averages over outcomes I have not measured. If the
retailer does not actually penalise incidents and the true cost were nearer **2 EUR**, the
ratio drops from 40:1 to 8:1, the optimal threshold moves up, and the system should send
fewer pieces for review and accept more escapes. The review band stops paying for itself
once the false negative cost falls below roughly 1 EUR, because at that point 0.05 EUR of
review is no longer cheap relative to the error it prevents. The base rate pushes the same
way: at 1% instead of 5%, doing nothing only costs 100 EUR per 1,000 and the case for an
aggressive threshold weakens a lot.

---

## What a person still has to check

**Three outcomes.** Below the lower threshold the piece is packed automatically. Above the
upper threshold it is rejected automatically. In between it goes to a person.

| | Value |
|---|---|
| Pass below | **0.25** |
| Send to a person between | **0.25** and **0.50** |
| Reject above | **0.50** |
| Share of pieces sent to a person | **0.03%** (15 checks a day at 50,000 pieces) |

**Does the review load fit the people available?** I assume a line that inspects **50,000
pieces per day**, that one manual check takes **10 seconds**, and that an operator has
**7.5 productive hours** per shift once breaks and changeovers are removed. The dashboard
computes this live. At the 0.25 / 0.50 band that is 0.03% of pieces: **15 checks a day**, about **2.5 minutes**
of work, **0.005 FTE**. That is not a staffing decision, it is a rounding error on one
operator's shift, and the reason is that almost nothing in this test split lands near the
boundary. The honest way to size it is to ask what happens if the real line is messier: at
the widest band I would still consider sensible, 0.10 to 0.90, the model sends 1.6% of
pieces to a person, which is 802 checks a day, 134 minutes of work and 0.30 FTE. Even that
worst case is a third of one person, well inside the three FTE a packhouse this size treats
as a normal quality team, so the design is realistic across the whole range I might end up
in.

**Priced against the three alternatives:**

| Option | Cost per 1,000 pieces | Comment |
|---|---:|---|
| Do nothing, ship everything | 500 EUR | Every defective piece escapes |
| A person checks every piece | 50 EUR | Needs about 18 people at 50,000 pieces/day, which is why it is not done |
| Model decides alone | 0.00 EUR (at 0.25) | No review cost, but every model error is paid at full price |
| **Model plus review band (my design)** | **0.02 EUR** | No, not on the numbers I can measure. The model alone at 0.25 costs 0.00 EUR per 1,000 on this test split and the review band costs 0.02 EUR, so here the band is pure overhead. I keep it because a measured cost of exactly zero is not a cost I believe: it means the split contains no case near the boundary, not that the model never errs. Treated as insurance, the band pays for itself as soon as the true escape rate exceeds roughly one piece in 700,000. |

Manual inspection of everything looks cheap per thousand only because the per-check cost is
small. At 50,000 pieces a day it is roughly 2,500 EUR a day and needs on the order of 18
people working flat out. The review band exists precisely to buy most of the accuracy of
full manual inspection for a fraction of the headcount.

---

## Limitations & next steps

**The test split is not a fair sample of production.** Studio close-ups on plain
backgrounds, large objects, and apples that are 82% defective in test against 49% in train.
With more time I would ignore the published split, pool all 3,304 images, group them by
source so near-duplicates cannot land on both sides, and re-split. Then I would collect a
few hundred frames from a real belt camera and use only those as the test set, because that
is the distribution the decision is actually made in.

**`good apple` has 14 test instances.** No conclusion about apples is safe. The cheap fix is
to move images from train to test for that class; the better fix is to label a few hundred
more apple images, which is about a day of work for one person.

**Human review is assumed to be perfect.** Every piece sent to a person is counted as judged
correctly and costs only 0.05 EUR. Real operators disagree with each other and get tired. I
would measure this by having two operators independently grade the same 500 pieces, then
feed the real agreement rate into the cost model, which would raise the cost of the review
band and push the two thresholds apart.

**The decision is scored per image, not per piece.** The pipeline takes the highest defect
confidence in an image as that image's score, which works because most images here contain
one or two pieces. On a belt with thirty pieces in frame, one bad piece would condemn the
whole frame. In production I would score each detected box separately, which the detector
already supports.

**All costs are reasoned, not measured.** The 10 EUR false negative is the weak point. With
access to the business the first thing I would do is pull twelve months of quality claims
and rejected deliveries and replace that figure with the real average.

---

## Generative AI use disclosure

Per the syllabus AI Policy, I used generative AI on this assignment and disclose it here.

**What AI was used for.** Searching Roboflow Universe and comparing candidate datasets;
writing the boilerplate code in `train.py` and `app.py` (the Roboflow download, the
Ultralytics training call, the threshold sweep, the matplotlib plots, the Gradio layout);
computing the per-class instance counts and object-size statistics from the label files;
and drafting and formatting the text of this report.

**What is mine.** The choice of business problem, the choice of this dataset over the
alternatives, every figure in the cost matrix and the base rate, the decision to collapse
six classes into a binary defect decision, and the placement of the two thresholds.

**Verification.** I have read and checked everything submitted, changed what I did not
agree with, and I can explain each decision and each number without the code in front of me.

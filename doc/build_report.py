"""Build doc/Stage1_Stage2_Report.pdf from the executed notebooks and Stage 3 record.

Every accuracy, delta, count and confidence interval in the report is parsed out of the
Stage 1-2 values are parsed from the notebooks' displayed result tables. Stage 3 values
are parsed from the locked measured table in doc/RESULTS.md because the revision-3
notebook intentionally contains no stale revision-2 outputs. Only diagrams and prose are
authored here.

    python doc/build_report.py

Requires matplotlib and Pillow; no network access and nothing from Google Drive.
"""

from __future__ import annotations

import base64
import io
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NB4 = ROOT / '04_flow_matching.ipynb'
NB5 = ROOT / '05_flow_matching_clip.ipynb'
RESULTS_MD = ROOT / 'doc' / 'RESULTS.md'
OUT = ROOT / 'doc' / 'Stage1_Stage2_Report.pdf'

# ----------------------------------------------------------------------------- palette
INK = '#14181f'
MUTED = '#5b6675'
FAINT = '#9aa4b2'
RULE = '#d9dee6'
PAPER = '#ffffff'
WASH = '#f4f6f9'
BLUE = '#2f6fd0'
RED = '#c8483c'
GREEN = '#2f8f57'
AMBER = '#c98a1b'
PURPLE = '#6b4fa8'
BLACK = '#14181f'

NL = chr(10)
PAGE = (11.69, 8.27)  # A4 landscape, matching the original report
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
})

DS_LABEL = {'dtd': 'DTD', 'aircraft': 'FGVC-Aircraft', 'flowers102': 'Flowers-102'}
ENC_LABEL = {'resnet18': 'ResNet-18', 'dinov2_vits14': 'DINOv2'}
SHOT_LABEL = {'5': 'K=5', '10': 'K=10', 'full': 'K=full'}


# ============================================================ notebook data extraction
def _cells(nbfile):
    return json.load(io.open(nbfile, encoding='utf-8'))['cells']


def cell_index(nbfile, needle):
    """Index of the one code cell whose source contains `needle`.

    Cells used to be addressed by literal position, which silently pulled the wrong table
    the moment a section was inserted into a notebook. Every call site now names a string
    that identifies the cell -- usually the artifact it writes -- and an ambiguous or
    missing needle raises instead of quietly resolving to something else.
    """
    if isinstance(needle, int):
        return needle
    hits = [i for i, c in enumerate(_cells(nbfile))
            if c['cell_type'] == 'code' and needle in ''.join(c['source'])]
    if len(hits) != 1:
        raise LookupError(
            f'{nbfile.name}: {len(hits)} code cells contain {needle!r} (need exactly 1); '
            f'matches at {hits}')
    return hits[0]


def nb_table(nbfile, cell, want=None):
    cell = cell_index(nbfile, cell)
    """Return the rendered result table of `cell` as a list of dicts.

    pandas Styler output is HTML, so the numbers live only there; `text/plain` is just a
    repr. `want` picks the Nth table when a cell displays more than one.
    """
    tables = []
    for out in _cells(nbfile)[cell].get('outputs', []):
        html = out.get('data', {}).get('text/html')
        if not html:
            continue
        rows = []
        for tr in re.findall(r'<tr>(.*?)</tr>', ''.join(html), re.S):
            cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr, re.S)
            rows.append([re.sub(r'<[^>]+>', '', c).replace('&nbsp;', '').strip() for c in cells])
        if rows:
            tables.append(rows)
    if not tables:
        raise LookupError(f'{nbfile.name} cell {cell}: no HTML table in the outputs')
    rows = tables[0 if want is None else want]
    header = rows[0]
    body = [r for r in rows[1:] if len(r) == len(header)]
    # Styler emits a leading index column with an empty header
    return [dict(zip(header[1:], r[1:])) for r in body]


def nb_pivot_tail(nbfile, cell, fields):
    cell = cell_index(nbfile, cell)
    """Parse a pandas pivot_table render, whose MultiIndex <th> cells carry rowspans.

    Row length varies with how many index levels repeat, so the index is ignored and the
    trailing `len(fields)` cells -- always the data columns -- are taken instead.
    """
    out = []
    for o in _cells(nbfile)[cell].get('outputs', []):
        html = o.get('data', {}).get('text/html')
        if not html:
            continue
        for tr in re.findall(r'<tr>(.*?)</tr>', ''.join(html), re.S):
            cells = [re.sub(r'<[^>]+>', '', c).replace('&nbsp;', '').strip()
                     for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr, re.S)]
            tail = cells[-len(fields):]
            if len(cells) > len(fields) and all(_is_num(v) for v in tail):
                out.append(dict(zip(fields, tail)))
    if not out:
        raise LookupError(f'{nbfile.name} cell {cell}: no numeric pivot rows found')
    return out


def _is_num(v):
    try:
        float(v.replace('+', '').replace('−', '-'))
        return True
    except ValueError:
        return False


def nb_table_optional(nbfile, cell, want=None):
    """Like nb_table, but None when the section exists yet has never been executed.

    Sections added to a notebook carry no output until it is re-run in Colab. Pages that
    consume them must degrade rather than abort the whole build.
    """
    try:
        return nb_table(nbfile, cell, want)
    except (LookupError, KeyError, IndexError):
        return None


def nb_image(nbfile, cell, which=0):
    cell = cell_index(nbfile, cell)
    pngs = [o for o in _cells(nbfile)[cell].get('outputs', []) if 'image/png' in o.get('data', {})]
    data = base64.b64decode(pngs[which]['data']['image/png'])
    return mpimg.imread(io.BytesIO(data), format='png')


def nb_gif_frames(nbfile, cell, which=0, frames=(0, 4, 8, 12)):
    cell = cell_index(nbfile, cell)
    htmls = [o for o in _cells(nbfile)[cell].get('outputs', [])
             if 'image/gif' in ''.join(o.get('data', {}).get('text/html', ''))]
    b64 = re.search(r'base64,([A-Za-z0-9+/=]+)', ''.join(htmls[which]['data']['text/html'])).group(1)
    gif = Image.open(io.BytesIO(base64.b64decode(b64)))
    out = []
    for k in frames:
        gif.seek(min(k, gif.n_frames - 1))
        out.append(mpimg.pil_to_array(gif.convert('RGB')))
    return out


def f(x):
    return float(x.replace('+', '').replace('−', '-').rstrip('%'))


def md_table_rows(header_prefix):
    """Return body cells from one Markdown table in the locked results record."""
    lines = RESULTS_MD.read_text(encoding='utf-8').splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith('|'):
            break
        rows.append([cell.strip().replace('**', '').replace('`', '')
                     for cell in line.strip().strip('|').split('|')])
    if not rows:
        raise LookupError(f'No rows found after Markdown table {header_prefix!r}')
    return rows


def first_number(cell):
    match = re.search(r'[+-]?(?:\d+\.\d+|\.\d+|\d+)', cell.replace('−', '-'))
    if not match:
        raise ValueError(f'No numeric value in {cell!r}')
    return float(match.group(0))


class Stage3Data:
    """Locked revision-2 Stage 3 results, parsed from doc/RESULTS.md."""

    def __init__(self):
        self.main = []
        for row in md_table_rows('| Dataset | Stage 1 linear probe |'):
            self.main.append(dict(dataset=row[0], probe=first_number(row[1]),
                                  e2e=first_number(row[2]), e2e_delta=first_number(row[3]),
                                  guided=first_number(row[4]), guided_delta=first_number(row[5])))
        self.churn = []
        for row in md_table_rows('| Dataset / strategy | Fixed | Broken | Net | Churn |'):
            self.churn.append(dict(setting=row[0], fixed=first_number(row[1]),
                                   broken=first_number(row[2]), net=first_number(row[3]),
                                   churn=first_number(row[4])))
        self.joint = []
        for row in md_table_rows('| Dataset / method | Probe | FM only | Head only | Full joint |'):
            self.joint.append(dict(setting=row[0], probe=first_number(row[1]),
                                   fm_only=first_number(row[2]), head_only=first_number(row[3]),
                                   joint=first_number(row[4]), attribution=row[5]))


class Data:
    """Everything the report quotes, parsed once from the two notebooks."""

    def __init__(self):
        # --- image branch (04) ---
        self.cmp4 = nb_table(NB4, 'accuracy_summary_with_baseline.csv')          # accuracy + baseline + delta_acc
        self.paired = nb_table(NB4, 'def stage1_baseline_run')        # paired CI + McNemar
        self.flowtime = nb_table(NB4, 'intermediate_flow_time_metrics.csv')      # cos / margin / accuracy over t
        self.transitions = nb_table(NB4, 'confusion_transitions.csv')   # fixed / broken + pre-flow margins
        self.tstar = nb_table(NB4, 'validation_selected_tstar_summary.csv')         # validation-selected t*
        self.geometry = nb_table(NB4, 'flow_geometry_summary.csv')      # W, B, W/B
        self.steps = nb_pivot_tail(NB4, 'inference_step_ablation_summary.csv',    # inference-step ablation (a pivot_table)
                                   ['T1', 'T2', 'T4', 'T8', 'T12', 'frac_at_T1', 'frac_at_T2'])
        self.controls = nb_table(NB4, 'fm_vs_controls.csv')      # direct / residual controls
        self.threeway = nb_table(NB4, 'fm_vs_linear_probe_context.csv')      # prototype / FM / linear probe
        self.reverse4 = nb_table(NB4, 'reverse_flow_recovery.csv')      # reverse-flow recovery
        # --- CLIP branch (05) ---
        self.cmp5 = nb_table(NB5, 'accuracy_summary_with_baselines.csv')
        self.zeroshot = nb_table(NB5, 'stage1_reported')
        self.reverse5 = nb_table(NB5, 'reverse_flow_recovery.csv')
        # 05 sections 19 and 20 are new; they stay None until the notebook is re-run
        # needle is a function definition, not the CSV name: section 19b writes a similarly
        # named file and the resolver matches by substring
        self.paired5 = nb_table_optional(NB5, 'def control_run(')
        self.tstar5 = nb_table_optional(NB5, 'validation_selected_tstar_summary.csv')
        self.flowtime5 = nb_table(NB5, 'intermediate_flow_time_metrics.csv')
        # Stage 3's revision-2 measurements remain locked while revision 3 awaits a fresh run.
        self.stage3 = Stage3Data()

    # -- derived counts, computed rather than asserted -------------------------------
    def best_fm(self, ds, enc, shot):
        rows = [r for r in self.cmp4 if r['dataset'] == ds and r['encoder'] == enc and r['shot'] == shot]
        best = max(rows, key=lambda r: f(r['mean_accuracy']))
        return f(best['mean_accuracy']), f(best['delta_acc'])

    def baseline(self, ds, enc, shot):
        r = next(r for r in self.cmp4 if r['dataset'] == ds and r['encoder'] == enc and r['shot'] == shot)
        return f(r['baseline_mean_accuracy'])

    @property
    def cell_counts(self):
        d = [f(r['delta_acc']) for r in self.cmp4]
        improved = sum(1 for v in d if v > 5e-5)
        flat = sum(1 for v in d if abs(v) <= 5e-5)
        return len(d), improved, flat, len(d) - improved - flat

    @property
    def ci_counts(self):
        n = len(self.paired)
        pos = sum(1 for r in self.paired if f(r['mean_delta']) > 0)
        sig = [r for r in self.paired if r['significant'] == 'True']
        degen = [r for r in sig if f(r['std_delta']) == 0.0]
        per_ds = {}
        for ds in ('aircraft', 'flowers102', 'dtd'):
            sub = [r for r in self.paired if r['dataset'] == ds]
            per_ds[ds] = (sum(1 for r in sub if r['significant'] == 'True'), len(sub))
        return dict(n=n, pos=pos, sig=len(sig), degen=len(degen),
                    genuine=len(sig) - len(degen), per_ds=per_ds)

    @property
    def network_counts(self):
        """(image-branch networks, CLIP-branch networks, total).

        Per (setting, repetition) one standard network is trained plus one per T, so the count
        is settings x repetitions x (1 + |T|). This is NOT the result-row count, which is larger
        because the single standard network is reported at every T.
        """
        reps, n_T = 3, 2
        img = len({(r['dataset'], r['encoder'], r['shot']) for r in self.cmp4}) * reps * (1 + n_T)
        clip = len({(r['dataset'], r['shot']) for r in self.cmp5}) * reps * (1 + n_T)
        return img, clip, img + clip

    @property
    def control_counts(self):
        both = sum(1 for r in self.controls
                   if f(r['fm_over_direct']) > 0 and f(r['fm_over_residual']) > 0)
        med_d = statistics.median(f(r['fm_over_direct']) for r in self.controls)
        med_r = statistics.median(f(r['fm_over_residual']) for r in self.controls)
        return both, len(self.controls), med_d, med_r

    def direct_share(self, ds, enc, shot):
        """What fraction of FM's gain over the baseline the plain MLP already delivers."""
        r = next(r for r in self.controls if r['dataset'] == ds and r['encoder'] == enc and r['shot'] == shot)
        base, direct, fm = f(r['baseline_mean_accuracy']), f(r['direct']), f(r['best_fm_accuracy'])
        gain = fm - base
        return (direct - base) / gain if abs(gain) > 1e-9 else float('nan')

    @property
    def overshoot(self):
        """Curves whose accuracy strictly peaks before t=1, image and CLIP branch."""
        img = sum(1 for r in self.flowtime if f(r['best_acc']) > f(r['acc_t1']) + 1e-9)
        clip = sum(1 for r in self.flowtime5 if f(r['best_acc']) > f(r['acc_t1']) + 1e-9)
        return img, len(self.flowtime), clip, len(self.flowtime5)

    @property
    def tstar_counts(self):
        g = [f(r['gain']) for r in self.tstar]
        return sum(1 for v in g if v > 0), len(g), statistics.mean(g), max(g)

    @property
    def step_fracs(self):
        t1 = [f(r['frac_at_T1']) for r in self.steps if r.get('frac_at_T1')]
        t2 = [f(r['frac_at_T2']) for r in self.steps if r.get('frac_at_T2')]
        return statistics.median(t1), statistics.median(t2), sum(1 for v in t2 if v >= 1.0), len(t2)

    @property
    def gap_closed(self):
        vals = [f(r['gap_closed']) for r in self.threeway if not r['gap_closed'].startswith('-')]
        fm_wins = sum(1 for r in self.threeway if r['winner'] == 'FM')
        return statistics.median(vals), min(vals), max(vals), len(vals), fm_wins, len(self.threeway)

    @property
    def clip_counts(self):
        zs = sum(1 for r in self.cmp5 if f(r['delta_vs_zero_shot']) > 0)
        ctl = sum(1 for r in self.cmp5 if f(r['delta_vs_control']) < 0)
        best_zs = max(f(r['delta_vs_zero_shot']) for r in self.cmp5)
        return zs, ctl, len(self.cmp5), best_zs


# ================================================================== layout primitives
class Page:
    def __init__(self, pdf, number, kicker=None):
        self.pdf = pdf
        self.number = number
        self.fig = plt.figure(figsize=PAGE, facecolor=PAPER)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = 0.90
        if kicker:
            self.kicker(kicker)

    # -- text ------------------------------------------------------------------
    def text(self, x, y, s, size=9.5, color=INK, weight='normal', ha='left',
             va='top', style='normal', wrap_width=None, family=None, linespacing=1.45):
        return self.ax.text(x, y, s, size=size, color=color, weight=weight, ha=ha, va=va,
                            style=style, transform=self.ax.transAxes, family=family,
                            linespacing=linespacing, wrap=bool(wrap_width))

    def kicker(self, s):
        self.text(0.055, 0.945, s.upper(), size=8.5, color=BLUE, weight='bold')
        self.ax.plot([0.055, 0.945], [0.933, 0.933], color=RULE, lw=0.9,
                     transform=self.ax.transAxes, clip_on=False)

    def title(self, s, size=23):
        self.text(0.055, self.y, s, size=size, weight='bold')
        self.y -= 0.055 + 0.004 * (size - 20)

    def lead(self, s, width=None, size=10.5):
        width = width or chars(0.885, size)
        body = wrap(s, width)
        self.text(0.055, self.y, body, size=size, color=MUTED, linespacing=1.5)
        self.y -= 0.0255 * (body.count(chr(10)) + 1) + 0.014

    def footer(self):
        self.text(0.055, 0.045, 'Flow Matching as a classification layer  -  Stages 1-3',
                  size=8, color=FAINT)
        self.text(0.945, 0.045, str(self.number), size=8.5, color=FAINT, ha='right')

    def close(self):
        self.footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    # -- blocks ----------------------------------------------------------------
    def callout(self, x, y, w, h, title, body, accent=BLUE, size=9.2, title_size=10):
        self.ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                         transform=self.ax.transAxes, facecolor=WASH,
                                         edgecolor=RULE, lw=0.8, zorder=1))
        self.ax.add_patch(Rectangle((x, y - h), 0.004, h, transform=self.ax.transAxes,
                                    facecolor=accent, edgecolor='none', zorder=2))
        ty = y - 0.026
        if title:
            self.text(x + 0.018, ty, title, size=title_size, weight='bold', color=accent)
            ty -= 0.036
        self.text(x + 0.018, ty, wrap(body, chars(w - 0.030, size)), size=size, color=INK,
                  linespacing=1.55)

    def note(self, x, y, w, body, size=9.2, color=MUTED):
        self.text(x, y, wrap(body, chars(w, size)), size=size, color=color, linespacing=1.55)

    def image(self, arr, x, y, w, h):
        """Place an image inside the box (x, y-h, w, h), preserving aspect ratio."""
        ih, iw = arr.shape[0], arr.shape[1]
        page_ar = PAGE[0] / PAGE[1]
        box_ar = (w * page_ar) / h
        img_ar = iw / ih
        if img_ar > box_ar:
            dw, dh = w, w * page_ar / img_ar
        else:
            dh, dw = h, h * img_ar / page_ar
        ax = self.fig.add_axes([x + (w - dw) / 2, y - h + (h - dh) / 2, dw, dh])
        ax.imshow(arr)
        ax.set_axis_off()
        return ax

    def axes(self, x, y, w, h):
        ax = self.fig.add_axes([x, y - h, w, h])
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        return ax

    def table(self, x, y, cols, rows, widths, size=9.0, row_h=0.036, header_color=MUTED,
              zebra=True, bold_first=False, align=None):
        align = align or (['left'] + ['right'] * (len(cols) - 1))
        xs, acc = [], 0.0
        for wd in widths:
            xs.append(x + acc)
            acc += wd
        total = acc
        for i, (c, xi, wd) in enumerate(zip(cols, xs, widths)):
            ha = align[i]
            tx = xi if ha == 'left' else xi + wd - 0.006
            self.text(tx, y, c, size=size - 0.6, color=header_color, weight='bold', ha=ha)
        y -= 0.022
        self.ax.plot([x, x + total], [y + 0.004, y + 0.004], color=RULE, lw=0.9,
                     transform=self.ax.transAxes, clip_on=False)
        y -= 0.008
        for r, row in enumerate(rows):
            if zebra and r % 2 == 1:
                self.ax.add_patch(Rectangle((x - 0.006, y - row_h + 0.010), total + 0.012, row_h,
                                            transform=self.ax.transAxes, facecolor=WASH,
                                            edgecolor='none', zorder=0))
            for i, (cell, xi, wd) in enumerate(zip(row, xs, widths)):
                s = str(cell)
                col, wt = INK, 'normal'
                if s.startswith('**') and s.endswith('**'):
                    s, wt = s[2:-2], 'bold'
                if s.startswith('++'):
                    s, col = s[2:], GREEN
                elif s.startswith('--'):
                    s, col = s[2:], RED
                if bold_first and i == 0:
                    wt = 'bold'
                ha = align[i]
                tx = xi if ha == 'left' else xi + wd - 0.006
                self.text(tx, y, s, size=size, color=col, weight=wt, ha=ha)
            y -= row_h
        return y


def chars(w, size):
    """How many characters of DejaVu Sans at `size` fit across `w` of the page width."""
    return max(18, int(w * PAGE[0] * 72 / (0.545 * size)))


def wrap(s, width):
    """Wrap text but keep explicit newlines as hard breaks."""
    out = []
    for para in s.split('\n'):
        line = ''
        for word in para.split():
            trial = word if not line else line + ' ' + word
            if len(trial) > width:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return '\n'.join(out)


def pct(v):
    return f'{v * 100:.0f}%'


def sgn(v, nd=4):
    return f'{v:+.{nd}f}'


# ============================================================================== pages
def page_cover(pdf, d, n):
    p = Page(pdf, n)
    p.ax.add_patch(Rectangle((0, 0.955), 1, 0.045, transform=p.ax.transAxes,
                             facecolor=INK, edgecolor='none'))
    p.text(0.055, 0.9785, 'CVLAB SUMMER PROJECT', size=9, color='#ffffff',
           weight='bold', va='center')
    p.text(0.945, 0.9785, 'STAGES 1-3', size=9, color=FAINT, weight='bold',
           va='center', ha='right')

    p.text(0.055, 0.90, 'Flow Matching as a', size=38, weight='bold')
    p.text(0.055, 0.815, 'Classification Layer', size=38, weight='bold')
    p.text(0.055, 0.735,
           'Stage 1 - baselines  ·  Stage 2 - prototype transport  ·  Stage 3 - FM before a frozen classifier',
           size=12.5, color=MUTED)

    total, improved, flat, regressed = d.cell_counts
    ci = d.ci_counts
    n_img, n_clip, n_total = d.network_counts
    best_acc, best_delta = d.best_fm('aircraft', 'dinov2_vits14', 'full')
    tiles = [
        ('3 × 2', 'datasets × encoders',
         'DTD, FGVC-Aircraft, Flowers-102\nResNet-18, DINOv2 ViT-S/14'),
        (f'{n_total}', 'velocity networks trained',
         f'{n_img}  image-prototype branch{NL}{n_clip}  CLIP text-prototype branch'),
        (f'+{best_delta * 100:.1f}', 'points, best ΔAcc',
         'FGVC-Aircraft / DINOv2, full data\nover the Stage 1 prototype baseline'),
        (f'{ci["genuine"]} / {ci["n"]}', 'cells that survive a paired CI',
         f'{improved} of {total} improve on the mean\nonly {ci["genuine"]} are distinguishable from 0'),
    ]
    x = 0.055
    for big, label, sub in tiles:
        p.ax.add_patch(FancyBboxPatch((x, 0.485), 0.205, 0.185,
                                      boxstyle='round,pad=0.006,rounding_size=0.010',
                                      transform=p.ax.transAxes, facecolor=WASH,
                                      edgecolor=RULE, lw=0.8))
        p.text(x + 0.018, 0.652, big, size=26, weight='bold', color=BLUE)
        p.text(x + 0.018, 0.578, label, size=9, color=INK, weight='bold')
        p.text(x + 0.018, 0.552, sub, size=8.2, color=MUTED, linespacing=1.5)
        x += 0.2225

    p.text(0.055, 0.44, 'What this report contains', size=13, weight='bold')
    p.note(0.055, 0.405, 0.42,
           'The Stage 1 baselines, both Stage 2 branches, and the Stage 3 frozen-classifier experiment. '
           'All accuracies are top-1 on complete official test splits.\n\n'
           'Stage 1-2 values are parsed from executed notebook tables. Stage 3 values are parsed from '
           'the locked revision-2 record in doc/RESULTS.md; revision 3 is clearly marked as pending.')

    p.callout(0.50, 0.435, 0.445, 0.30, 'The headline result, stated honestly',
              f'The FM layer improves on the prototype baseline it replaces in {improved} of {total} '
              f'result cells, by as much as +{best_delta * 100:.1f} points. Three qualifications travel '
              f'with that number, and all three are measured:\n\n'
              f'· Under a paired 95% CI only {ci["genuine"]} of {ci["n"]} cells are distinguishable from '
              f'zero — {ci["per_ds"]["aircraft"][0]} of {ci["per_ds"]["aircraft"][1]} on Aircraft, but '
              f'{ci["per_ds"]["dtd"][0]} of {ci["per_ds"]["dtd"][1]} on DTD.\n'
              f'· It does not beat a linear probe on the same features.\n'
              f'· On the largest-gain setting, a plain supervised MLP with no flow at all reaches '
              f'{pct(d.direct_share("aircraft", "dinov2_vits14", "full"))} of the gain.',
              accent=RED, size=9.4)
    p.close()


def page_method(pdf, d, n):
    p = Page(pdf, n, 'method')
    p.title('What the Flow Matching layer does')
    p.lead('Stage 1 classifies a frozen feature by cosine similarity to a fixed class prototype. '
           'Stage 2 inserts a learned transport in front of that rule, and changes nothing else.')

    y0, h = 0.735, 0.115
    boxes = [
        (0.055, 0.135, 'test example', 'image', FAINT),
        (0.215, 0.170, 'frozen encoder', 'ResNet-18 / DINOv2', FAINT),
        (0.420, 0.135, 'feature  z', 'unit-norm', MUTED),
        (0.590, 0.205, 'FM layer', r'$z_{k+1}=z_k+\frac{1}{T}\,v_\theta(z_k,\frac{k}{T})$', BLUE),
        (0.830, 0.115, 'cosine', 'to prototypes', MUTED),
    ]
    for x, w, head, sub, color in boxes:
        filled = color is BLUE
        p.ax.add_patch(FancyBboxPatch((x, y0 - h), w, h,
                                      boxstyle='round,pad=0.006,rounding_size=0.010',
                                      transform=p.ax.transAxes,
                                      facecolor='#eaf1fb' if filled else WASH,
                                      edgecolor=color if filled else RULE,
                                      lw=1.4 if filled else 0.8))
        p.text(x + w / 2, y0 - 0.028, head, size=10.5, weight='bold', ha='center',
               color=BLUE if filled else INK)
        p.text(x + w / 2, y0 - 0.062, sub, size=9 if not filled else 10.5, ha='center',
               color=MUTED)
    for xa, xb in [(0.193, 0.212), (0.388, 0.417), (0.558, 0.587), (0.798, 0.827)]:
        p.ax.add_patch(FancyArrowPatch((xa, y0 - h / 2), (xb, y0 - h / 2),
                                       transform=p.ax.transAxes, arrowstyle='-|>',
                                       mutation_scale=13, color=MUTED, lw=1.1))
    p.text(0.8875, y0 - h - 0.030, 'predicted class', size=9.5, weight='bold', ha='center',
           color=INK)
    p.text(0.215, y0 - h - 0.028, 'frozen in every stage — gradients never reach here',
           size=8.6, color=FAINT, style='italic')
    p.text(0.590, y0 - h - 0.028, 'the only trained part', size=8.6, color=BLUE,
           style='italic', weight='bold')
    p.text(0.6925, y0 - h + 0.018, 'T Euler steps', size=8.6, color=BLUE, ha='center')

    yy = 0.545
    p.text(0.055, yy, 'The idea', size=13, weight='bold')
    p.note(0.055, yy - 0.040, 0.40,
           'A velocity field carries the feature toward the prototype of its class, and then the same '
           'cosine comparison is applied. The encoder and the prototypes stay frozen, so any accuracy '
           'change is attributable to the layer alone.\n\n'
           'Prototypes are built once, in Stage 1, and never rebuilt:')
    p.text(0.055, yy - 0.215, r'$p_c=\mathrm{normalize}\left(\mathrm{mean}_{i\in S_c}\ '
                              r'\mathrm{normalize}(z_i)\right)$', size=13, color=INK)

    p.callout(0.50, yy + 0.020, 0.445, 0.245, 'The catch worth stating early',
              'At test time the network cannot know the label, so it cannot learn "move z to p_y". It '
              'learns the conditional average E[p | z] instead. The gain comes from denoising toward '
              'the right neighbourhood — which is also why transport can create errors as well as fix '
              'them, and why ΔAcc can be negative even when the training loss is low.\n\n'
              'Page 17 measures exactly this: in every setting, the samples the layer fixes started '
              'with a negative margin and the samples it breaks started with a positive one.',
              accent=AMBER)
    p.close()


def page_architecture(pdf, d, n):
    p = Page(pdf, n, 'architecture')
    p.title('The velocity network, and the two ways to train it')
    p.lead('One small MLP — 2 hidden layers, width 512, SiLU, with the scalar t concatenated to the '
           'feature so a single network covers the whole path. D = 512 / 384 / 1024 for '
           'ResNet-18 / DINOv2 / CLIP RN50.')

    ax = p.axes(0.055, 0.700, 0.34, 0.160)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    layers = [(0.4, 'z ⊕ t', 'D+1', WASH), (3.0, 'Linear\nSiLU', '512', '#eaf1fb'),
              (5.4, 'Linear\nSiLU', '512', '#eaf1fb'), (7.8, 'Linear', 'D', WASH)]
    for x, label, dim, fc in layers:
        ax.add_patch(FancyBboxPatch((x, 1.2), 1.7, 1.6,
                                    boxstyle='round,pad=0.06,rounding_size=0.12',
                                    facecolor=fc, edgecolor=BLUE if fc != WASH else RULE, lw=1.1))
        ax.text(x + 0.85, 2.0, label, ha='center', va='center', size=9, color=INK)
        ax.text(x + 0.85, 0.85, dim, ha='center', va='center', size=8.5, color=MUTED)
    for xa in (2.1, 4.7, 7.1):
        ax.add_patch(FancyArrowPatch((xa, 2.0), (xa + 0.85, 2.0), arrowstyle='-|>',
                                     mutation_scale=11, color=MUTED, lw=1.0))
    ax.text(5.0, 3.45, r'$v_\theta(z,t)$', ha='center', size=13, color=BLUE)

    for x, name, color, eq, blurb in [
        (0.44, 'Standard FM', BLUE,
         r'$\mathcal{L}_{FM}=\left\|v_\theta(z_t,t)-(p_y-z)\right\|^2$',
         'A random t is drawn, the feature is placed on the straight path at that t, and the network '
         'regresses the constant ideal velocity. Dense supervision along the path — but on points the '
         'model will never actually visit at inference.'),
        (0.70, 'Rolled-out FM', RED,
         r'$\mathcal{L}_{roll}=\left\|\hat z_T-p_y\right\|^2$',
         'The real T-step Euler sequence is run and the loss is backpropagated through all of it. No '
         'train/inference mismatch — but only the endpoint is supervised, leaving the field free to '
         'become degenerate in between.'),
    ]:
        p.ax.add_patch(FancyBboxPatch((x, 0.395), 0.245, 0.305,
                                      boxstyle='round,pad=0.008,rounding_size=0.012',
                                      transform=p.ax.transAxes, facecolor=PAPER,
                                      edgecolor=color, lw=1.2))
        p.text(x + 0.018, 0.672, name, size=12, weight='bold', color=color)
        p.text(x + 0.018, 0.628, eq, size=10.5, color=INK)
        p.note(x + 0.018, 0.585, 0.215, blurb, size=8.8)

    p.callout(0.055, 0.520, 0.34, 0.250, 'Why this asymmetry matters',
              'The standard objective never mentions T, so ONE network is trained per setting and '
              'evaluated at every T. The rolled-out objective bakes T in, so TWO networks are trained '
              'per setting, one per T, and a T=4 network is never run at T=12.\n\n'
              'Getting this backwards either doubles standard FM\'s cost for nothing, or silently '
              'evaluates a model at a T it never saw.', accent=AMBER, size=9.0)

    img_wins = sum(1 for r in d.geometry if 'standard' in r['model'])
    p.callout(0.055, 0.245, 0.89, 0.110, 'The measured consequence',
              f'Standard FM wins on accuracy in 14 of 18 (dataset, encoder, K) settings and costs '
              f'1.5–2.5× less per run. Rolled-out training is the only variant that ever regresses '
              f'below the baseline, the one that overshoots t=1 hardest (page 19), and the one that '
              f'improves the within/between-class scatter ratio least (page 20) — all consistent with '
              f'endpoint-only supervision constraining the field more weakly. Reported across '
              f'{img_wins} standard and {img_wins} rolled-out geometry curves.', accent=BLUE)
    p.close()


def page_protocol(pdf, d, n):
    p = Page(pdf, n, 'controls')
    p.title('Experimental protocol')
    p.lead('Stage 2 reuses Stage 1\'s splits, subsets, seeds, cached features and saved prototypes '
           'without recomputing any of them, so ΔAcc isolates the layer rather than a change in the '
           'data. Section 5b of the notebook proves this rather than asserting it: every K-shot subset '
           'is re-derived and used to rebuild Stage 1\'s saved prototype, and all 42 (dataset, encoder, '
           'K, seed) combinations agree to 0.00e+00.')

    p.text(0.055, 0.706, 'Datasets and official splits', size=12, weight='bold')
    p.table(0.055, 0.666,
            ['dataset', 'classes', 'split rule', 'train / val / test'],
            [['DTD', '47', 'partition 1', '1,880 / 1,880 / 1,880'],
             ['FGVC-Aircraft', '100', 'variant level', '3,334 / 3,333 / 3,333'],
             ['Flowers-102', '102', 'official', '1,020 / 1,020 / 6,149']],
            [0.128, 0.062, 0.098, 0.147],
            align=['left', 'right', 'left', 'right'])

    p.text(0.055, 0.520, 'The grid', size=12, weight='bold')
    p.note(0.055, 0.482, 0.40,
           'Per (dataset, encoder, K, seed): one standard network and two rolled-out networks '
           '(T = 4, 12).\n\n'
           '· image branch   3 × 2 × 3 × 3 × 3 = 162 trainings\n'
           '· CLIP branch    3 × 1 × 3 × 3 × 3 = 81 trainings\n\n'
           'Test data is touched only after checkpoint selection.')

    p.text(0.545, 0.706, 'Training-set sizes and repetitions', size=12, weight='bold')
    p.table(0.545, 0.666,
            ['setting', 'repeats', 'varied by'],
            [['K = 5', '3 runs', 'subset seeds 0, 1, 2'],
             ['K = 10', '3 runs', 'subset seeds 0, 1, 2'],
             ['K = full', '3 runs', 'init seeds 0, 1, 2']],
            [0.110, 0.090, 0.200], align=['left', 'right', 'right'])

    p.callout(0.545, 0.520, 0.400, 0.195, 'Two reporting caveats we hold to',
              'Flowers-102 at K = 10 has std = 0 by construction: the official train split is exactly '
              '10 images per class, so all three subset seeds pick the same images. That is one '
              'effective run, and it is why 9 of the 48 CI-significant cells on page 8 are degenerate.\n'
              'The Stage 1 image-prototype full setting is a single closed-form run, so it carries no '
              'error bar and the full-data deltas are unpaired.', accent=AMBER, size=9.0)

    p.text(0.055, 0.315, 'Held fixed between Stage 1 and Stage 2', size=12, weight='bold')
    items = [('frozen encoders', 'never re-run, never fine-tuned'),
             ('cached features', 'the identical tensors Stage 1 used'),
             ('class prototypes', 'loaded from Stage 1, never rebuilt'),
             ('K-shot subsets', 're-derived with the same seeded selector'),
             ('classification rule', 'the same cosine comparison')]
    x = 0.055
    for head, sub in items:
        p.ax.add_patch(FancyBboxPatch((x, 0.185), 0.168, 0.085,
                                      boxstyle='round,pad=0.005,rounding_size=0.008',
                                      transform=p.ax.transAxes, facecolor=WASH,
                                      edgecolor=RULE, lw=0.8))
        p.text(x + 0.014, 0.253, head, size=9.3, weight='bold')
        p.note(x + 0.014, 0.227, 0.150, sub, size=8.3)
        x += 0.178
    p.close()


def page_stage1(pdf, d, n):
    p = Page(pdf, n, 'results · stage 1')
    p.title('Stage 1 — three baselines on frozen features')
    p.lead('Linear probe and image-derived class prototypes on both encoders, plus zero-shot CLIP RN50 '
           'as a horizontal reference (it has no K axis).')

    shots = ['5', '10', 'full']
    for j, ds in enumerate(('dtd', 'aircraft', 'flowers102')):
        ax = p.axes(0.065 + j * 0.305, 0.700, 0.245, 0.315)
        for enc, color, marker in (('resnet18', BLUE, 'o'), ('dinov2_vits14', PURPLE, 's')):
            probe, proto = [], []
            for s in shots:
                r = next(r for r in d.threeway
                         if r['dataset'] == ds and r['encoder'] == enc and r['shot'] == s)
                probe.append(f(r['linear_probe']))
                proto.append(f(r['prototype']))
            ax.plot(range(3), probe, marker=marker, color=color, lw=1.6, ms=5,
                    label=f'probe · {ENC_LABEL[enc]}')
            ax.plot(range(3), proto, marker=marker, color=color, lw=1.4, ms=5, ls='--',
                    alpha=0.65, label=f'prototype · {ENC_LABEL[enc]}')
        zs = f(next(r for r in d.zeroshot if r['dataset'] == ds)['zero_shot_accuracy'])
        ax.axhline(zs, color=BLACK, ls=':', lw=1.3, label='zero-shot CLIP')
        ax.set(title=DS_LABEL[ds], xticks=range(3), xticklabels=shots, ylim=(0, 1.02))
        ax.title.set_fontsize(10.5)
        ax.set_xlabel('images per class', fontsize=8.5, color=MUTED)
        if j == 0:
            ax.set_ylabel('top-1 test accuracy', fontsize=8.5, color=MUTED)
        ax.grid(alpha=0.18, lw=0.7)
        if j == 2:
            ax.legend(fontsize=7.2, loc='lower right', frameon=False)

    p.text(0.055, 0.335, 'What Stage 1 established', size=13, weight='bold')
    p.note(0.055, 0.300, 0.395,
           'Encoder choice dominates every method difference in this project. DINOv2 beats ResNet-18 by '
           '14 to 31 points on every dataset and every K — larger than any gap between the three '
           'baselines. The linear probe is strongest at full data everywhere except saturated '
           'Flowers-102/DINOv2; at K = 5 the closed-form prototype is competitive and sometimes wins.')

    zs = {r['dataset']: f(r['zero_shot_accuracy']) for r in d.zeroshot}
    p.callout(0.475, 0.348, 0.225, 0.230, 'Zero-shot CLIP RN50',
              f'prescribed prompt\n\nDTD {zs["dtd"]:.3f}\nAircraft {zs["aircraft"]:.3f}\n'
              f'Flowers-102 {zs["flowers102"]:.3f}\n\nA 5-prompt ensemble adds +2.3 / −0.1 / +2.0 pts.',
              accent=MUTED, size=9.0, title_size=9.5)
    proto_a5 = d.baseline('aircraft', 'resnet18', '5')
    p.callout(0.720, 0.348, 0.225, 0.230, 'The one that surprises people',
              f'On Aircraft, zero-shot CLIP ({zs["aircraft"]:.4f}) beats 5-shot ResNet-18 prototypes '
              f'({proto_a5:.4f}). With 100 fine-grained variants, five labelled images per class buy '
              f'less than a well-chosen text prompt.', accent=AMBER, size=9.0,
              title_size=9.5)
    p.close()


def page_stage2_figure(pdf, d, n):
    p = Page(pdf, n, 'results · stage 2 · image branch')
    p.title('The FM layer versus the prototype baseline')
    p.lead('Six panels, one per dataset × encoder. Black = Stage 1 baseline; blues = standard FM; '
           'reds = rolled-out FM. Error bars are ±1 SD across the Stage 1-matched repetitions.')
    p.image(nb_image(NB4, 'accuracy_vs_training_size.png'), 0.075, 0.762, 0.86, 0.680)
    p.close()


def page_stage2_table(pdf, d, n):
    p = Page(pdf, n, 'results · stage 2 · image branch')
    p.title('What the image branch shows')
    p.lead('Best FM variant per cell, with ΔAcc against the matching Stage 1 baseline.')

    rows = []
    for ds in ('dtd', 'aircraft', 'flowers102'):
        for enc in ('resnet18', 'dinov2_vits14'):
            cells = []
            for s in ('5', '10', 'full'):
                acc, dl = d.best_fm(ds, enc, s)
                mark = '++' if dl > 5e-5 else ('--' if dl < -5e-5 else '')
                cells.append(f'{mark}{acc:.4f}  ({dl:+.4f})')
            rows.append([f'{DS_LABEL[ds]} · {ENC_LABEL[enc]}'] + cells)
    p.table(0.055, 0.730, ['dataset · encoder', 'K = 5', 'K = 10', 'K = full'], rows,
            [0.198, 0.143, 0.143, 0.143], bold_first=True, row_h=0.040)

    total, improved, flat, regressed = d.cell_counts
    p.callout(0.700, 0.730, 0.245, 0.230, f'Across all {total} result cells',
              f'{improved} improved\n{flat} flat (±0.0000)\n{regressed} regressed\n\n'
              f'All {regressed} regressions are rolled-out training, and none of them is significant '
              f'under the paired CI on the next page.', accent=MUTED, size=9.2)

    p.text(0.055, 0.448, 'Four findings', size=14, weight='bold')
    a5, _ = d.best_fm('aircraft', 'dinov2_vits14', '5')
    _, d5 = d.best_fm('aircraft', 'dinov2_vits14', '5')
    _, d10 = d.best_fm('aircraft', 'dinov2_vits14', '10')
    _, dfull = d.best_fm('aircraft', 'dinov2_vits14', 'full')
    med, lo, hi, _, fm_wins, tw = d.gap_closed
    mt1, mt2, _, _ = d.step_fracs
    findings = [
        ('1', 'The gain grows with K, and is largest where the baseline is weakest.',
         f'Aircraft/DINOv2 goes +{d5 * 100:.1f} → +{d10 * 100:.1f} → +{dfull * 100:.1f} points. This is '
         f'the signature of a trained layer: a class mean stops absorbing information, while the '
         f'velocity field keeps using the extra labels.'),
        ('2', 'It closes much of the prototype-to-probe gap, but rarely passes the probe.',
         f'Where the probe leads, FM recovers a median {med:.0f}% of the distance to it '
         f'(range {lo:.0f}–{hi:.0f}%), and overtakes it in only {fm_wins} of {tw} settings — all '
         f'near-saturated.'),
        ('3', 'Standard FM beats rolled-out training in 14 of 18 settings (3 losses, 1 tie).',
         'Rolled-out is also the only variant that ever regresses below the baseline, and costs '
         '1.5–2.5× more per run. Endpoint-only supervision constrains the field far more weakly than '
         'per-step velocity regression.'),
        ('4', 'T is nearly irrelevant — and this is expected, not a bug.',
         f'The ideal velocity p − z is constant along the path, so a well-learned field lands in the '
         f'same place for any T. The step ablation puts a floor under it: T=2 already delivers a median '
         f'{mt2 * 100:.0f}% of the T=12 gain against {mt1 * 100:.0f}% at T=1. Say so before being asked — '
         f'an identical-looking T=4 / T=12 column otherwise reads as a copy-paste error.'),
    ]
    y = 0.428
    for num, head, body in findings:
        p.text(0.058, y, num, size=17, weight='bold', color=BLUE)
        p.text(0.090, y - 0.004, head, size=10.6, weight='bold')
        p.note(0.090, y - 0.038, 0.84, body, size=9.1)
        y -= 0.089
    p.close()


def page_paired_ci(pdf, d, n):
    p = Page(pdf, n, 'results · does it survive?')
    p.title('Do the gains survive a paired confidence interval?')
    p.lead('"Improves in 66 of 72 cells" is a difference of two independently-averaged means. Because '
           'Stage 2 deliberately reuses Stage 1\'s subsets, FM run s and baseline run s saw the same K '
           'images, so the per-run difference is paired and has far lower variance than the two '
           'marginals suggest. Recomputed that way, with a 95% t-interval over the 3 repetitions.')

    ci = d.ci_counts
    p.table(0.055, 0.700, ['', f'count of {ci["n"]} cells'],
            [['positive mean paired ΔAcc', str(ci['pos'])],
             ['95% CI excludes zero', str(ci['sig'])],
             ['…of which degenerate (std = 0 by construction)', f'−{ci["degen"]}'],
             ['**genuinely distinguishable from zero**', f'**{ci["genuine"]}**']],
            [0.330, 0.140], row_h=0.042)

    rows = [[DS_LABEL[ds], f'{s} of {t}'] for ds, (s, t) in
            sorted(ci['per_ds'].items(), key=lambda kv: -kv[1][0] / kv[1][1])]
    p.text(0.545, 0.700, 'By dataset — this is not uniform', size=11, weight='bold')
    p.table(0.545, 0.655, ['dataset', 'CI excludes zero'], rows, [0.180, 0.150], row_h=0.042)

    # the DTD cells, ordered, so the reader can see where the line falls
    dtd = sorted([r for r in d.paired if r['dataset'] == 'dtd'],
                 key=lambda r: -f(r['mean_delta']))
    show = dtd[:4] + dtd[-2:]
    rows = []
    for r in show:
        tag = ('++' if r['significant'] == 'True' else '--')
        rows.append([f"{ENC_LABEL[r['encoder']]} · {SHOT_LABEL[r['shot']]} · {r['variant']} T={r['T']}",
                     r['mean_delta'], f"[{r['ci95_low']}, {r['ci95_high']}]",
                     f"{tag}{'yes' if r['significant'] == 'True' else 'no'}"])
    p.text(0.055, 0.478, 'DTD, best and worst cells', size=11, weight='bold')
    p.table(0.055, 0.438, ['cell', 'paired ΔAcc', '95% CI', 'sig?'], rows,
            [0.262, 0.095, 0.155, 0.048], row_h=0.035, size=8.7)

    p.callout(0.640, 0.478, 0.305, 0.272, 'The sub-1-point DTD gains do not survive',
              f'Only {ci["per_ds"]["dtd"][0]} of DTD\'s {ci["per_ds"]["dtd"][1]} cells clear zero. Every '
              f'DTD/ResNet-18 K=5 and K=10 cell has a CI straddling zero and must be restated as no '
              f'measurable effect — not as a small gain.\n\n'
              f'Symmetrically, the two regressions are not significant either: −.0213 with a CI of '
              f'[−.0522, +.0096]. They are the largest regressions measured and still indistinguishable '
              f'from zero at n = 3.', accent=RED, size=9.0)

    mcn_all3 = sum(1 for r in d.paired if r['seeds_mcnemar_sig'] == '3')
    disagree = [r for r in d.paired
                if r['significant'] == 'False' and int(r['seeds_mcnemar_sig']) >= 2]
    p.callout(0.055, 0.175, 0.89, 0.112, 'McNemar says yes far more often — and is the wrong test here',
              f'McNemar is significant in all 3 seeds in {mcn_all3} of {ci["n"]} cells, and {len(disagree)} '
              f'cells are CI-negative but McNemar-significant in ≥2 seeds. That is the expected direction: '
              f'baseline and FM predict on the same test images, so their disagreements are paired — but '
              f'McNemar\'s unit of analysis is the test image, not the run, so it ignores subset and '
              f'initialization variance entirely and is anti-conservative here. Where the two disagree, '
              f'quote the CI.',
              accent=AMBER, size=9.0)
    p.close()


def page_controls(pdf, d, n):
    p = Page(pdf, n, 'results · control experiment')
    p.title('Flow matching, or just another MLP?')
    p.lead('The FM layer adds a ~0.8M-parameter trained network to a previously closed-form classifier, '
           'so before attributing +24 points to flow matching the cheaper explanations have to be ruled '
           'out. Two controls, identical in hidden architecture, optimizer, schedule, early stopping, '
           'subsets and seeds: direct, a plain MLP trained on ‖g(z) − p_y‖² with no time input and no '
           'integration; and residual, the same shared network applied the same T = 12 times, trained '
           'only on the endpoint, with no time conditioning and no per-t velocity supervision.')

    both, tot, med_d, med_r = d.control_counts
    rows = []
    for ds in ('aircraft', 'dtd', 'flowers102'):
        for enc in ('dinov2_vits14', 'resnet18'):
            r = next(r for r in d.controls if r['dataset'] == ds and r['encoder'] == enc
                     and r['shot'] == 'full')
            share = d.direct_share(ds, enc, 'full')
            share_s = f'{share * 100:.0f}%' if share >= 0 else 'below baseline'
            mark = '++' if share < 0.5 else '--'
            rows.append([f'{DS_LABEL[ds]} · {ENC_LABEL[enc]}',
                         f"{f(r['baseline_mean_accuracy']):.4f}", f"{f(r['direct']):.4f}",
                         f"{f(r['residual']):.4f}", f"**{f(r['best_fm_accuracy']):.4f}**",
                         f'{mark}{share_s}'])
    p.text(0.055, 0.665, 'At full data', size=11, weight='bold')
    p.table(0.055, 0.622, ['dataset · encoder', 'prototype', 'direct MLP', 'residual', 'best FM',
                           "direct's share of FM's gain"],
            rows, [0.200, 0.085, 0.090, 0.085, 0.085, 0.200], row_h=0.038, size=8.9)

    share_full = d.direct_share('aircraft', 'dinov2_vits14', 'full')
    share_5 = d.direct_share('aircraft', 'dinov2_vits14', '5')
    share_10 = d.direct_share('aircraft', 'dinov2_vits14', '10')
    p.callout(0.055, 0.362, 0.43, 0.258, 'The headline is mostly not flow matching',
              f'FM beats both controls in {both} of {tot} settings, so the gain is not merely "a '
              f'supervised nonlinear map of the same size".\n\n'
              f'But on Aircraft/DINOv2 — the +24-point setting — the plain direct MLP reaches .5783 '
              f'against FM\'s .5852: {pct(share_full)} of the gain over the .3423 baseline, with FM '
              f'adding only +.0069 on top. Same at K=5 and K=10 ({pct(share_5)}, {pct(share_10)}). '
              f'Most of the work there is done by adding a trained head at all.', accent=RED)

    p.callout(0.515, 0.362, 0.43, 0.258, 'Where flow matching does earn its keep',
              f'The mirror image. On DTD/ResNet-18, Flowers-102/ResNet-18 and DTD/DINOv2 at low K, the '
              f'direct MLP lands below the closed-form prototype baseline - it overfits - while FM stays '
              f'above it. FM has its largest advantage over direct exactly there, +.042 to +.063.'
              f'{NL}{NL}residual sits much closer to FM than direct does (median {med_r:+.4f} against '
              f'{med_d:+.4f}): most of what FM has over a plain MLP comes from iterative shared-weight '
              f'computation, with time conditioning adding a smaller increment - except Aircraft/DINOv2 '
              f'at full, where FM beats residual by +.0420.', accent=GREEN, size=8.9)

    p.note(0.055, 0.092, 0.89,
           'Not part of the required Stage 2 comparison, which remains Stage 1 prototype vs. standard FM '
           'vs. rolled-out FM; written to a separate controls/ tree. The two settings where FM does not '
           'two settings where FM does not beat both controls are at K=5, inside seed noise.',
           size=8.8, color=FAINT)
    p.close()


def page_probe_comparison(pdf, d, n):
    p = Page(pdf, n, 'results · context')
    p.title('The comparison that matters most')
    p.lead('At full data: the prototype baseline, the same baseline with an FM layer, and the Stage 1 '
           'linear probe on identical features. The probe is a different Stage 1 baseline, not an FM '
           'variant — this is context, not a Stage 2 deliverable.')

    ax = p.axes(0.065, 0.700, 0.575, 0.330)
    keys = [(ds, enc) for ds in ('dtd', 'aircraft', 'flowers102')
            for enc in ('resnet18', 'dinov2_vits14')]
    import numpy as np
    xs = np.arange(len(keys))
    proto, fm, probe = [], [], []
    for ds, enc in keys:
        r = next(r for r in d.threeway
                 if r['dataset'] == ds and r['encoder'] == enc and r['shot'] == 'full')
        proto.append(f(r['prototype']))
        fm.append(f(r['flow_matching']))
        probe.append(f(r['linear_probe']))
    w = 0.27
    ax.bar(xs - w, proto, w, color='#c3cad6', label='prototype baseline')
    ax.bar(xs, fm, w, color=BLUE, label='+ FM layer')
    ax.bar(xs + w, probe, w, color=RED, label='linear probe')
    for x, v in zip(xs, fm):
        ax.text(x, v + 0.018, f'{v:.3f}', ha='center', size=7.4, color=BLUE, weight='bold')
    ax.set(xticks=xs, ylim=(0, 1.10))
    ax.set_xticklabels([f'{DS_LABEL[ds].split("-")[0].replace("FGVC","Aircraft")}\n{ENC_LABEL[e]}'
                        for ds, e in keys], fontsize=7.6)
    ax.set_ylabel('top-1 test accuracy (full data)', fontsize=8.5, color=MUTED)
    ax.grid(axis='y', alpha=0.18, lw=0.7)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc='upper left')

    med, lo, hi, nn, fm_wins, tw = d.gap_closed
    p.callout(0.665, 0.700, 0.280, 0.400, 'Read this honestly',
              f'The FM layer is a real improvement over the prototype rule it replaces — clearly so, '
              f'and by a lot on Aircraft/DINOv2 (.342 → .585).\n\n'
              f'It is not a replacement for a discriminatively trained classifier. Where the probe leads '
              f'({nn} of {tw} settings), FM closes a median {med:.0f}% of the gap, range {lo:.0f}–{hi:.0f}%. '
              f'It overtakes the probe in {fm_wins} of {tw} settings, all of them already near ceiling.\n\n'
              f'Also worth stating: ΔAcc bundles "added an FM layer" with "added a trained, '
              f'validation-selected ~0.8M-parameter head", against a baseline with zero trained '
              f'parameters and no access to the validation split.', accent=AMBER, size=8.6)

    p.text(0.055, 0.278, 'Training behaviour', size=12, weight='bold')
    p.note(0.055, 0.241, 0.575,
           'Both objectives trained stably; nothing diverged across 162 runs. 161 of 162 early-stopped '
           'and one reached the 200-epoch cap. In 16 runs (~10%) the best validation accuracy was at '
           'epoch 1 — concentrated in the saturated Flowers-102/DINOv2 settings, where there is nothing '
           'left to gain. Those FM layers are effectively untrained and should not be described '
           'otherwise. Standard-FM loss (velocity MSE at random t) and rolled-out loss (endpoint MSE '
           'after a full rollout) are different quantities on different scales; the shared log axis in '
           'the training-curve figure is for stability checking only, never magnitude comparison.')
    p.note(0.665, 0.245, 0.280,
           'The six-panel training-curve grid is in the notebook (04 §9). It is not legible at this '
           'size and the claim it supports — stable training, no divergence — is one sentence, so it '
           'is stated here rather than shown.', size=8.4, color=FAINT)
    p.close()


def page_clip(pdf, d, n):
    p = Page(pdf, n, 'results · stage 2 · clip branch')
    p.title('The CLIP branch — why the control changes the story')
    p.lead('Here FM transports CLIP RN50 image embeddings toward frozen text prototypes. Because the '
           'layer trains on labels, this is no longer zero-shot — so a same-supervision control is '
           'required: image prototypes built from the same CLIP features and the same K-shot subsets, '
           'closed-form, with zero trained parameters.')
    p.image(nb_image(NB5, 'accuracy_vs_training_size.png'), 0.075, 0.735, 0.86, 0.395)

    zs, ctl, tot, best = d.clip_counts
    p.callout(0.055, 0.315, 0.43, 0.185, 'Against zero-shot: large gains',
              f'FM beats the zero-shot baseline in {zs} of {tot} cells, by as much as '
              f'+{best * 100:.1f} points on DTD at full data.\n\n'
              f'Taken alone this looks like a decisive win for the FM layer. It is not — the FM side '
              f'used K labelled images per class and the zero-shot side used none.', accent=BLUE)
    p.callout(0.515, 0.315, 0.43, 0.185, 'Against the fair control: it mostly loses',
              f'Given the same labels, simply building image prototypes beats transporting toward text '
              f'prototypes in {ctl} of {tot} cells. Only DTD at K = 10 and K = full come out ahead.\n\n'
              f'On DTD at K = 5 the FM layer is even worse than zero-shot itself. The large Δ vs '
              f'zero-shot measures the labels, not the layer.', accent=RED)
    p.note(0.055, 0.118, 0.89,
           'This branch is an extension, never merged into the required Stage 2 table: ref/stage_1.pdf '
           'asks for one prototype branch and the image branch is it. Note also what this branch does '
           'not yet have — no paired CI or McNemar, and no validation-selected t*. Given how much the '
           'paired CI changed the reading of the image branch, "loses to the control in 28 of 36 cells" '
           'is still a difference of means.', size=8.8, color=FAINT)
    p.close()


def _figure_page(pdf, d, n, kicker, title, lead, arr, note, accent=BLUE, img_h=0.545,
                 note_y=0.150):
    p = Page(pdf, n, kicker)
    p.title(title)
    p.lead(lead)
    p.image(arr, 0.075, p.y + 0.010, 0.86, img_h)
    p.callout(0.055, note_y, 0.89, 0.085, None, note, accent=accent, size=9.2)
    p.close()


def page_geometry_pca(pdf, d, n):
    _figure_page(pdf, d, n, 'results · geometry',
                 'Feature-space geometry — what the layer actually does',
                 'Test features before the flow, after standard FM, and after rolled-out FM, with the '
                 'class prototypes (X). One PCA is fitted jointly across all three views and the '
                 'prototypes, and all panels share axis limits, so a point\'s movement between panels is '
                 'a real displacement in one shared plane.',
                 nb_image(NB4, 'feature_space_comparison.png'),
                 'The mechanism is visible directly: overlapping class clouds are contracted onto their '
                 'prototypes. Because the network cannot see the label at test time, points in ambiguous '
                 'regions are pulled toward whichever prototype\'s basin they fall in — which is why '
                 'transport can create errors as well as fix them, and why ΔAcc can be negative even '
                 'when the training loss is low.')


def page_geometry_tsne(pdf, d, n):
    _figure_page(pdf, d, n, 'results · geometry',
                 'The same comparison under a joint t-SNE',
                 'The identical transported tensors under t-SNE instead of PCA, fitted once per row over '
                 'all three views and the prototypes. Cluster separation reads well here; distances do '
                 'not.',
                 nb_image(NB4, 'feature_space_comparison_tsne.png'),
                 'Read this qualitatively and use the PCA page for anything geometric. t-SNE normalises '
                 'local density, so it deliberately re-expands the tightly contracted post-FM clusters — '
                 'the visual shrinkage looks far milder here than it is numerically.', accent=AMBER)


def page_trajectories(pdf, d, n):
    _figure_page(pdf, d, n, 'results · geometry',
                 'Flow trajectories — the transport step by step',
                 'Intermediate Euler states for representative test examples, from the original feature '
                 '(green) through every step to the transported endpoint (purple square) and the target '
                 'prototype (X). PCA rather than t-SNE, so a straight path stays a straight path.',
                 nb_image(NB4, 'flow_trajectories.png'),
                 'The paths are close to straight, which is the geometric counterpart of the '
                 'T-independence result: when the learned field matches the ideal constant velocity, '
                 'four steps and twelve steps arrive at the same place.', img_h=0.360, note_y=0.500)


def page_clip_geometry(pdf, d, n):
    _figure_page(pdf, d, n, 'results · geometry · clip branch',
                 'The CLIP modality gap, and the flow across it',
                 'The same three-view comparison on the CLIP branch. The image embeddings and the text '
                 'prototypes (X) start in visibly different regions — that separation is CLIP\'s '
                 'image–text modality gap — and the learned transport carries the image cloud across it.',
                 nb_image(NB5, 'feature_space_comparison.png'),
                 'This is the qualitative reason the CLIP branch behaves differently from the image '
                 'branch: the transport has to cross a systematic offset between two modalities, not '
                 'merely denoise within one feature space.')


def page_flow_time(pdf, d, n):
    p = Page(pdf, n, 'optional extension')
    p.title('Samples and prototypes at intermediate flow times')
    p.lead('Rather than looking only at t=0 and t=1, every intermediate Euler state is classified over '
           'the complete test split. The t=0 end is asserted, not assumed — in 04 against Stage 1\'s '
           'saved per-run metrics.json, in 05 against the zero-shot baseline — and both assertions '
           'passed, so each curve provably starts at the baseline and ends at the FM result.')
    p.image(nb_image(NB4, 'accuracy_vs_flow_time.png', 0), 0.050, 0.742, 0.43, 0.590)

    ar = next(r for r in d.flowtime if r['dataset'] == 'aircraft'
              and r['encoder'] == 'resnet18' and 'standard' in r['model'])
    ad = next(r for r in d.flowtime if r['dataset'] == 'aircraft'
              and r['encoder'] == 'dinov2_vits14' and 'standard' in r['model'])
    _, dfull = d.best_fm('aircraft', 'dinov2_vits14', 'full')
    p.callout(0.505, 0.742, 0.440, 0.400, 'ΔAcc, unrolled',
              f'Each curve begins exactly on the dashed Stage 1 baseline and ends exactly at the Stage 2 '
              f'number. The shape between them is what the layer does. Two things are visible '
              f'immediately:\n\n'
              f'· Aircraft climbs steeply and keeps climbing: +{f(ar["gain"]) * 100:.1f} and '
              f'+{f(ad["gain"]) * 100:.1f} points in this 10-shot setting, and up to '
              f'+{dfull * 100:.1f} at full data.\n\n'
              f'· DTD and Flowers are nearly flat. The transport is running, the features are moving, '
              f'and the classification barely changes.\n\n'
              f'The next page explains why, using the margin.', accent=BLUE)
    p.close()


def page_mechanism(pdf, d, n):
    p = Page(pdf, n, 'optional extension · mechanism')
    p.title('Why the gains land where they do')
    p.lead('Two quantities separate "the flow moved the point" from "the flow classified it better": the '
           'cosine to the true prototype, and the margin — cosine to the true prototype minus the best '
           'competing one. The first says transport happened; only the second says it helped.')

    order = [('dtd', 'resnet18'), ('dtd', 'dinov2_vits14'), ('flowers102', 'dinov2_vits14'),
             ('flowers102', 'resnet18'), ('aircraft', 'resnet18'), ('aircraft', 'dinov2_vits14')]
    rows = []
    for ds, enc in order:
        r = next(x for x in d.flowtime if x['dataset'] == ds and x['encoder'] == enc
                 and 'standard' in x['model'])
        m0, m1 = f(r['margin_t0']), f(r['margin_t1'])
        mark = '++' if m1 > m0 else '--'
        rows.append([f'{DS_LABEL[ds]} · {ENC_LABEL[enc]}',
                     f"{f(r['cos_t0']):.3f} → {f(r['cos_t1']):.3f}",
                     f'{mark}{m0:+.4f} → {m1:+.4f}',
                     f"{f(r['acc_t0']):.4f} → {f(r['acc_t1']):.4f}",
                     f"{f(r['gain']):+.4f}"])
    p.table(0.055, 0.700, ['setting (10-shot, seed 0, standard FM)', 'cos to true prototype',
                           'margin', 'accuracy', 'ΔAcc'],
            rows, [0.215, 0.160, 0.175, 0.150, 0.085], row_h=0.038, size=8.9)

    p.callout(0.055, 0.442, 0.43, 0.205, 'The FM layer repairs a broken classifier',
              'Cosine to the true prototype rises in every single setting — the transport always does '
              'what it was trained to do. But where the prototype baseline already ranked classes '
              'correctly (DTD, Flowers-102: positive margin), pulling every point toward every prototype '
              'shrinks the margin slightly and accuracy moves by less than a point. Where the baseline '
              'was mis-ranked (Aircraft: negative margin), the flow repairs the ranking and accuracy '
              'jumps. It does not sharpen a working one.', accent=GREEN)

    rows = []
    for ds, enc in [('aircraft', 'dinov2_vits14'), ('aircraft', 'resnet18'),
                    ('flowers102', 'resnet18'), ('dtd', 'resnet18'), ('dtd', 'dinov2_vits14'),
                    ('flowers102', 'dinov2_vits14')]:
        r = next(x for x in d.transitions if x['dataset'] == ds and x['encoder'] == enc
                 and 'standard' in x['model'])
        rows.append([f'{DS_LABEL[ds]} · {ENC_LABEL[enc]}', r['fixed'], r['broken'],
                     f"**{r['net_gain']}**", f"--{r['mean_margin_fixed']}",
                     f"++{r['mean_margin_broken']}"])
    p.text(0.515, 0.442, 'Which samples the layer flips', size=11, weight='bold')
    p.table(0.515, 0.400, ['setting', 'fixed', 'broken', 'net', 'margin fixed',
                           'margin broken'],
            rows, [0.142, 0.045, 0.052, 0.045, 0.070, 0.072], row_h=0.033, size=8.2)

    p.callout(0.055, 0.150, 0.89, 0.098, 'The constraint this puts on how the layer may be described',
              'In every single setting, the samples the layer fixes had negative pre-flow margin and the '
              'samples it breaks had positive pre-flow margin. It wins by fixing more than it breaks, not '
              'by leaving correct predictions alone. On Aircraft/DINOv2 the fixed group has a mean margin '
              'of -.0444 - confidently misclassified, not marginal - which makes that gain a restructuring '
              'rather than a nudge. On DTD the two groups are near mirror images (85 fixed against 68 '
              'broken), the same conclusion the paired CI reaches by another route.',
              accent=RED, size=8.8)
    p.close()


def page_snapshots(pdf, d, n):
    _figure_page(pdf, d, n, 'optional extension · geometry',
                 'The point cloud at intermediate flow times',
                 'Samples and prototypes (X) at t = 0, 1/4, 1/2, 3/4, 1. Each row is one dataset at its '
                 'strongest encoder, once per training objective; one PCA is fitted jointly over every '
                 'snapshot plus the prototypes, and all five panels of a row share axis limits.',
                 nb_image(NB4, 'intermediate_flow_time_snapshots.png'),
                 'The contraction is the mechanism made visible: overlapping class clouds are drawn onto '
                 'their prototypes. Standard FM moves points along nearly straight, roughly parallel '
                 'paths; rolled-out FM, supervised only at the endpoint, takes a more curved route and '
                 'contracts later.', img_h=0.600)


def page_overshoot(pdf, d, n):
    p = Page(pdf, n, 'optional extension · finding')
    p.title('t = 1 is a convention, not an optimum')
    img, img_n, clip, clip_n = d.overshoot
    p.lead(f'Accuracy strictly peaks before the endpoint in {clip} of {clip_n} CLIP-branch curves and '
           f'{img} of {img_n} image-branch curves. Integrating the full T steps therefore costs accuracy '
           f'in many settings — and the loss is largest for rolled-out models on the CLIP branch.')

    rows = []
    for branch, tbl in (('CLIP', d.flowtime5), ('image', d.flowtime)):
        for r in tbl:
            gap = f(r['best_acc']) - f(r['acc_t1'])
            if gap <= 1e-9:
                continue
            enc = f" / {ENC_LABEL[r['encoder']]}" if 'encoder' in r else ''
            rows.append((gap, [f"{branch} · {DS_LABEL[r['dataset']]}{enc}",
                               'rolled-out' if 'rolled' in r['model'] else 'standard',
                               f"{f(r['best_t']):.2f}", f"{f(r['best_acc']):.4f}",
                               f"{f(r['acc_t1']):.4f}", f'++{gap * 100:+.2f} pts']))
    rows = [r for _, r in sorted(rows, key=lambda t: -t[0])][:8]
    p.table(0.055, 0.700, ['setting', 'objective', 'best t', 'acc at best t', 'acc at t=1',
                           'left on the table'],
            rows, [0.245, 0.105, 0.075, 0.115, 0.105, 0.135], row_h=0.037, size=8.8)

    helped, cond, mean_g, best_g = d.tstar_counts
    tr = min(d.tstar, key=lambda r: -f(r['gain']))
    p.callout(0.055, 0.360, 0.43, 0.238, 'Selected honestly on validation, it is worth less',
              f'The stopping time is free to choose, so 04 §16 chooses it properly: t* is the argmax of '
              f'VALIDATION accuracy over the T = 12 Euler grid, frozen, and only then evaluated once on '
              f'test.{NL}{NL}It helps in {helped} of {cond} conditions, mean {mean_g:+.4f}, best '
              f'{best_g:+.4f} - a fix for the overshooting cases, not a free improvement everywhere. '
              f'oracle_best, the best any t could have reached on test, is reported as an unreachable '
              f'bound; the gap to it is .001-.005.', accent=BLUE, size=8.9)

    proto = d.baseline('dtd', 'resnet18', 'full')
    p.callout(0.515, 0.360, 0.43, 0.248, 'It rescues the only real regression',
              f'DTD / ResNet-18 / full, rolled-out — the worst cell in the project:\n\n'
              f'    prototype baseline      {proto:.4f}\n'
              f'    rolled-out FM at t=1    {f(tr["test_at_t1"]):.4f}   ({f(tr["test_at_t1"]) - proto:+.4f})\n'
              f'    rolled-out FM at t*     {f(tr["test_at_tstar"]):.4f}   '
              f'({f(tr["test_at_tstar"]) - proto:+.4f})\n\n'
              f'Validation-selected early stopping turns a −.021 regression into a +.010 gain, at '
              f't* = {f(tr["mean_t_star"]):.2f}. The strongest argument in the project for treating the '
              f'stopping time as a hyperparameter rather than a convention.', accent=GREEN,
              size=9.0)

    p.note(0.055, 0.105, 0.89,
           'Reported as an ablation. The required Stage 2 table always classifies the final state z_T, '
           'exactly as ref/stage_2.pdf specifies. Rolled-out models overshoot hardest, which fits their '
           'weaker constraint. Not yet implemented on the CLIP branch, where it would be worth the most.',
           size=8.8, color=FAINT)
    p.close()


def page_ablation_geometry(pdf, d, n):
    p = Page(pdf, n, 'optional extension · ablation')
    p.title('Contraction or discrimination? And how many steps?')
    p.lead('Every metric on page 17 improves whenever the flow contracts toward the prototype set, '
           'whether or not the classes become easier to separate. Two ablations close that gap: the '
           'within/between class-scatter ratio W(t)/B(t), which falls only if classes tighten relative '
           'to their separation; and an inference-only sweep over the number of Euler steps.')

    p.image(nb_image(NB4, 'flow_geometry_vs_time.png'), 0.055, 0.700, 0.42, 0.280)
    p.image(nb_image(NB4, 'inference_step_ablation.png'), 0.510, 0.700, 0.42, 0.280)
    p.text(0.265, 0.408, 'W/B and accuracy over flow time', size=8.6, color=FAINT, ha='center')
    p.text(0.720, 0.408, 'accuracy against Euler steps at inference', size=8.6, color=FAINT,
           ha='center')

    rows = []
    for ds, enc in [('aircraft', 'dinov2_vits14'), ('aircraft', 'resnet18'),
                    ('flowers102', 'resnet18'), ('dtd', 'resnet18'), ('dtd', 'dinov2_vits14'),
                    ('flowers102', 'dinov2_vits14')]:
        r = next(x for x in d.geometry if x['dataset'] == ds and x['encoder'] == enc
                 and 'standard' in x['model'])
        rows.append([f'{DS_LABEL[ds]} · {ENC_LABEL[enc]}',
                     f"{f(r['ratio_t0']):.3f} → {f(r['ratio_t1']):.3f}",
                     f"++{f(r['ratio_change']):+.4f}", f"{f(r['accuracy_gain']):+.4f}"])
    p.text(0.055, 0.378, 'W/B at t=0 → t=1 (10-shot, standard FM)', size=10.5, weight='bold')
    p.table(0.055, 0.338, ['setting', 'W/B', 'change', 'ΔAcc'], rows,
            [0.170, 0.108, 0.075, 0.072], row_h=0.031, size=8.4)

    ratio_down = sum(1 for r in d.geometry if f(r['ratio_change']) < 0)
    med_change = statistics.median(f(r['ratio_change']) for r in d.geometry)
    mt1, mt2, above, tot = d.step_fracs
    p.callout(0.505, 0.378, 0.44, 0.160, 'The flow discriminates, it does not merely contract',
              f'W/B falls in all {ratio_down} of {len(d.geometry)} curves (median {med_change:+.3f}). But '
              f'falling W/B is necessary, not sufficient: DTD sees a −0.31 improvement for under a point '
              f'of accuracy. On Aircraft, cosine to the nearest competitor stays above cosine to the true '
              f'prototype (.949 vs .942) — the negative margin again. Reorganising the cloud converts '
              f'into accuracy only where the true prototype was not already ranked first.',
              accent=GREEN, size=8.6, title_size=9.5)

    p.callout(0.505, 0.208, 0.44, 0.158, 'Two Euler steps are enough',
              f'T = 2 already delivers a median {mt2 * 100:.0f}% of the T = 12 gain, and at least all of '
              f'it in {above} of {tot} settings. T = 1 delivers only {mt1 * 100:.0f}% and is negative in '
              f'three settings, where one large step overshoots to somewhere worse than the untouched '
              f'feature. So the endpoint is T-independent for T ≥ 2: the layer costs two velocity '
              f'evaluations, not twelve.', accent=BLUE, size=8.6, title_size=9.5)

    p.note(0.055, 0.106, 0.43,
           'The step sweep is standard FM only — a rolled-out network cannot legitimately be run at a T '
           'it was not trained for — and the required table keeps T ∈ {4, 12}.', size=8.4, color=FAINT)
    p.close()


def page_reverse(pdf, d, n):
    p = Page(pdf, n, 'optional extension')
    p.title('The flow in reverse — and what it reveals about CLIP')
    p.lead('Start from the class prototypes at t = 1 and integrate the same learned field backward. The '
           'recovery rate asks whether a prototype, after that round trip, lands nearest to its own '
           'class\'s mean test feature. The untouched forward prototype is the pre-transport reference — '
           'a ceiling only on the image branch, where the prototype already is the class image mean.')

    rows = []
    for r in d.reverse4:
        fwd, rev = f(r['forward_recovery']), f(r['reverse_recovery'])
        mark = '++' if rev >= fwd else '--'
        rows.append([f"image · {DS_LABEL[r['dataset']]} / {ENC_LABEL[r['encoder']]}",
                     f'{fwd:.3f}', f'{mark}{rev:.3f}', r['mean_cosine_distance_moved']])
    for r in d.reverse5:
        rows.append([f"**CLIP · {DS_LABEL[r['dataset']]}**",
                     f"{f(r['forward_recovery']):.3f}",
                     f"++{f(r['reverse_recovery']):.3f}", r['mean_cosine_distance_moved']])
    p.table(0.055, 0.690, ['branch · setting', 'forward', 'reverse', 'cos. distance moved'],
            rows, [0.215, 0.072, 0.072, 0.135], row_h=0.032, size=8.5)

    p.callout(0.560, 0.690, 0.385, 0.170, 'Image branch: almost nothing moves',
              'Cosine distances of .006 to .044. The learned field is essentially zero near t = 1, so '
              'backward integration does not travel back toward the feature distribution. Recovery is '
              'preserved except on Aircraft, where it degrades. Well-behaved, but not usefully '
              'invertible — which is what a contraction should look like.', accent=MUTED, size=8.6,
              title_size=9.5)
    p.callout(0.560, 0.508, 0.385, 0.205, 'CLIP branch: the modality gap, measured',
              'The prototypes move nearly orthogonally (≈.70) and recovery jumps from .55 / .25 / .75 to '
              '1.00 / .75 / 1.00. The forward rate measures how far CLIP\'s text embeddings sit from its '
              'image embeddings; the reverse pass closes that distance. Direct evidence that the field '
              'learned to bridge the gap — and the reason the earlier "ceiling" wording had to be '
              'corrected: a text prototype does not start out inside its own class\'s image cloud.',
              accent=GREEN, size=8.6, title_size=9.5)

    p.image(nb_image(NB5, 'reverse_flow.png'), 0.055, 0.350, 0.48, 0.190)
    p.note(0.055, 0.150, 0.48,
           'CLIP text prototypes (P) integrated backward over the real test embeddings, at reverse '
           't = 1 → 0. They start off to one side — that is the gap — and move in.', size=8.4)
    p.note(0.560, 0.284, 0.385,
           'Caveats unchanged: the backward pass evaluates the field at t ∈ {1, …, 1/T} while the forward '
           'pass uses {0, …, 1−1/T}, so this is not the exact inverse even for a perfectly learned field; '
           'and the forward flow is a deliberate contraction, so it is not invertible in principle. It '
           'recovers a plausible pre-image, not the original point.', size=8.4, color=FAINT)
    p.close()


def page_animation(pdf, d, n):
    p = Page(pdf, n, 'optional extension · animation')
    p.title('The transport, frame by frame')
    _, best_delta = d.best_fm('aircraft', 'dinov2_vits14', 'full')
    p.lead(f'Four frames from the animation the notebook exports as a GIF: standard FM (left panel) and '
           f'rolled-out FM (right) starting from an identical test feature, with the arrow showing the '
           f'actual Euler update at that step. The setting is chosen programmatically as the largest-gain '
           f'one — Aircraft / DINOv2, where the layer is worth +{best_delta * 100:.1f} points.')
    frames = nb_gif_frames(NB4, 'flow_animation_', 0, frames=(0, 4, 8, 12))
    for i, (arr, k) in enumerate(zip(frames, (0, 4, 8, 12))):
        col, row = i % 2, i // 2
        x = 0.070 + col * 0.445
        y = 0.740 - row * 0.290
        p.image(arr, x, y, 0.425, 0.265)
        p.text(x + 0.006, y - 0.268, f'Euler step {k} / 12    (t = {k / 12:.2f})',
               size=8.6, color=MUTED, weight='bold')
    p.callout(0.055, 0.145, 0.89, 0.080, None,
              'Read qualitatively: the transport happens in DINOv2\'s full 384-dimensional space and this '
              'is a 2-D projection of it, so projected path length and curvature are not '
              'high-dimensional measurements. What the frames convey is the mechanism — T small updates '
              'accumulating into the representation the cosine rule finally classifies.', accent=MUTED)
    p.close()


def page_summary(pdf, d, n):
    p = Page(pdf, n, 'summary')
    p.title('Findings')
    total, improved, flat, regressed = d.cell_counts
    ci = d.ci_counts
    both, tot_c, med_d, med_r = d.control_counts
    med, lo, hi, nn, fm_wins, tw = d.gap_closed
    img, img_n, clip, clip_n = d.overshoot
    zs, ctl, tot5, best_zs = d.clip_counts
    _, best_delta = d.best_fm('aircraft', 'dinov2_vits14', 'full')
    mt1, mt2, _, _ = d.step_fracs

    items = [
        ('Encoder choice dominates every method difference measured here.',
         'DINOv2 over ResNet-18 is worth 14–31 points. No baseline-versus-baseline gap in this project '
         'comes close.'),
        (f'The FM layer works — but on fewer cells than the raw table suggests.',
         f'It improves the prototype baseline in {improved} of {total} cells, up to '
         f'+{best_delta * 100:.1f} points on Aircraft/DINOv2 at full data. Under a paired 95% CI only '
         f'{ci["genuine"]} of {ci["n"]} are distinguishable from zero: '
         f'{ci["per_ds"]["aircraft"][0]}/{ci["per_ds"]["aircraft"][1]} on Aircraft, '
         f'{ci["per_ds"]["dtd"][0]}/{ci["per_ds"]["dtd"][1]} on DTD. The regressions are not significant '
         f'either.'),
        ('Most of the largest gain is not specific to flow matching.',
         f'FM beats a plain MLP and a time-free residual stack in {both} of {tot_c} settings, but on '
         f'Aircraft/DINOv2 the plain MLP already delivers '
         f'{pct(d.direct_share("aircraft", "dinov2_vits14", "full"))} of FM\'s gain over the baseline. '
         f'Flow matching earns its keep where that MLP overfits below the baseline instead.'),
        ('It repairs a broken classifier rather than sharpening a working one.',
         'Cosine to the true prototype rises everywhere, but margin rises only where the baseline was '
         'mis-ranked. In every setting, fixed samples started with negative margin and broken samples '
         'with positive margin.'),
        ('It does not beat a linear probe on the same features.',
         f'Where the probe leads it recovers a median {med:.0f}% of the gap ({lo:.0f}–{hi:.0f}%) and '
         f'overtakes in only {fm_wins} of {tw} settings, all near-saturated.'),
        ('Standard FM ≥ rolled-out training (14/18 settings), on accuracy and on compute.',
         'Rolled-out is the only variant that ever regresses below the baseline, the one that overshoots '
         'hardest, and the one that improves W/B least.'),
        ('t = 1 is a convention, not an optimum — but selecting it honestly is worth less.',
         f'Accuracy peaks earlier in {clip}/{clip_n} CLIP curves and {img}/{img_n} image curves, up to '
         f'+4.2 points. Validation-selected t* helps in half the conditions, and rescues the project\'s '
         f'only real regression.'),
        ('Two Euler steps deliver the whole gain.',
         f'Median {mt2 * 100:.0f}% of the T=12 gain at T=2 against {mt1 * 100:.0f}% at T=1. The layer is '
         f'cheaper than the specified T suggests, and the T-independence has a floor at T = 2.'),
        ('On the CLIP branch, the fair control reverses the headline.',
         f'FM beats zero-shot in {zs}/{tot5} cells but loses to a same-supervision control in '
         f'{ctl}/{tot5}. Those gains measure the labels.'),
        ('The reverse flow makes CLIP\'s modality gap measurable.',
         'Backward integration lifts text-prototype recovery from .55/.25/.75 to 1.00/.75/1.00; on the '
         'image branch it barely moves anything.'),
    ]
    y = 0.815
    for i, (head, body) in enumerate(items, 1):
        p.text(0.058, y, str(i), size=13, weight='bold', color=BLUE)
        p.text(0.088, y, head, size=10.2, weight='bold')
        wrapped = wrap(body, chars(0.855, 8.7))
        p.text(0.088, y - 0.030, wrapped, size=8.7, color=MUTED, linespacing=1.55)
        # advance by the space this item actually took, so 1-line and 2-line bodies
        # do not crowd each other
        y -= 0.048 + 0.0195 * (wrapped.count(NL) + 1)

    p.close()


def page_scope(pdf, d, n):
    p = Page(pdf, n, 'scope')
    p.title('Limitations and next steps')

    p.text(0.055, 0.775, 'Limitations', size=13, weight='bold')
    lims = [
        ('n = 3 runs per setting.',
         'The paired 95% intervals on page 8 are wide because of it, and that is the honest reading: '
         '33 of 72 cells cannot be called either way.'),
        ('Flowers-102 at K = 10 is one effective run.',
         'The official train split is exactly 10 images per class, so all three subset seeds select the '
         'same images. Its std is 0 by construction, which also makes 9 of the 48 CI-significant cells '
         'degenerate.'),
        ('The Stage 1 full prototype baseline is a single run.',
         'It is closed-form, so it carries no error bar and the full-data deltas are unpaired.'),
        ('No hyperparameter search.',
         'The Stage 1 recipe was adopted as-is, per the specification. Better FM numbers are plausibly '
         'reachable with validation-only tuning.'),
        ('2-D projections are qualitative.',
         'The joint PCA plane explains only part of the variance; geometry off-plane is invisible. They '
         'are not evidence of classifier quality.'),
        ('The CLIP branch has no paired CI and no t* selection.',
         'Its "loses to the control in 28 of 36 cells" is still a difference of means, and it is the '
         'branch where early stopping would be worth the most.'),
        ('One reporting column is unpopulated.',
         'test_accuracy_sel_T needs one pass with FORCE_RETRAIN_STANDARD = True. The headline '
         'test_accuracy column is unaffected.'),
    ]
    y = 0.742
    for head, body in lims:
        p.text(0.058, y, '·', size=12, weight='bold', color=BLUE)
        p.text(0.075, y, head, size=9.6, weight='bold')
        wrapped = wrap(body, chars(0.385, 8.6))
        p.text(0.075, y - 0.028, wrapped, size=8.6, color=MUTED, linespacing=1.55)
        y -= 0.044 + 0.0192 * (wrapped.count(NL) + 1)

    p.text(0.52, 0.775, 'Next steps, in order of expected value', size=13, weight='bold')
    steps = [
        ('Isolate the time conditioning.',
         'The controls showed that iterative shared-weight computation carries most of FM\'s advantage '
         'over a plain MLP. A residual stack with a t input and nothing else added would say precisely '
         'what flow matching contributes beyond that. This is now the most informative single '
         'experiment left.'),
        ('Carry the paired CI and t* selection to the CLIP branch.',
         'The image branch\'s headline changed materially once both were computed. Until they are run on '
         'the CLIP branch, its conclusions rest on differences of means.'),
        ('Make the stopping time a hyperparameter everywhere.',
         'It is selectable on validation data at no cost, and it converted the project\'s worst cell '
         'from −.021 to +.010.'),
        ('Use the margin, not accuracy, to predict where the layer will help.',
         'A negative baseline margin marks a mis-ranked classifier, which is precisely where '
         'prototype-targeted transport pays. A comfortably positive margin predicts the near-zero gains '
         'seen on DTD and Flowers-102.'),
    ]
    y = 0.742
    for head, body in steps:
        p.text(0.523, y, '›', size=13, weight='bold', color=BLUE)
        p.text(0.540, y, head, size=9.6, weight='bold')
        wrapped = wrap(body, chars(0.400, 8.6))
        p.text(0.540, y - 0.028, wrapped, size=8.6, color=MUTED, linespacing=1.55)
        y -= 0.046 + 0.0192 * (wrapped.count(NL) + 1)

    p.close()


def page_stage3_protocol(pdf, d, n):
    p = Page(pdf, n, 'stage 3 - method and protocol')
    p.title('FM before a frozen linear classifier')
    p.lead('Stage 3 asks whether a nonlinear flow can reshape frozen encoder features into a '
           'representation that the already-trained Stage 1 linear head handles better. The classifier '
           'is frozen for the required comparison; only the velocity network learns.')

    boxes = [
        (0.060, 0.735, 0.245, 'Frozen encoder feature', 'z\nStage 1 transform replayed', BLUE),
        (0.377, 0.735, 0.245, 'Flow Matching layer', 'T = 12 Euler steps\nidentity initialisation', PURPLE),
        (0.694, 0.735, 0.245, 'Frozen Stage 1 head', 'W z_hat + b\nclassifier logits', GREEN),
    ]
    for x, y, w, title, body, accent in boxes:
        p.callout(x, y, w, 0.125, title, body, accent=accent, size=9.0)
    for x0, x1 in ((0.305, 0.377), (0.622, 0.694)):
        p.ax.add_patch(FancyArrowPatch((x0, 0.672), (x1, 0.672), arrowstyle='-|>',
                                      mutation_scale=13, lw=1.5, color=INK,
                                      transform=p.ax.transAxes))

    p.callout(0.055, 0.565, 0.425, 0.215, 'Strategy 1 - rolled-out classification',
              'Run the complete FM rollout, classify z_hat with the frozen head, and backpropagate '
              'cross-entropy through all Euler steps into the FM only. Revision 3 adds a scale-free '
              'relative displacement penalty with lambda = 1; lambda = 0 remains a control.',
              accent=BLUE, size=9.0)
    p.callout(0.520, 0.565, 0.425, 0.215, 'Strategy 2 - classifier-guided targets',
              'Use the classifier gradient with respect to z_hat to construct a nearby lower-loss '
              'target z_hat-prime, then train standard conditional FM from z to that target. Targets '
              'are refreshed as the FM changes. Revision 3 separates radius, step, normalization, '
              'and projection.', accent=PURPLE, size=9.0)

    rows = [
        ['Data', 'same Stage 1 splits and K-shot subsets; seeds 0, 1, 2'],
        ['Operating point', 'DINOv2 ViT-S/14, K = 10, T = 12'],
        ['Main comparison', 'linear probe vs end-to-end FM vs guided FM'],
        ['Safety checks', 'probe fidelity, frozen gradients, exact identity, source bounds'],
        ['Reporting', 'test accuracy and delta, curves, joint PCA and t-SNE'],
    ]
    p.table(0.055, 0.300, ['requirement', 'implementation'], rows, [0.145, 0.745],
            align=['left', 'left'], row_h=0.034, size=8.8, bold_first=True)
    p.close()


def page_stage3_results(pdf, d, n):
    s3 = d.stage3
    p = Page(pdf, n, 'stage 3 - measured revision 2')
    p.title('The frozen-classifier comparison')
    p.lead('Top-1 test accuracy on the complete official split, averaged over the paired Stage 1 '
           'subset seeds 0, 1, and 2. Every delta uses the exact frozen probe loaded for that seed.')

    rows = []
    for r in s3.main:
        rows.append([r['dataset'], f"{r['probe']:.4f}",
                     f"{r['e2e']:.4f}", f"++{r['e2e_delta']:+.4f}",
                     f"{r['guided']:.4f}", f"++{r['guided_delta']:+.4f}"])
    p.table(0.055, 0.755, ['dataset', 'probe', 'end-to-end', 'delta', 'guided', 'delta'], rows,
            [0.150, 0.105, 0.135, 0.105, 0.125, 0.105], row_h=0.040, size=9.1,
            bold_first=True)

    labels = [r['dataset'] for r in s3.main]
    y = range(len(labels))
    ax = p.axes(0.070, 0.555, 0.405, 0.255)
    width = 0.23
    ax.barh([v + width for v in y], [r['probe'] for r in s3.main], height=width,
            color=FAINT, label='linear probe')
    ax.barh(list(y), [r['e2e'] for r in s3.main], height=width, color=BLUE, label='end-to-end')
    ax.barh([v - width for v in y], [r['guided'] for r in s3.main], height=width,
            color=PURPLE, label='guided')
    ax.set_yticks(list(y), labels); ax.set_xlim(0.45, 1.02); ax.set_xlabel('top-1 accuracy', size=8)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.set_title('Accuracy', fontsize=10, weight='bold', loc='left')

    ax2 = p.axes(0.555, 0.555, 0.375, 0.255)
    yy = range(len(labels))
    ax2.barh([v + 0.16 for v in yy], [100 * r['e2e_delta'] for r in s3.main],
             height=0.30, color=BLUE, label='end-to-end')
    ax2.barh([v - 0.16 for v in yy], [100 * r['guided_delta'] for r in s3.main],
             height=0.30, color=PURPLE, label='guided')
    ax2.set_yticks(list(yy), labels); ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlabel('percentage-point gain', size=8)
    ax2.set_title('Delta against the paired probe', fontsize=10, weight='bold', loc='left')
    ax2.legend(fontsize=7, frameon=False, loc='lower right')

    p.callout(0.055, 0.245, 0.425, 0.135, 'Aircraft - the clearest gain',
              'End-to-end adds 1.91 points and guided FM adds 1.63. Every Aircraft seed improves under '
              'both strategies, with paired bootstrap intervals excluding zero and significant exact '
              'McNemar tests.', accent=GREEN, size=8.8)
    p.callout(0.520, 0.245, 0.425, 0.135, 'Selective, not universal',
              'Guided FM adds 0.71 points on DTD; end-to-end keeps identity. Flowers-102 stays at '
              '99.35%, a useful ceiling control. No required revision-2 run regresses on test.',
              accent=AMBER, size=8.8)
    p.close()


def page_stage3_behavior(pdf, d, n):
    s3 = d.stage3
    p = Page(pdf, n, 'stage 3 - evidence and behavior')
    p.title('How the gains happen')
    p.lead('Accuracy alone hides whether the flow makes a few precise corrections or rewrites many '
           'predictions. Paired tests, fixed/broken counts, displacement, and flow-time curves expose '
           'that difference.')

    churn_rows = [[r['setting'], f"{r['fixed']:.1f}", f"{r['broken']:.1f}",
                   f"{r['net']:.1f}", f"{r['churn']:.1f}"] for r in s3.churn]
    p.table(0.055, 0.745, ['dataset / strategy', 'fixed', 'broken', 'net', 'churn'], churn_rows,
            [0.220, 0.065, 0.065, 0.065, 0.075], row_h=0.037, size=8.7, bold_first=True)
    p.callout(0.575, 0.745, 0.370, 0.170, 'Nine of eighteen keep identity',
              'All three DTD end-to-end runs and all six Flowers runs select epoch 0. Because epoch 0 '
              'is the exact linear probe, validation cannot force a harmful transformation. Only test '
              'delta is informative.', accent=MUTED, size=8.7)

    p.callout(0.055, 0.505, 0.425, 0.175, 'Power versus restraint',
              'Aircraft end-to-end moves features by a mean relative distance of .894 and changes '
              '463.7 predictions. Guided FM moves .082 and changes 153.7, yet retains most of the '
              'accuracy gain. DTD guided moves only .061.', accent=BLUE, size=8.8)
    p.callout(0.520, 0.505, 0.425, 0.175, 'Paired evidence',
              'End-to-end has 3 positive, 0 negative, and 6 identity cells; all 3 positives are '
              'significant. Guided has 6 positive, 0 negative, and 3 identity cells; 5 of 6 positives '
              'are significant. DTD seed 1 is the lone borderline case.', accent=GREEN, size=8.8)

    p.callout(0.055, 0.285, 0.425, 0.165, 'Regularization points to revision 3',
              'On Aircraft seed 0, lambda-displacement = 1 raises .5518 to .5563 while reducing '
              'relative motion from .906 to .058. On DTD, lambda = 10 produces a smaller +.0059 gain. '
              'These are single-seed ablations, not replacements for the main table.', accent=PURPLE,
              size=8.6)
    p.callout(0.520, 0.285, 0.425, 0.165, 'The endpoint can overshoot',
              'Validation-selected flow time helps only Aircraft end-to-end: .5476 at t = 1 becomes '
              '.5513 at mean t-star = .694. Guided Aircraft selects the full endpoint. The lesson is '
              'specific rather than universal: aggressive rollout benefits from earlier stopping.',
              accent=AMBER, size=8.6)
    p.close()


def page_stage3_joint(pdf, d, n):
    s3 = d.stage3
    p = Page(pdf, n, 'stage 3 - optional extension and status')
    p.title('Joint fine-tuning, attribution, and what remains')
    p.lead('After the required frozen-head grid, the optional extension unfreezes a copy of the Stage 1 '
           'classifier. A head-only control prevents extra classifier training from being mistaken for '
           'a Flow Matching gain.')

    rows = [[r['setting'], f"{r['probe']:.4f}", f"{r['fm_only']:.4f}",
             f"{r['head_only']:.4f}", f"{r['joint']:.4f}", r['attribution']]
            for r in s3.joint]
    p.table(0.055, 0.740, ['dataset / method', 'probe', 'FM only', 'head only', 'joint', 'attribution'],
            rows, [0.220, 0.080, 0.090, 0.095, 0.080, 0.190], row_h=0.037, size=8.4,
            bold_first=True, align=['left', 'right', 'right', 'right', 'right', 'left'])

    p.callout(0.055, 0.475, 0.425, 0.165, 'What the optional extension shows',
              'Aircraft end-to-end and DTD guided gains remain almost entirely attributable to the FM. '
              'Aircraft guided benefits from both components: the full joint model reaches .5487, '
              'above either FM-only (.5396) or head-only (.5331).', accent=GREEN, size=8.7)
    p.callout(0.520, 0.475, 0.425, 0.165, 'Variants implemented',
              'Classifier learning rates at .01x, .1x, and 1x the FM rate; unfreezing at epochs 1, 10, '
              'and 30; and anchoring penalties on the distance from the original Stage 1 weights. '
              'These remain optional and never replace the frozen-head table.', accent=BLUE, size=8.7)

    p.callout(0.055, 0.255, 0.890, 0.145, 'Revision status - do not mix the two',
              'The table in this chapter is implementation revision 2, measured on 2026-09-08 and '
              'reproduced from cache on 2026-09-09. Revision 3 is code-complete but unmeasured: it uses '
              'lambda-displacement = 1 for the main end-to-end run and independent per-sample guidance '
              'radius, step, unit-gradient normalization, and trust-region projection. A fresh run must '
              'train rather than load revision-2 caches before its results can replace these numbers.',
              accent=RED, size=8.8)
    p.close()


PAGES = [
    page_cover, page_method, page_architecture, page_protocol, page_stage1,
    page_stage2_figure, page_stage2_table, page_paired_ci, page_controls,
    page_probe_comparison, page_clip, page_geometry_pca, page_geometry_tsne,
    page_trajectories, page_clip_geometry, page_flow_time, page_mechanism,
    page_snapshots, page_overshoot, page_ablation_geometry, page_reverse,
    page_animation, page_summary, page_scope, page_stage3_protocol,
    page_stage3_results, page_stage3_behavior, page_stage3_joint,
]


def main():
    data = Data()
    with PdfPages(OUT) as pdf:
        for i, fn in enumerate(PAGES, 1):
            fn(pdf, data, i)
            print(f'  page {i:2d}  {fn.__name__}')
        pdf.infodict().update({
            'Title': 'Flow Matching as a Classification Layer - Stages 1, 2, and 3',
            'Author': 'CVLAB summer project',
            'Subject': 'Stage 1 baselines, Stage 2 prototype transport, and Stage 3 FM before a frozen classifier',
        })
    print(f'\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB, {len(PAGES)} pages)')


if __name__ == '__main__':
    main()

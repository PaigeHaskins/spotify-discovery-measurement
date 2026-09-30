"""
config.py
=========
One place for every definition and threshold used in the project.

WHY THIS FILE EXISTS
Measurement work lives or dies on its definitions ("what counts as a stream?",
"what counts as coming back?"). Keeping them here, instead of scattered through
the code, means:
  * a reviewer can see every analytical choice in one screen, and
  * changing a definition (e.g. a 14-day instead of 30-day window) re-flows
    through the SQL, the model, the causal analysis and the dashboard at once.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# All paths are built from the project folder so the code runs the same on any
# machine (your laptop, Colab, a reviewer's computer).
PROJECT_ROOT = Path(__file__).resolve().parent
SQL_DIR = PROJECT_ROOT / "sql"                 # the SQL transformation layer
RAW_DIR = PROJECT_ROOT / "data" / "raw"        # put your Spotify export here
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "spotify_master.csv"  # combined JSON
DB_PATH = PROJECT_ROOT / "data" / "spotify.db" # local SQLite "warehouse"
OUTPUT_DIR = PROJECT_ROOT / "outputs"          # JSON results + dashboard
DOCS_DIR = PROJECT_ROOT / "docs"               # GitHub Pages serves this folder
EXPERIMENT_DIR = PROJECT_ROOT / "experiment"   # self-experiment schedule

# Sample-data mode (python run_all.py --sample) writes to separate places so
# fake data can never overwrite or be confused with your real results.
SAMPLE_RAW_DIR = PROJECT_ROOT / "data" / "sample_raw"
SAMPLE_DB_PATH = PROJECT_ROOT / "data" / "sample.db"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sample"

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
# Spotify stores timestamps in UTC. Converting to your local time zone matters
# because "hour of day" and "which calendar day" are behavioral features:
# an 11pm listen in Boston is 3-4am UTC, which would land on the wrong day.
LOCAL_TIMEZONE = "America/New_York"

# ---------------------------------------------------------------------------
# Core metric definitions
# ---------------------------------------------------------------------------
# A play only counts as a real stream after 30 seconds. This mirrors the
# threshold Spotify uses for counting streams, so our "stream" means the same
# thing a Spotify analyst would mean.
MIN_PLAY_MS = 30_000

# A discovered artist "converts" if you come back and stream them again on a
# LATER calendar day within this many days. Same-day replays don't count:
# a new-listener promotion only matters if the listener returns.
RETENTION_WINDOW_DAYS = 30

# Left-censoring fix. Your export starts on some date, but you knew plenty of
# artists before that date. Without a burn-in, every favorite artist would look
# like a brand-new "discovery" in the first weeks of the data. We treat the
# first BURN_IN_DAYS as a warm-up: artists first seen then are "already known"
# and excluded from discovery metrics.
BURN_IN_DAYS = 90

# ---------------------------------------------------------------------------
# Sleep and background listening (set aside from the main analysis)
# ---------------------------------------------------------------------------
# Brown noise, sleep sounds, and all-night "background jazz" play while I'm
# asleep or not paying attention. Counting them would inflate discoveries and
# returns (a noise track "comes back" every night), so they are identified,
# reported separately on the dashboard, and removed from everything else.
#
# A play is BACKGROUND if it happened in a background SESSION, and an artist
# is BACKGROUND if most of their plays are sleep/noise content or happened in
# background sessions. All plays of background artists are set aside.

# A new session starts after a gap of this many minutes with nothing playing.
SESSION_GAP_MINUTES = 20

# Background session = long stretch of autoplay with almost no interaction.
# Awake listening is full of skips and clicks (over a third of all plays), so
# a session where under 5% of tracks involved a skip or click is someone who
# isn't touching the phone.
BACKGROUND_MAX_ACTION_RATE = 0.05
BACKGROUND_LONG_SESSION_MIN = 180      # any time of day: 3+ hours untouched
BACKGROUND_NIGHT_SESSION_MIN = 60      # or 1+ hour untouched, started at night
BACKGROUND_NIGHT_START_HOURS = (20, 4) # night = starts 8pm through 3:59am

# Artist-level rules.
BACKGROUND_SESSION_SHARE = 0.8  # 80%+ of the artist's plays were in background sessions
BACKGROUND_KEYWORD_SHARE = 0.6  # 60%+ of the artist's plays have a sleep/noise album or track title
# Lowercase phrases matched inside album and track titles. Kept specific
# ("brown noise", not "brown") so real artists aren't swept up; the 60% share
# rule protects artists with one sleepy song title.
BACKGROUND_KEYWORDS = (
    "noise", "sleep", "sleeping", " rain ", "rain sounds", "rainfall", "rainstorm", "thunder",
    "ocean waves", "ocean sounds", "sea waves", "waves for", "nature sounds", "binaural",
    "meditation", "meditative", "relaxing", "relaxation", "spa music", "background music",
    "bedtime", "soothing", "solfeggio", " hz", "jazz for", "sleep jazz", "study music",
)
# Manual overrides after reviewing outputs/background_artists.csv.
# Use the artist name exactly as it appears in that file.
BACKGROUND_ALWAYS = ()   # e.g. ("Some Jazz Trio",)
BACKGROUND_NEVER = ()    # e.g. ("Chelsea Cutler",)

# ---------------------------------------------------------------------------
# Discovery path (used by the causal analysis)
# ---------------------------------------------------------------------------
# Spotify's `reason_start` field records WHY a track started playing.
#   clickrow  -> you clicked that exact track in a list    => ACTIVE choice
#   trackdone -> the previous track ended and this one
#                followed (autoplay, radio, playlist order) => PASSIVE exposure
# Other values (playbtn, fwdbtn, appload, remote, ...) are ambiguous, so they
# are labeled "other" and left out of the passive-vs-active comparison rather
# than guessed at.
ACTIVE_REASONS = ("clickrow",)
PASSIVE_REASONS = ("trackdone",)

# ---------------------------------------------------------------------------
# Modeling / statistics
# ---------------------------------------------------------------------------
RANDOM_SEED = 42      # fixed seed = reproducible results
TEST_SHARE = 0.25     # the most recent 25% of discoveries are held out
N_BOOTSTRAP = 500     # resamples used for confidence intervals
PSM_CALIPER_SD = 0.2  # max matching distance, in SDs of the logit propensity
                      # score (0.2 is the standard rule of thumb, Austin 2011)
PSM_NEIGHBORS = 5     # each passive discovery is compared with up to 5 similar
                      # active ones: steadier estimates and better balance than
                      # a single match, especially for rare flags like offline
MIN_GROUP_SIZE = 30   # below this, an analysis reports "not enough data"
                      # instead of producing a noisy, misleading number

# ---------------------------------------------------------------------------
# Self-experiment (randomized A/B test on your own listening)
# ---------------------------------------------------------------------------
# Unit = one day. Days are grouped into consecutive PAIRS and a coin flip
# decides which day of each pair gets the treatment. Pairing guarantees equal
# arm sizes and cancels out slow drifts in your listening habits.
EXPERIMENT_TREATMENT_LABEL = "Smart Shuffle on"
EXPERIMENT_CONTROL_LABEL = "Normal listening"
EXPERIMENT_N_PAIRS = 14          # 14 pairs = 28 days
EXPERIMENT_START_DATE = None     # None = next Monday; or "2026-10-05"
EXPERIMENT_ALPHA = 0.05          # false-positive rate we accept
EXPERIMENT_POWER = 0.80          # chance to detect a real effect of the MDE size

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
AUTHOR_NAME = "Paige Haskins"
SHOW_ARTIST_NAMES = True  # set False to hide artist names in the dashboard

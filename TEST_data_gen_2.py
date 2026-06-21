import random
import uuid
import json
import os
from datetime import datetime, timedelta
import psycopg2
import pandas as pd

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "retail_cv_deep_dive"
DB_USER = "postgres"
DB_PASSWORD = "abc123"

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cursor = conn.cursor()
print("Database connection successful.")

# ── EXPORT MODE ────────────────────────────────────────────────────────────
# Options: "csv", "excel", "database"
EXPORT_MODE = "csv"
EXPORT_DIR  = "synthetic_data_export"

# ── CONFIGURATION ──────────────────────────────────────────────────────────

# Trimmed for a lighter demo dataset (was: DAYS_TO_GENERATE=30,
# WEEKDAY_CUSTOMERS=200, WEEKEND_CUSTOMERS=900). Row volume scales roughly
# linearly with days × customers, so this cuts total rows by ~85-90%
# relative to the original 30-day config while keeping every zone, camera,
# and use case (emotion / ReID / crowd / intrusion) represented.
DAYS_TO_GENERATE     = 181
START_DATE           = datetime(2026, 1, 1)
OPERATING_START_HOUR = 10
OPERATING_END_HOUR   = 22

WEEKDAY_CUSTOMERS = 60
WEEKEND_CUSTOMERS = 150   # 2.5x multiplier (was 4.5x) — still busier than weekdays, less extreme

# ── ZONE DEFINITIONS ───────────────────────────────────────────────────────

ZONES = {
    # Emotion
    "Atrium_Walkway"            : {"camera_id": "CAM_01",                    "type": "walkway",    "use_case": "emotion"},

    # ReID — Entrances
    "Mall_Main_Entrance"        : {"camera_id": "CAM_MALL_MAIN_ENTRANCE",    "type": "entrance",   "use_case": "reid"},
    "Mall_Entrance_North"       : {"camera_id": "CAM_MALL_ENTRANCE_NORTH",   "type": "entrance",   "use_case": "reid"},
    "Mall_Entrance_South"       : {"camera_id": "CAM_MALL_ENTRANCE_SOUTH",   "type": "entrance",   "use_case": "reid"},
    "Mall_Entrance_East"        : {"camera_id": "CAM_MALL_ENTRANCE_EAST",    "type": "entrance",   "use_case": "reid"},
    "Mall_Entrance_West"        : {"camera_id": "CAM_MALL_ENTRANCE_WEST",    "type": "entrance",   "use_case": "reid"},

    # ReID — Exits
    "Mall_Exit_Underground"     : {"camera_id": "CAM_MALL_EXIT_UNDERGROUND", "type": "exit",       "use_case": "reid"},
    "Mall_Exit_North"           : {"camera_id": "CAM_MALL_EXIT_NORTH",       "type": "exit",       "use_case": "reid"},
    "Mall_Exit_South"           : {"camera_id": "CAM_MALL_EXIT_SOUTH",       "type": "exit",       "use_case": "reid"},
    "Mall_Exit_East"            : {"camera_id": "CAM_MALL_EXIT_EAST",        "type": "exit",       "use_case": "reid"},
    "Mall_Exit_West"            : {"camera_id": "CAM_MALL_EXIT_WEST",        "type": "exit",       "use_case": "reid"},

    # ReID — Corridors
    "Corridor_L1"               : {"camera_id": "CAM_CORRIDOR_L1",           "type": "corridor",   "use_case": "reid"},
    "Corridor_Ground"           : {"camera_id": "CAM_CORRIDOR_GROUND",       "type": "corridor",   "use_case": "reid"},

    # ReID — Stores (entry + exit for dwell time)
    "Store_A_Entry"             : {"camera_id": "CAM_STORE_A_ENTRY",         "type": "store_entry","use_case": "reid"},
    "Store_A_Exit"              : {"camera_id": "CAM_STORE_A_EXIT",          "type": "store_exit", "use_case": "reid"},
    "Store_B_Entry"             : {"camera_id": "CAM_STORE_B_ENTRY",         "type": "store_entry","use_case": "reid"},
    "Store_B_Exit"              : {"camera_id": "CAM_STORE_B_EXIT",          "type": "store_exit", "use_case": "reid"},
    "Pharmacy_Entry"            : {"camera_id": "CAM_PHARMACY_ENTRY",        "type": "store_entry","use_case": "reid"},
    "Pharmacy_Exit"             : {"camera_id": "CAM_PHARMACY_EXIT",         "type": "store_exit", "use_case": "reid"},
    "Grocery_Entry"             : {"camera_id": "CAM_GROCERY_ENTRY",         "type": "store_entry","use_case": "reid"},
    "Grocery_Exit"              : {"camera_id": "CAM_GROCERY_EXIT",          "type": "store_exit", "use_case": "reid"},
    "Clothes_Entry"             : {"camera_id": "CAM_CLOTHES_ENTRY",         "type": "store_entry","use_case": "reid"},
    "Clothes_Exit"              : {"camera_id": "CAM_CLOTHES_EXIT",          "type": "store_exit", "use_case": "reid"},
    "Shoes_Entry"                : {"camera_id": "CAM_SHOES_ENTRY",          "type": "store_entry","use_case": "reid"},
    "Shoes_Exit"                 : {"camera_id": "CAM_SHOES_EXIT",           "type": "store_exit", "use_case": "reid"},
    "Gaming_Entry"               : {"camera_id": "CAM_GAMING_ENTRY",         "type": "store_entry","use_case": "reid"},
    "Gaming_Exit"                : {"camera_id": "CAM_GAMING_EXIT",          "type": "store_exit", "use_case": "reid"},
    "Kitchenware_Entry"          : {"camera_id": "CAM_KITCHENWARE_ENTRY",    "type": "store_entry","use_case": "reid"},
    "Kitchenware_Exit"           : {"camera_id": "CAM_KITCHENWARE_EXIT",     "type": "store_exit", "use_case": "reid"},

    # Crowd Analysis
    "Customer_Seating_Area"     : {"camera_id": "CAM_CAFE_MAIN_AREA",        "type": "cafe",       "use_case": "crowd"},

    # Intrusion Detection
    "Renovation_Area_Restricted": {"camera_id": "CAM_CLOSED_CORRIDOR",       "type": "restricted", "use_case": "intrusion"},
}

# ── ZONE ID MAPPING (fake IDs for export mode, real IDs for database mode) ─

ZONE_IDS = {zone_name: idx + 1 for idx, zone_name in enumerate(ZONES.keys())}

# Reverse lookup: zone_id -> zone_name (used as a safety net / validation only)
ZONE_ID_TO_NAME = {v: k for k, v in ZONE_IDS.items()}

# ── OBJECT COUNT RANGES BY ZONE TYPE (weekday baseline) ─────────────────────
# Weekend = 4-5x multiplier

OBJECT_COUNT_RANGES = {
    "entrance"   : (1, 3),
    "exit"       : (1, 3),
    "corridor"   : (2, 8),
    "store_entry": (1, 4),
    "store_exit" : (1, 4),
    "store"      : (5, 20),
    "cafe"       : (20, 40),
    "walkway"    : (3, 12),
    "restricted" : (0, 2),
}

def get_object_count_for_zone(zone_type, is_weekend):
    """
    Generate realistic object count for a zone.
    Weekday: base range. Weekend: 4-5x multiplier.
    """
    lo, hi = OBJECT_COUNT_RANGES.get(zone_type, (1, 5))
    base = random.randint(lo, hi)
    
    if is_weekend:
        multiplier = random.uniform(4.0, 5.0)
        return int(base * multiplier)
    return base

def get_or_create_zone_id(zone_name, camera_id, zone_type):
    """Used only in database mode — fetches or creates zone in PostgreSQL."""
    cursor.execute(
        "SELECT zone_id FROM zones WHERE zone_name = %s AND camera_id = %s",
        (zone_name, camera_id)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("""
        INSERT INTO zones (zone_name, camera_id, zone_type, floor, reid_enabled)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING zone_id
    """, (zone_name, camera_id, zone_type, "L1", False))
    conn.commit()
    return cursor.fetchone()[0]

# ── EMOTION DISTRIBUTION ───────────────────────────────────────────────────

EMOTION_DIST = {
    "neutral" : 0.55,
    "happy"   : 0.25,
    "sad"     : 0.08,
    "angry"   : 0.05,
    "fear"    : 0.04,
    "disgust" : 0.02,
    "surprise": 0.01,
}

EMOTION_MORNING = {
    "neutral" : 0.65,
    "happy"   : 0.10,
    "sad"     : 0.12,
    "angry"   : 0.05,
    "fear"    : 0.04,
    "disgust" : 0.03,
    "surprise": 0.01,
}

EMOTION_LUNCH = {
    "neutral" : 0.40,
    "happy"   : 0.40,
    "sad"     : 0.05,
    "angry"   : 0.04,
    "fear"    : 0.03,
    "disgust" : 0.02,
    "surprise": 0.06,
}

EMOTION_DINNER = {
    "neutral" : 0.35,
    "happy"   : 0.45,
    "sad"     : 0.05,
    "angry"   : 0.04,
    "fear"    : 0.03,
    "disgust" : 0.02,
    "surprise": 0.06,
}

def get_emotion_dist(hour):
    if 10 <= hour < 12:
        return EMOTION_MORNING
    elif 12 <= hour < 14:
        return EMOTION_LUNCH
    elif 18 <= hour <= 22:
        return EMOTION_DINNER
    else:
        return EMOTION_DIST

def sample_emotion(hour):
    dist  = get_emotion_dist(hour)
    keys  = list(dist.keys())
    probs = list(dist.values())
    return random.choices(keys, weights=probs, k=1)[0]

# ── DWELL TIME BY ZONE TYPE (seconds) ─────────────────────────────────────

DWELL_RANGES = {
    "entrance"   : (10,   30),
    "exit"       : (10,   30),
    "corridor"   : (20,   60),
    "store_entry": (5,    15),
    "store_exit" : (5,    15),
    "store"      : (120,  900),
    "cafe"       : (900,  2700),
    "walkway"    : (15,   45),
    "restricted" : (30,   50),
}

def get_dwell_time(zone_type):
    lo, hi = DWELL_RANGES.get(zone_type, (10, 60))
    return random.randint(lo, hi)

# ── MOVEMENT TYPE LOGIC ────────────────────────────────────────────────────

def get_movement_type(position_in_journey, journey_length, zone_type, dwell_secs):
    """
    Derive movement_type from position in journey and dwell time.
    - First zone in journey  → 'in'
    - Last zone in journey   → 'out'
    - Middle zones:
        if dwell_time exceeds threshold → 'dwell'
        otherwise                       → 'in'
    """
    DWELL_THRESHOLD = 60  # seconds

    if position_in_journey == 0:
        return "in"
    elif position_in_journey == journey_length - 1:
        return "out"
    else:
        return "dwell" if dwell_secs >= DWELL_THRESHOLD else "in"

# ── DIRECTION AND MOVEMENT_DIRECTION ──────────────────────────────────────

DIRECTIONS           = ["up", "down", "left", "right"]
MOVEMENT_DIRECTIONS  = ["up", "down", "left", "right"]

# Zone type → realistic dominant movement direction
ZONE_DIRECTION_BIAS = {
    "entrance"   : ["right", "down"],
    "exit"       : ["left", "up"],
    "corridor"   : ["right", "left"],
    "store_entry": ["down", "right"],
    "store_exit" : ["up", "left"],
    "walkway"    : ["right", "left"],
    "cafe"       : ["down", "right"],
    "restricted" : ["right", "down"],
}

def get_direction(zone_type, movement_type):
    """
    direction — crossing event direction (only on 'in' or 'out' movement_type).
    Returns None for 'dwell'.
    """
    if movement_type == "dwell":
        return None
    bias = ZONE_DIRECTION_BIAS.get(zone_type, DIRECTIONS)
    return random.choice(bias)

def get_movement_direction(zone_type):
    """
    movement_direction — continuous frame-to-frame movement inside zone.
    Always populated when in a zone.
    """
    bias = ZONE_DIRECTION_BIAS.get(zone_type, MOVEMENT_DIRECTIONS)
    # Slight randomness — 80% follows bias, 20% any direction
    if random.random() < 0.8:
        return random.choice(bias)
    return random.choice(MOVEMENT_DIRECTIONS)

# ── JOURNEY TEMPLATES ──────────────────────────────────────────────────────

ENTRANCES     = [z for z, v in ZONES.items() if v["type"] == "entrance"]
EXITS         = [z for z, v in ZONES.items() if v["type"] == "exit"]
MIDDLES       = [z for z, v in ZONES.items() if v["type"] in
                 ("corridor", "store_entry", "store_exit", "walkway", "cafe")]

# Built dynamically from ZONES so adding/removing a store only requires
# editing the ZONES dict above — every "X_Entry" zone is auto-paired with
# its matching "X_Exit" zone.
STORE_PAIRS = {
    zname: zname.replace("_Entry", "_Exit")
    for zname, meta in ZONES.items()
    if meta["type"] == "store_entry"
}
STORE_ENTRIES = list(STORE_PAIRS.keys())

# ── STORE VISIT DWELL TIME (in-store browsing time, NOT the entry/exit
# checkpoint crossing time) — used by the dashboard's "Store Dwell Time"
# panel via entry/exit timestamp subtraction, so these ranges define the
# GAP we insert between a store's Entry and Exit timestamps in the journey
# loop below. Differentiated by store category to be more realistic:
# quick transactional visits (Pharmacy) vs. long browsing visits (Grocery).
STORE_NAME_FROM_ENTRY = {e: e.replace("_Entry", "") for e in STORE_ENTRIES}

STORE_VISIT_DWELL_RANGES = {
    "Store_A"     : (120,  600),   # generic apparel/retail — moderate browse
    "Store_B"     : (120,  600),   # generic apparel/retail — moderate browse
    "Pharmacy"    : (60,   240),   # quick, transactional
    "Grocery"     : (300,  1200),  # long — many aisles, list-based shopping
    "Clothes"     : (240,  900),   # browsing + fitting room
    "Shoes"       : (180,  600),   # browsing + trying on
    "Gaming"      : (240,  900),   # browsing, demoing
    "Kitchenware" : (180,  720),   # moderate-long browsing
}

def get_store_visit_dwell(store_entry_zone_name):
    """Return a random in-store visit duration (seconds) for the store
    that store_entry_zone_name belongs to, using its category-specific
    range. Falls back to a generic moderate range for any unmapped store."""
    store_name = STORE_NAME_FROM_ENTRY.get(store_entry_zone_name, "")
    lo, hi = STORE_VISIT_DWELL_RANGES.get(store_name, (120, 600))
    return random.randint(lo, hi)

def generate_journey():
    """
    Generate a realistic zone sequence for one customer.
    MANDATORY: Start at entrance → visit at least 1 corridor/walkway AND
    1-2 distinct stores → exit.

    Structure:
    - Entrance
    - Corridor/Walkway (required)
    - 1-2 store visits (each an Entry + Exit pair, required, no repeats)
    - Exit

    NOTE: trimmed from the original 1-3 stores + 0-2 optional extra
    non-store zones. Each store visit contributes 2 rows (Entry + Exit) on
    top of dwell-time/timestamp logic, so capping at 2 stores and dropping
    the optional padding zones meaningfully shrinks rows-per-customer for
    a leaner demo dataset, while every journey still guarantees the
    "1 corridor/walkway + 1+ store visit" shape the dashboard relies on.
    """
    journey = [random.choice(ENTRANCES)]

    # Guarantee at least 1 corridor/walkway
    corridor_choices = [z for z in MIDDLES if ZONES[z]["type"] in ("corridor", "walkway")]
    journey.append(random.choice(corridor_choices))

    # Guarantee 1-2 distinct store visits (each Entry+Exit pair)
    n_stores = random.randint(1, min(2, len(STORE_ENTRIES)))
    store_choices = random.sample(STORE_ENTRIES, k=n_stores)
    for store_entry in store_choices:
        journey.append(store_entry)
        journey.append(STORE_PAIRS[store_entry])

    journey.append(random.choice(EXITS))
    return journey

# ── HELPERS ────────────────────────────────────────────────────────────────

def is_weekend(date):
    return date.weekday() >= 5

def random_operating_time(date, hour_start=None, hour_end=None):
    start  = hour_start or OPERATING_START_HOUR
    end    = hour_end   or OPERATING_END_HOUR
    hour   = random.randint(start, end - 1)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(date.year, date.month, date.day, hour, minute, second)

def random_bbox():
    x1 = random.randint(50, 800)
    y1 = random.randint(50, 500)
    x2 = x1 + random.randint(60, 180)
    y2 = y1 + random.randint(150, 400)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

# ── IN-MEMORY STORAGE FOR EXPORT ──────────────────────────────────────────

detection_rows = []
emotion_rows   = []

# ── INSERT HELPER ──────────────────────────────────────────────────────────

def insert_detection_event(
    timestamp, camera_id, zone_id, object_class, confidence,
    model_name="PeopleNet",
    reid_id=None, reid_type=None,
    movement_type=None, direction=None, movement_direction=None,
    dwell_time=None, previous_zone_id=None,
    object_count_in_zone=None, journey_end=False, zone_type=None,
    zone_name=None
):
    """
    Appends a row to detection_rows in-memory (export mode)
    or writes directly to PostgreSQL (database mode).
    Returns a fake sequential ID for export mode.
    
    LOGIC:
    - zone_name is REQUIRED — every row must carry its own zone name at insertion time.
      Falls back to a reverse lookup via ZONE_ID_TO_NAME only as a safety net; if that
      also fails, raises an error rather than silently writing a null.
    - object_count_in_zone is ALWAYS populated based on zone_type and is_weekend
    - dwell_time is ONLY populated when movement_type == "dwell"
    - direction is ONLY populated when movement_type in ["in", "out"]
    """
    fake_id = len(detection_rows) + 1
    
    # Determine if weekend for realistic object counts
    is_wknd = is_weekend(timestamp.date()) if timestamp else False
    
    # Auto-generate object_count_in_zone if not provided
    if object_count_in_zone is None and zone_type is not None:
        object_count_in_zone = get_object_count_for_zone(zone_type, is_wknd)
    elif object_count_in_zone is None:
        object_count_in_zone = 1
    
    # CRITICAL: dwell_time ONLY for dwell movement_type
    if movement_type != "dwell":
        dwell_time = None

    # CRITICAL: zone_name must always be resolved — never silently null.
    # 1) Use the explicitly passed zone_name if given (preferred — set at call site).
    # 2) Fall back to reverse lookup from zone_id.
    # 3) If still missing, fail loudly instead of writing a null row.
    resolved_zone_name = zone_name
    if resolved_zone_name is None:
        resolved_zone_name = ZONE_ID_TO_NAME.get(zone_id)
    if resolved_zone_name is None:
        raise ValueError(
            f"zone_name could not be resolved for zone_id={zone_id}, camera_id={camera_id}. "
            f"Every detection_event row MUST have a zone_name."
        )

    detection_rows.append({
        "unique_detection_id" : fake_id,
        "timestamp"           : timestamp,
        "frame_number"        : random.randint(1, 1000),
        "camera_id"           : camera_id,
        "zone_id"             : zone_id,
        "zone_name"           : resolved_zone_name,
        "zone_type"           : zone_type,
        "polygon_zone_id"     : zone_id,
        "model_name"          : model_name,
        "object_class"        : object_class,
        "confidence"          : float(confidence),
        "bbox"                : json.dumps(random_bbox()),
        "track_id"            : f"{camera_id}_T{random.randint(1,999):03d}",
        "reid_id"             : reid_id,
        "reid_type"           : reid_type,
        "previous_zone_id"    : previous_zone_id,
        "movement_type"       : movement_type,
        "direction"           : direction,
        "movement_direction"  : movement_direction,
        "dwell_time"          : dwell_time,
        "object_count_in_zone": object_count_in_zone,
        "processing_latency"  : round(random.uniform(10, 80), 2),
        "journey_end"         : journey_end,
    })

    return fake_id


def insert_emotion_event(det_id, emotion, confidence):
    emotion_rows.append({
        "unique_detection_id" : det_id,
        "emotion"             : emotion,
        "confidence"          : round(confidence, 4),
        "bbox"                : json.dumps(random_bbox()),
    })

# ── GENERATION FUNCTIONS ───────────────────────────────────────────────────

def generate_emotion_data(date, zone_id, camera_id):
    """Generate emotion detection events for one day in Atrium_Walkway."""
    n = random.randint(300, 600) if is_weekend(date) else random.randint(80, 150)
    zone_type = "walkway"
    zone_name = "Atrium_Walkway"

    for _ in range(n):
        ts         = random_operating_time(date)
        emotion    = sample_emotion(ts.hour)
        confidence = round(random.uniform(0.55, 0.99), 4)
        mv_dir     = get_movement_direction(zone_type)

        det_id = insert_detection_event(
            timestamp          = ts,
            camera_id          = camera_id,
            zone_id            = zone_id,
            object_class       = "face",
            confidence         = confidence,
            model_name         = "DeepFace",
            movement_type      = "in",
            direction          = None,
            movement_direction = mv_dir,
            dwell_time         = None,
            zone_type          = zone_type,
            zone_name          = zone_name,
        )
        insert_emotion_event(det_id, emotion, confidence)

    return n


def generate_reid_data(date):
    """Generate ReID customer journeys for one day."""
    n_customers  = WEEKEND_CUSTOMERS if is_weekend(date) else WEEKDAY_CUSTOMERS
    n_customers += random.randint(-20, 20)

    for _ in range(n_customers):
        reid_id  = f"REID_{str(uuid.uuid4())[:8].upper()}"
        journey  = generate_journey()
        ts       = random_operating_time(date)
        jlen     = len(journey)

        for i, zone_name in enumerate(journey):
            meta         = ZONES[zone_name]
            zone_id      = ZONE_IDS[zone_name]
            zone_type    = meta["type"]
            dwell_secs   = get_dwell_time(zone_type)

            reid_type    = "create" if i == 0 else ("delete" if i == jlen - 1 else "match")
            mv_type      = get_movement_type(i, jlen, zone_type, dwell_secs)
            direction    = get_direction(zone_type, mv_type)
            mv_dir       = get_movement_direction(zone_type)
            is_last      = (i == jlen - 1)
            prev_zone_id = ZONE_IDS.get(journey[i - 1]) if i > 0 else None
            
            # CRITICAL: dwell_time ONLY if movement_type == "dwell"
            final_dwell_time = float(dwell_secs) if mv_type == "dwell" else None

            insert_detection_event(
                timestamp          = ts,
                camera_id          = meta["camera_id"],
                zone_id            = zone_id,
                object_class       = "person",
                confidence         = round(random.uniform(0.6, 0.99), 4),
                reid_id            = reid_id,
                reid_type          = reid_type,
                movement_type      = mv_type,
                direction          = direction,
                movement_direction = mv_dir,
                dwell_time         = final_dwell_time,
                previous_zone_id   = prev_zone_id,
                journey_end        = is_last,
                zone_type          = zone_type,
                zone_name          = zone_name,
            )

            # ── Timestamp advance to the next zone in the journey ──
            # Default: dwell_secs (checkpoint/zone time) + a small random
            # walking gap (5-30s) to the next zone.
            # SPECIAL CASE: if this zone is a store_entry, the *next* zone
            # in the journey is always its matching store_exit (see
            # generate_journey() — stores are always appended as adjacent
            # Entry/Exit pairs). In that case we advance by the store's
            # category-specific in-store visit duration instead, so the
            # Exit timestamp minus this Entry timestamp reflects realistic
            # browsing time (this is what the dashboard's "Store Dwell
            # Time" panel measures via timestamp subtraction).
            if zone_type == "store_entry":
                ts += timedelta(seconds=get_store_visit_dwell(zone_name))
            else:
                ts += timedelta(seconds=dwell_secs + random.randint(5, 30))

    return n_customers


def generate_crowd_data(date, zone_id, camera_id):
    """Generate crowd count data for Customer_Seating_Area every 5 minutes."""
    rows         = 0
    current_time = datetime(date.year, date.month, date.day, OPERATING_START_HOUR, 0, 0)
    end_time     = datetime(date.year, date.month, date.day, OPERATING_END_HOUR, 0, 0)
    zone_type    = "cafe"
    zone_name    = "Customer_Seating_Area"

    while current_time < end_time:
        hour = current_time.hour

        if 12 <= hour < 14 or 18 <= hour < 22:
            base_count = random.randint(20, 40)
        elif 10 <= hour < 12:
            base_count = random.randint(2, 10)
        else:
            base_count = random.randint(5, 18)

        if is_weekend(date):
            base_count = min(40, int(base_count * random.uniform(1.5, 2.0)))

        for _ in range(random.randint(1, 3)):
            insert_detection_event(
                timestamp            = current_time + timedelta(seconds=random.randint(0, 59)),
                camera_id            = camera_id,
                zone_id              = zone_id,
                object_class         = "person",
                confidence           = round(random.uniform(0.6, 0.99), 4),
                movement_type        = "dwell",
                direction            = None,
                movement_direction   = get_movement_direction(zone_type),
                dwell_time           = float(random.randint(30, 120)),
                object_count_in_zone = base_count,
                zone_type            = zone_type,
                zone_name            = zone_name,
            )
            rows += 1

        current_time += timedelta(minutes=5)

    return rows


def generate_intrusion_data(date, zone_id, camera_id, zone_type="restricted", zone_name="Renovation_Area_Restricted"):
    """
    Generate intrusion events — ~2-5 per week, before operating hours (morning: 7-9am) 
    and after operating hours (night: 23:00-23:59).
    movement_type: 'in' at start, 'dwell' in middle, 'out' at end.
    dwell_time: ONLY for 'dwell' frames.
    """
    rows = 0

    # ~3 days per week have an intrusion
    if random.random() > 3 / 7:
        return 0

    n_events = random.randint(1, 2)

    for _ in range(n_events):
        # 50/50 chance of morning or night intrusion
        if random.random() < 0.5:
            # Morning intrusion (before operating hours: 7-9am)
            hour   = random.randint(7, 9)
        else:
            # Night intrusion (after operating hours: 23:00-23:59)
            hour   = 23
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts     = datetime(date.year, date.month, date.day, hour, minute, second)

        duration = random.randint(30, 50)
        n_frames = duration * 2  # ~2 detections per second

        for f in range(n_frames):
            frame_ts = ts + timedelta(seconds=f * 0.5)
            
            # movement_type: 'in' at start, 'dwell' in middle, 'out' at end
            if f == 0:
                mv_type   = "in"
                direction = get_direction(zone_type, "in")
                dwell_secs = None
            elif f == n_frames - 1:
                mv_type   = "out"
                direction = get_direction(zone_type, "out")
                dwell_secs = None
            else:
                mv_type   = "dwell"
                direction = None
                dwell_secs = float(duration)

            insert_detection_event(
                timestamp          = frame_ts,
                camera_id          = camera_id,
                zone_id            = zone_id,
                object_class       = "person",
                confidence         = round(random.uniform(0.5, 0.95), 4),
                movement_type      = mv_type,
                direction          = direction,
                movement_direction = get_movement_direction(zone_type),
                dwell_time         = dwell_secs,
                zone_type          = zone_type,
                zone_name          = zone_name,
            )
            rows += 1

        time_period = "Morning (pre-opening)" if ts.hour < 10 else "Night (post-closing)"
        print(f"    Intrusion at {ts.strftime('%H:%M:%S')} ({time_period}) — {duration}s duration")

    return rows

# ── MAIN GENERATION LOOP ───────────────────────────────────────────────────

os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"Generating {DAYS_TO_GENERATE} days of synthetic data from {START_DATE.date()}...\n")

total_emotion   = 0
total_reid      = 0
total_crowd     = 0
total_intrusion = 0

for day_offset in range(DAYS_TO_GENERATE):
    date      = START_DATE + timedelta(days=day_offset)
    day_label = date.strftime("%Y-%m-%d (%A)")
    print(f"Day {day_offset+1:02d}/{DAYS_TO_GENERATE}: {day_label}")

    n = generate_emotion_data(date, ZONE_IDS["Atrium_Walkway"], "CAM_01")
    total_emotion += n
    print(f"  Emotion     : {n} detections")

    n = generate_reid_data(date)
    total_reid += n
    print(f"  ReID        : {n} customer journeys")

    n = generate_crowd_data(date, ZONE_IDS["Customer_Seating_Area"], "CAM_CAFE_MAIN_AREA")
    total_crowd += n
    print(f"  Crowd       : {n} rows")

    meta_restricted = ZONES["Renovation_Area_Restricted"]
    n = generate_intrusion_data(
        date, 
        ZONE_IDS["Renovation_Area_Restricted"], 
        meta_restricted["camera_id"],
        zone_type=meta_restricted["type"],
        zone_name="Renovation_Area_Restricted",
    )
    total_intrusion += n
    if n > 0:
        print(f"  Intrusion   : {n} detection rows")

# ── EXPORT OR WRITE TO DATABASE ────────────────────────────────────────────

df_detections = pd.DataFrame(detection_rows)
df_emotions   = pd.DataFrame(emotion_rows)

print(f"\nTotal rows — detection_events: {len(df_detections)}, emotion_events: {len(df_emotions)}")
print(f"\nSample detection_events:\n{df_detections.head(5).to_string()}")
print(f"\nEmotion distribution:\n{df_emotions['emotion'].value_counts()}")
print(f"\nmovement_type distribution:\n{df_detections['movement_type'].value_counts()}")
print(f"\nmovement_direction distribution:\n{df_detections['movement_direction'].value_counts()}")
print(f"\ndirection distribution:\n{df_detections['direction'].value_counts(dropna=False)}")
print(f"\njourney_end distribution:\n{df_detections['journey_end'].value_counts()}")

# ── VALIDATION: zone_name, object_count_in_zone and dwell_time ─────────────
print(f"\nzone_name stats:")
print(f"  Null count: {df_detections['zone_name'].isna().sum()} / {len(df_detections)}")
print(f"  Unique zone_names: {df_detections['zone_name'].nunique()}")
assert df_detections['zone_name'].isna().sum() == 0, "FAIL: zone_name contains nulls!"
print(f"  ✓ PASS: No null zone_name values")

print(f"\nobject_count_in_zone stats:")
print(f"  Non-null count: {df_detections['object_count_in_zone'].notna().sum()} / {len(df_detections)}")
print(f"  Min: {df_detections['object_count_in_zone'].min()}, Max: {df_detections['object_count_in_zone'].max()}")
print(f"  Mean: {df_detections['object_count_in_zone'].mean():.2f}")

print(f"\ndwell_time stats:")
print(f"  Non-null count: {df_detections['dwell_time'].notna().sum()} / {len(df_detections)}")
print(f"  Rows with movement_type='dwell': {(df_detections['movement_type'] == 'dwell').sum()}")
print(f"  Rows with movement_type='dwell' AND dwell_time is NULL: {((df_detections['movement_type'] == 'dwell') & (df_detections['dwell_time'].isna())).sum()}")
print(f"  Rows with movement_type!='dwell' AND dwell_time is NOT NULL: {((df_detections['movement_type'] != 'dwell') & (df_detections['dwell_time'].notna())).sum()}")

if EXPORT_MODE == "csv":
    det_path = os.path.join(EXPORT_DIR, "detection_events.csv")
    emo_path = os.path.join(EXPORT_DIR, "emotion_events.csv")
    df_detections.to_csv(det_path, index=False)
    df_emotions.to_csv(emo_path, index=False)
    print(f"\nExported to CSV:")
    print(f"  {det_path}")
    print(f"  {emo_path}")

elif EXPORT_MODE == "excel":
    excel_path = os.path.join(EXPORT_DIR, "synthetic_data.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_detections.to_excel(writer, sheet_name="detection_events", index=False)
        df_emotions.to_excel(writer, sheet_name="emotion_events",     index=False)
    print(f"\nExported to Excel: {excel_path}")

elif EXPORT_MODE == "database":
    print("\nWriting to PostgreSQL...")
    print("NOTE: zone_name is NOT written to detection_events table — the table uses")
    print("      zone_id as a foreign key to the zones table, where zone_name lives.")
    print("      zone_name is kept in the in-memory/CSV/Excel export for readability/QA only.")

    # Resolve real zone IDs from database
    real_zone_ids = {}
    for zone_name, meta in ZONES.items():
        real_zone_ids[ZONE_IDS[zone_name]] = get_or_create_zone_id(
            zone_name, meta["camera_id"], meta["type"]
        )

    def resolve(fake_id):
        return real_zone_ids.get(fake_id) if fake_id is not None else None

    det_id_map = {}  # fake_id → real_id for emotion_events FK

    for row in detection_rows:
        cursor.execute("""
            INSERT INTO detection_events (
                timestamp, frame_number, camera_id, zone_id, polygon_zone_id,
                model_name, object_class, confidence, bbox, track_id,
                reid_id, reid_type, previous_zone_id,
                movement_type, direction, movement_direction,
                dwell_time, object_count_in_zone, processing_latency,
                journey_end
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s
            ) RETURNING unique_detection_id
        """, (
            row["timestamp"],
            row["frame_number"],
            row["camera_id"],
            resolve(row["zone_id"]),
            resolve(row["polygon_zone_id"]),
            row["model_name"],
            row["object_class"],
            row["confidence"],
            row["bbox"],
            row["track_id"],
            row["reid_id"],
            row["reid_type"],
            resolve(row["previous_zone_id"]),
            row["movement_type"],
            row["direction"],
            row["movement_direction"],
            row["dwell_time"],
            row["object_count_in_zone"],
            row["processing_latency"],
            row["journey_end"],
        ))
        real_id = cursor.fetchone()[0]
        det_id_map[row["unique_detection_id"]] = real_id

    for row in emotion_rows:
        real_det_id = det_id_map.get(row["unique_detection_id"])
        if real_det_id:
            cursor.execute("""
                INSERT INTO emotion_events (unique_detection_id, emotion, confidence, bbox)
                VALUES (%s, %s, %s, %s)
            """, (real_det_id, row["emotion"], row["confidence"], row["bbox"]))

    conn.commit()
    print(f"Written to database — {len(detection_rows)} detection rows, {len(emotion_rows)} emotion rows.")

print(f"""
{'='*50}
Generation complete.
  Emotion events   : {total_emotion}
  ReID journeys    : {total_reid} customers
  Crowd rows       : {total_crowd}
  Intrusion rows   : {total_intrusion}
  Export mode      : {EXPORT_MODE}
{'='*50}
""")
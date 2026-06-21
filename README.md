# retail-deep-dive-computer-vision

> A full-stack computer vision intelligence system for retail and mall environments — from model inference to business analytics.

Built as part of the **[Inside AI with Kayana](https://www.youtube.com/@InsideAIwithKayana)** YouTube channel. This project demonstrates how a single camera network can produce intelligent, actionable data for marketing, operations, and security teams in retail environments.

---

## What This Project Does

This system layers five computer vision technologies on top of a standard mall camera network:

| Technology | What It Produces |
|---|---|
| **Emotion Classification** | Aggregate emotional state of customers by zone and hour |
| **ReID + Customer Journey** | Cross-camera identity matching and Sankey diagram flow visualisation |
| **Security Notifications** | Automated alerts for intrusion detection and restricted zone breaches |
| **Notifications** | Staffing level & Intrusion alerts  |
| **BI Dashboard** | Unified Metabase dashboard connecting all data streams |

---

## Demo

> 📹 Video walkthrough coming soon on [Inside AI with Kayana](https://www.youtube.com/@InsideAIwithKayana)

<!-- Add output video screenshot here -->
<!-- Add Metabase dashboard screenshot here -->

---

## Tech Stack

| Layer | Tool |
|---|---|
| People & Face Detection | [NVIDIA PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) (DetectNet_v2, ONNX) |
| Emotion Classification | [DeepFace](https://github.com/serengil/deepface) |
| Zone Configuration | [Roboflow PolygonZone](https://polygonzone.roboflow.com/) |
| Database | PostgreSQL (local) |
| Dashboard | [Metabase](https://www.metabase.com/) (local) |
| Demo Environment | Jupyter Notebook (local) |
| Language | Python 3.13 |

---

## Project Structure

```
retail-deep-dive-computer-vision/
├── models/
│   └── peoplenet.onnx
├── videos/
│   ├── input.mp4
│   └── output.mp4
├── retail_cv_pipeline.ipynb
└── README.md
```

### Notebook Structure

```
── SETUP
   Cell 1   Imports
   Cell 2A  Configuration
   Cell 3   PostgreSQL Connection & Schema (5 tables)
   Cell 4   Zones & Rules Setup
   Cell 5   Global Functions

── EMOTION CLASSIFICATION
   Cell 6   Load Models
   Cell 7   Zone Verification
   Cell 8   Single Frame Verification
   Cell 9   Inference Loop
   Cell 10  Verify Results & Charts
   Cell 11  Close Connections
```

---

## Database 

Five PostgreSQL tables store all CV outputs:

```
zones              ← polygon zone definitions per camera
rules              ← notification trigger rules
detection_events   ← every PeopleNet detection with zone, class, bbox
emotion_events     ← DeepFace results linked to detection_events
notifications      ← triggered alerts linked to rules and detections
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/elmayana/retail-deep-dive-computer-vision.git
cd retail-deep-dive-computer-vision
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
install requirements.txt file here
```bash
pip install opencv-python deepface psycopg2-binary numpy matplotlib ipywidgets
```

### 4. Set up PostgreSQL

- Install PostgreSQL from https://www.postgresql.org/download/
- Create a database called `mall_cv`
- Update `DB_PASSWORD` in Cell 2 with your postgres password

### 5. Download PeopleNet

- Create a free NVIDIA NGC account at https://ngc.nvidia.com
- Download `deployable_quantized_onnx_v2.6.3` from the [PeopleNet NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)
- Rename the file to `peoplenet.onnx` and place it in the `models/` folder

### 6. Configure zone coordinates

- Extract a frame from your input video using Cell 7
- Upload `zone_setup_frame.jpg` to [Roboflow PolygonZone](https://polygonzone.roboflow.com/)
- Draw your zones and paste the coordinates into Cell 4

---

## How to Run

> 📹 Full walkthrough: [Inside AI with Kayana](https://www.youtube.com/@InsideAIwithKayana) *(video coming soon)*

Run the notebook cells in order from Cell 1 through Cell 11.

---

## References

- [NVIDIA PeopleNet on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)
- [NVIDIA TAO DetectNet_v2 Documentation](https://docs.nvidia.com/tao/tao-toolkit-archive/5.2.0/text/object_detection/detectnet_v2.html)
- [DeepFace GitHub](https://github.com/serengil/deepface)
- [Roboflow PolygonZone](https://polygonzone.roboflow.com/)
- [Pexels — Royalty Free Retail Footage](https://www.pexels.com)

---

## License

This project is licensed under the [GPL-3.0 License](https://github.com/elmayana/retail-deep-dive-computer-vision#GPL-3.0-1-ov-file).
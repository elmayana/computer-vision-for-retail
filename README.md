# Computer Vision for Retail

> A computer vision intelligence system for retail and mall environments — from model inference to business analytics.

<!-- Add output video screenshot here -->
<!-- Add Metabase dashboard screenshot here -->

Built as part of the **[Inside AI with Kayana](https://www.youtube.com/@InsideAIwithKayana)** YouTube channel. This project shows you how a standard camera network can be turned into an intelligent data system — one that feeds marketing, operations, and security teams with real, actionable insights.

---

## What This Project Does

This system layers five computer vision technologies on top of a standard mall camera network:

| Technology | What It Produces |
|---|---|
| **Emotion Classification** | Capture emotional state of shoppers |
| **Multi-Camera ReID** | Cross-camera identity matching and customer journey visualisation |
| **Crowd Analysis** | Zone-level occupancy counts and trend tracking |
| **Intrusion Detection** | Automated alerts when restricted zones are breached |
| **BI Dashboard** | Unified Streamlit dashboard connecting all data streams |
| **Real-Time Notifications** | Telegram alerts powered by Postgres LISTEN/NOTIFY |

---

## Demo

> 📹 Video walkthrough coming soon on [Inside AI with Kayana](https://www.youtube.com/@InsideAIwithKayana)


---

## Tech Stack

| Layer | Tool |
|---|---|
| People Detection | [NVIDIA PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) (ONNX) |
| Re-Identification | [OSNet via torchreid](https://github.com/KaiyangZhou/deep-person-reid) |
| Emotion Classification | [DeepFace](https://github.com/serengil/deepface) |
| Zone Configuration | [Roboflow PolygonZone](https://polygonzone.roboflow.com/) |
| Object Tracking | [ByteTrack](https://github.com/ifzhang/ByteTrack) |
| Database | PostgreSQL (local) |
| Dashboard | Streamlit |
| Notifications | Telegram Bot API + Postgres LISTEN/NOTIFY |
| Demo Environment | Jupyter Notebook (VS Code) |
| Language | Python 3.13 |

---

## Project Structure

```
retail-deep-dive-computer-vision/
├── .streamlit/
├── models/
│   └── peoplenet.onnx
├── Input_Videos/
│   ├── Emotion_Classification
│   ├── ReID
│   ├── Crowd_Analysis
│   ├── Intrusion_Detection
├── Output_Videos/
│   ├── Emotion_Classification
│   ├── ReID
│   ├── Crowd_Analysis
│   ├── Intrusion_Detection
├── zone_setup_frames/
├── alert_crops/
├── synthetic_data_export/
├── demo_code.ipynb       ← main notebook (inference pipeline)
├── dashboard_combined_app.py            ← Streamlit analytics dashboard with real and synthetic data combined
├── dashboard_real_app.py            ← Streamlit analytics dashboard with real data only
├── notification_service.py        ← real-time Telegram alert service
└── README.md
```


## Database 

Five PostgreSQL tables store all CV outputs:

```
zones              ← polygon zone definitions per camera
rules              ← notification trigger rules
detection_events   ← all detection metadata from every use case
emotion_events     ← DeepFace results linked to detection_events
notifications      ← triggered alerts linked to rules and detections
reid_features      ← feature vectors for multicamera reidentification
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/elmayana/computer-vision-for-retail.git
cd computer-vision-for-retail
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies and pytorch
install requirements.txt file here
```bash
pip install -r requirements.txt
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX 
```
where [XXX] is the CUDA version (check using nvidia-smi command)

### 4. Set up PostgreSQL

- Install PostgreSQL from https://www.postgresql.org/download/
- Create a database
- Update `DB_PASSWORD` and `DB_USERNAME` with your password and username in demo-code.ipynb
- in the .streamlit folder, create a secrets.toml file
```python
[connections.postgresql]
dialect = "postgresql"
host = "localhost"
port = "your_port"
database = "your_database"
username = "your_username"
password = "your_password"
```

### 5. Download PeopleNet

- Download `deployable_quantized_onnx_v2.6.3` from the [PeopleNet NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)
- Place it in the `models/` folder

### 6. Download Input Videos

- Emotion Classification videos: (1) https://www.pexels.com/video/people-walking-inside-the-mall-4750042/   (2)https://www.pexels.com/video/toronto-canada-ontario-downton-14365388/
- Download the ReID input videos here: https://www.dropbox.com/scl/fi/pbip7ihu80owrcs8tfibp/ReID_input_videos.zip?rlkey=7jtpmtxtlranclrbf2snsdjjb&st=193sxvn3&dl=0
- Crowd Analysis video: https://www.pexels.com/video/time-lapse-video-of-people-inside-the-book-shop-4473910/
- Intrusion Detection video: https://www.pexels.com/video/elegant-man-walking-in-corridor-11903990/

---

## How to Run

> 📹 Full walkthrough: [Inside AI with Kayana](https://www.youtube.com/@InsideAIwithKayana) *(video coming soon)*

Run the notebook cells 

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
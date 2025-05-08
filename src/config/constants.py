"""
Configuration constants for the exam question classification system.
"""

CURRICULUM_CATEGORIES = [
    "Fundamental Physics",
    "Atomic and Nuclear Structure",
    "Production of Kilovoltage X-ray Beams",
    "Production of Megavoltage Radiation Beams",
    "Radiation Interactions",
    "Radiation Quantities and Units",
    "Radiation Measurement and Calibration",
    "Photon Beam Characteristics and Dosimetry",
    "Electron Beam Characteristics and Dosimetry",
    "Intensity Modulated Radiation Therapy (IMRT)",
    "Prescribing, Reporting, and Evaluating Radiotherapy Treatment Plans",
    "Imaging Fundamentals",
    "Simulation, Motion Management and Treatment Verification",
    "Clinical Brachytherapy",
    "Brachytherapy QA",
    "Advanced Treatment Planning and Special Procedures",
    "Particle Therapy",
    "Stereotactic Radiosurgery / Stereotactic Body Radiotherapy (SRS/SBRT)",
    "Quality Assurance in Radiation Oncology",
    "Radiation Protection and Shielding",
    "Safety and Incidents"
]

# Model configuration
DEFAULT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1

# Classification thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# File paths
MODEL_OUTPUT_DIR = "./transformer_model"
LOG_DIR = "./logs"
RAW_DATA_DIR = "./data/raw"
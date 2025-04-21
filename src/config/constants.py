"""
Configuration constants for the exam question classification system.
"""

# Define AAPM/ASTRO curriculum categories
CURRICULUM_CATEGORIES = {
    "Fundamental Physics": [
        "SI Units and Prefixes",
        "Classical Mechanics",
        "Electricity and Magnetism",
        "Basic Special Relativity"
    ],
    "Atomic and Nuclear Structure": [
        "The Atom",
        "Basic Radioactivity and Decay"
    ],
    "Radiation Production": [
        "Kilovoltage X-ray Beams",
        "Megavoltage Radiation Beams",
        "Linear Accelerators"
    ],
    "Radiation Interactions": [
        "Directly and Indirectly Ionizing Radiation",
        "X and γ Ray Interactions with Matter",
        "Charged Particle Interactions",
        "X-Ray Attenuation"
    ],
    "Radiation Quantities and Units": [
        "Radiation Quantities",
        "Kerma, Dose, and Exposure Relationships"
    ],
    "Radiation Measurement": [
        "Phantoms",
        "Ionization Chambers",
        "Dosimetry Instrumentation",
        "Beam Calibration"
    ],
    "Beam Characteristics": [
        "Photon Beam Properties",
        "Electron Beam Properties",
        "Monitor Unit Calculations",
        "Special Dosimetry Considerations"
    ],
    "Treatment Planning": [
        "Photon Treatment Planning",
        "Electron Treatment Planning",
        "Intensity Modulated Radiation Therapy",
        "Prescription and Plan Evaluation"
    ],
    "Imaging": [
        "X-ray Radiography",
        "Computed Tomography",
        "Nuclear Medicine Imaging",
        "Magnetic Resonance Imaging",
        "Ultrasound",
        "Medical Informatics"
    ],
    "Patient Management": [
        "Simulation",
        "Treatment Verification",
        "Motion Management"
    ],
    "Brachytherapy": [
        "Radioactive Sources",
        "Brachytherapy Dosimetry",
        "Delivery Techniques",
        "Site-Specific Applications",
        "Brachytherapy QA"
    ],
    "Special Procedures": [
        "Total Body Irradiation",
        "Total Skin Electron Therapy",
        "Intraoperative Radiotherapy",
        "Radiopharmaceutical Therapy",
        "Adaptive Radiotherapy"
    ],
    "Advanced Technologies": [
        "Particle Therapy",
        "Stereotactic Radiosurgery",
        "Stereotactic Body Radiotherapy",
        "Artificial Intelligence Applications"
    ],
    "Quality Assurance": [
        "Equipment QA",
        "Treatment QA",
        "Process-Specific QA"
    ],
    "Radiation Safety": [
        "Radiation Protection Principles",
        "Radiation Monitoring",
        "Safety Regulations",
        "Shielding Design",
        "Incident Management"
    ]
}

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
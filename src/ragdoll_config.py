# src/ragdoll_config.py
# This module defines default configurations, constants, and model parameters
# for the RAGdoll project. It helps centralize settings for easier management.

import os # For os.path, os.cpu_count, and environment variable access

# --- Shared File Path Constants for Data Storage ---
DEFAULT_SOURCE_DOCS_DIR = "data"

# These define filenames used when saving processed data.
VECTOR_STORE_SUBDIR_NAME = "vicinity_store" # Subdirectory within the main data_dir for Vicinity store
TEXT_CHUNKS_FILENAME = "text_chunks.json"   # File storing all extracted text chunks
CHUNK_IDS_FILENAME = "chunk_ids.json"       # File storing IDs for each chunk, corresponding to text_chunks
DETAILED_CHUNK_METADATA_FILENAME = "detailed_chunk_metadata.json" # File for rich metadata of each chunk
CHUNK_VECTORS_FILENAME = "chunk_vectors.npy" # Optional raw dump of chunk embeddings
VISUALIZATION_DATA_FILENAME = "visualization_plot_data.json" # Data for UMAP visualization

# --- Default Candidate Labels for Zero-Shot Classification ---
# A comprehensive list, can be overridden.
_DEFAULT_CLASSIFICATION_LABELS_FULL = [ 
    "Accounting", "Activism", "Advertising", "Aerospace Engineering", "Agriculture", "AI & Robotics",
    "Alternative Medicine", "Anatomy & Physiology", "Ancient Civilizations", "Animal Welfare", "Anthropology", "Archaeology",
    "Architecture", "Art History", "Artificial Intelligence", "Arts & Crafts", "Astronomy & Space Exploration", "Astrophysics",
    "Aviation", "Banking & Finance", "Behavioral Science", "Biochemistry", "Biography & Memoir", "Biology",
    "Biotechnology", "Botany", "Broadcasting", "Buddhism", "Business", "Calculus & Analysis", "Cartography",
    "Chemistry", "Child Development", "Chinese History", "Christianity", "Cinema & Film Studies", "Civil Engineering",
    "Civil Liberties & Rights", "Classical Studies", "Climate Change", "Climatology", "Cognitive Science", "Comedy",
    "Comics & Graphic Novels", "Communication Studies", "Comparative Literature", "Computer Hardware", "Computer Networks",
    "Computer Science", "Conflict Resolution", "Conservation", "Constitutional Law", "Construction", "Consumer Affairs",
    "Contemporary History", "Corporate Governance", "Cosmology", "Crafts", "Creative Writing", "Crime & Criminology",
    "Criminal Justice", "Critical Theory", "Cryptography", "Cuisine & Gastronomy", "Cultural Anthropology",
    "Cultural Heritage", "Cultural Studies", "Cybersecurity", "Dance", "Data Science", "Databases", "Democracy", "Demography",
    "Dentistry", "Design (General)", "Development Economics", "Digital Arts", "Digital Media", "Diplomacy",
    "Disability Studies", "Disaster Management", "Drama & Theater", "Drawing & Illustration", "Earth Sciences", "E-commerce",
    "Ecology", "Econometrics", "Economic History", "Economic Policy", "Economics", "Education Policy",
    "Educational Psychology", "Egyptology", "Elections & Voting", "Electrical Engineering", "Electronic Music", "Electronics",
    "Emergency Services", "Energy Policy", "Engineering (General)", "Entrepreneurship", "Environmental Ethics",
    "Environmental Health", "Environmental Law", "Environmental Science", "Environmentalism", "Epidemiology", "Espionage",
    "Ethics (General)", "Ethnic & Racial Studies", "Ethnomusicology", "European History", "Evolutionary Biology", "Exercise Science",
    "Family Studies", "Fantasy (Genre)", "Fashion", "Feminism", "Festivals & Celebrations", "Fiction (General)",
    "Film Production", "Financial Markets", "Fine Arts", "Fitness & Wellness", "Folk Music", "Folklore & Mythology",
    "Food Science & Industry", "Foreign Policy", "Forensic Science", "Forestry", "Fossil Fuels", "Game Design & Theory",
    "Gaming (Video & Tabletop)", "Gardening & Horticulture", "Genealogy", "Genetics & Genomics", "Geography", "Geology",
    "Geometry", "Geopolitics", "Gerontology", "Global Health", "Globalization", "Governance", "Government",
    "Graphic Design", "Green Technology", "Health Policy", "Healthcare Administration", "Higher Education", "Hinduism",
    "Historiography", "Historical Fiction", "History (General)", "History of Art", "History of Mathematics", "History of Medicine",
    "History of Philosophy", "History of Religion", "History of Science", "History of Technology", "Home & Garden",
    "Homeland Security", "Hospitality & Tourism", "Housing & Urban Development", "Human Anatomy", "Human Evolution",
    "Human Geography", "Human Resource Management", "Human Rights", "Humanitarian Aid", "Humor", "Hydrology",
    "Immigration", "Immunology", "Indigenous Cultures", "Industrial Design", "Industrial Engineering",
    "Inequality & Social Stratification", "Infectious Diseases", "Information Science", "Information Systems",
    "Information Technology (IT)", "Infrastructure", "Insurance", "Intellectual Property", "Intelligence Community",
    "Interior Design", "International Law", "International Relations", "International Trade", "Internet & Society",
    "Investment", "Islam", "Jazz Music", "Journalism", "Judaism", "Judicial System", "Labor & Employment",
    "Labor History", "Land Management & Public Lands", "Landscape Architecture", "Language & Linguistics", "Latin American History",
    "Law (General)", "Law Enforcement", "Leadership Studies", "Legal History", "Legislation", "Liberalism (Political)",
    "Library Science", "Literary Criticism", "Literature (General)", "Logistics & Supply Chain", "Machine Learning",
    "Macroeconomics", "Manufacturing", "Marine Biology", "Maritime History & Transport", "Market Research", "Marketing",
    "Martial Arts", "Mass Media", "Materials Science", "Mathematical Logic", "Mathematical Physics", "Mathematics (General)",
    "Mechanical Engineering", "Media Ethics", "Media Production", "Media Studies", "Medical Research", "Medical Specialties",
    "Medicine (General)", "Medieval History", "Mental Health", "Metaphysics", "Meteorology", "Microbiology", "Microeconomics",
    "Middle Eastern History", "Military History", "Military Science & Defense", "Mining & Metallurgy", "Modern Art",
    "Modern History", "Molecular Biology", "Monetary Policy", "Moral Philosophy", "Motorsports", "Multimedia",
    "Museum Studies", "Music (General)", "Music History", "Music Production", "Music Theory", "Musical Instruments",
    "Mythology", "Nanotechnology", "National Security", "Nationalism", "Native American Studies", "Natural Disasters",
    "Natural Language Processing", "Naval Warfare", "Negotiation", "Network Science", "Neuroscience", "New Media",
    "News & Current Events", "Non-Fiction (General)", "Nonprofit Sector", "Nuclear Energy & Technology", "Nuclear Physics",
    "Numerical Analysis", "Nursing", "Nutrition & Dietetics", "Oceanography", "Oncology", "Opera", "Operations Management",
    "Optical Physics", "Oral History", "Organic Chemistry", "Organizational Behavior", "Painting", "Paleontology",
    "Parenting & Family", "Particle Physics", "Patent Law", "Pathology", "Peace & Conflict Studies", "Pedagogy",
    "Pediatrics", "Performing Arts", "Personal Finance", "Personality Psychology", "Pharmaceuticals", "Philanthropy",
    "Philosophy (General)", "Philosophy of Language", "Philosophy of Mind", "Philosophy of Religion", "Philosophy of Science",
    "Phonetics & Phonology", "Photojournalism", "Photography", "Physical Chemistry", "Physical Education",
    "Physical Geography", "Physics (General)", "Planetary Science", "Plant Science", "Poetry", "Political Activism",
    "Political Economy", "Political History", "Political Ideologies", "Political Philosophy", "Political Science (General)",
    "Political Theory", "Politics (General)", "Pollution Control", "Popular Culture", "Population Studies", "Post-Colonial Studies",
    "Poverty & Development", "Prehistory", "Print Media", "Printmaking", "Prisons & Corrections", "Privacy & Surveillance",
    "Probability & Statistics", "Product Design", "Professional Ethics", "Programming", "Project Management", "Propaganda",
    "Property Law", "Prose", "Psychiatry", "Psychoanalysis", "Psychology (General)", "Psychotherapy", "Public Administration",
    "Public Art", "Public Health", "Public Opinion", "Public Policy", "Public Relations", "Public Speaking", "Publishing",
    "Quantum Mechanics", "Race & Ethnicity", "Radio Broadcasting", "Radiology", "Rail Transport", "Real Estate",
    "Recreation & Leisure", "Recycling & Waste Management", "Refugee Studies", "Regional Studies", "Rehabilitation Medicine",
    "Religion (General)", "Religious History", "Religious Texts", "Remote Sensing", "Renewable Energy", "Research Methods",
    "Retail Industry", "Rhetoric", "Risk Management", "Robotics", "Rock Music", "Role-Playing Games", "Roman History",
    "Rural Studies", "Russian History", "Sales", "Satire", "Scams & Fraud", "Science Fiction (Genre)",
    "Science Communication", "Scientific Method", "Sculpture", "Search Engines", "Secondary Education", "Security Studies",
    "Seismology", "Semantics & Pragmatics", "Semiotics", "Sexual Health & Education", "Social Anthropology", "Social Change",
    "Social Class", "Social Ethics", "Social History", "Social Issues (General)", "Social Justice", "Social Media",
    "Social Movements", "Social Network Analysis", "Social Policy", "Social Psychology", "Social Sciences (General)",
    "Social Theory", "Social Welfare", "Social Work", "Society (General)", "Sociobiology", "Sociolinguistics",
    "Sociology (General)", "Software Engineering", "Soil Science", "Solar System", "Sound & Audio Engineering",
    "Space Policy", "Special Education", "Spectroscopy", "Speech & Public Speaking", "Spirituality", "Sports (General)",
    "Sports Analytics", "Sports Coaching", "Sports History", "Sports Industry", "Sports Medicine", "Sports Psychology",
    "Statistics", "Stem Cell Research", "Stock Market & Investing", "Storytelling", "Strategic Management", "Street Art",
    "Structural Engineering", "Student Life", "Subcultures", "Substance Abuse & Addiction", "Supply Chain Management",
    "Surgery", "Sustainability", "Sustainable Development", "Symbolism", "Syntax", "Systems Theory", "Taxation & Tax Law",
    "Teaching & Pedagogy", "Technical Communication", "Technological Innovation", "Technology (General)",
    "Technology Ethics", "Telecommunications", "Television Studies", "Terrorism & Counter-terrorism", "Textile Arts",
    "Theater & Performance Studies", "Theology", "Theoretical Physics", "Thermodynamics", "Think Tanks", "Thriller (Genre)",
    "Toxicology", "Trade & Commerce", "Traditional Medicine", "Traffic & Transport Management", "Translation Studies",
    "Transportation Engineering", "Transportation Policy", "Travel & Tourism", "Treaties & International Agreements",
    "Urban Design & Planning", "Urban Ecology", "Urban History", "Urban Studies", "User Experience (UX)", "Veterinary Medicine",
    "Video Games", "Virology", "Virtual Reality & Augmented Reality", "Visual Arts (General)", "Visual Culture",
    "Volcanology", "War & Military Operations", "Waste Management", "Water Resources Management", "Weather & Meteorology",
    "Web Design & Development", "Wellness & Lifestyle", "Wildlife Conservation", "Wind Energy", "Women's History",
    "Women's Studies & Gender", "World Economy", "World History", "World Literature", "World Music", "World Politics",
    "World Religions", "World War I", "World War II", "Writing & Composition", "Yoga & Meditation", "Youth Studies", "Zoology",
    "General Information", "Miscellaneous", "Other"
]
# A shorter, more general list of classification labels.
DEFAULT_CLASSIFICATION_LABELS_SHORT = [
    "Technology", "Business", "Science", "Health", "Finance", "Law", "Politics", 
    "Education", "Environment", "Culture", "Arts", "Sports","Lifestyle", 
    "History", "Travel", "General Information", "Miscellaneous", "Other"
]
# By default, the system uses the shorter list for classification tasks.
DEFAULT_CLASSIFICATION_LABELS = DEFAULT_CLASSIFICATION_LABELS_SHORT


# --- Default Model Names and Parameters ---
# These specify the default machine learning models used in various pipeline stages.
DEFAULT_PIPELINE_EMBEDDING_MODEL = "minishlab/potion-base-8M" # For generating embeddings of document chunks
DEFAULT_QUERY_EMBEDDING_MODEL = "minishlab/potion-base-8M"   # For embedding user queries (often same as pipeline)

# --- Processing Resource Defaults ---
_DEFAULT_GPU_DEVICE_ID_PROCESSING_STATIC = 0 # Base default before env var override
_DEFAULT_CHUNK_PROCESSING_WORKERS_STATIC = max(1, os.cpu_count() // 2 if os.cpu_count() else 1) # Use half of CPU cores if available

# --- CHUNKER CONFIGURATION DEFAULTS ---
# This dictionary stores default parameters for each supported Chonkie chunker type.
# Keys within each chunker's dict aim to match the parameter names expected by the Chonkie library's constructors.
# The pipeline_orchestrator maps CLI/API arguments to these internal names.
CHUNKER_DEFAULTS = {
    "chonkie_token": { 
        "tokenizer": "gpt2", 
        "chunk_size": 256,   
        "chunk_overlap": 0.0,
    },
    "chonkie_sentence": { 
        "tokenizer_or_token_counter": "ragdoll_utils.BGE_TOKENIZER_INSTANCE", # String placeholder
        "chunk_size": 256,                
        "chunk_overlap": 0.0,             
        "min_sentences_per_chunk": 1,     
        "min_characters_per_sentence": 12,
    },
    "chonkie_recursive": { 
        "tokenizer_or_token_counter": "ragdoll_utils.BGE_TOKENIZER_INSTANCE", # String placeholder
        "chunk_size": 256,                
        "min_characters_per_chunk": 24,   
        "rules": None, # Default to Chonkie's internal general separators. Can be "markdown" string or RecursiveRules object.
        # "lang": "en", # Removed as it's usually a parameter to rules.from_recipe, not RecursiveChunker.__init__
    },
    "chonkie_sdpm": { 
        "embedding_model": "minishlab/potion-base-8M", 
        "chunk_size": 256,                
        "threshold": "auto",              
        "similarity_window": 1,           
        "min_sentences": 2,               
        "skip_window": 5,                 
        "mode": "window",                 
        "min_chunk_size": 20,             
        "min_characters_per_sentence": 12,
        # "tokenizer_or_token_counter" removed, not directly used by SDPMChunker constructor
    },
    "chonkie_semantic": { 
        "embedding_model": "minishlab/potion-base-8M",
        "chunk_size": 256, 
        "threshold": 0.3, 
        "min_sentences": 2,
        "mode": "window",
        "similarity_window": 1,
        "min_chunk_size": 20,
        "min_characters_per_sentence": 12,
        # "tokenizer_or_token_counter" removed
    },
    "chonkie_neural": { 
        "model": "mirth/chonky_distilbert_base_uncased_1", 
        "tokenizer": "mirth/chonky_distilbert_base_uncased_1", 
        "stride": 128,                    
        "min_characters_per_chunk": 30,   
    }
}
# Default chunker type to be used if not specified by the user.
DEFAULT_CHUNKER_TYPE = "chonkie_sdpm" 

# --- Classification Defaults ---
DEFAULT_CLASSIFIER_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33" # Zero-shot model
DEFAULT_CLASSIFICATION_BATCH_SIZE = 16 # Batch size for classifying chunks

# --- Reranking and RAG Defaults ---
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Cross-encoder for reranking
DEFAULT_RERANKER_TOP_N = 7                # Number of chunks to return after reranking
DEFAULT_RERANKER_BATCH_SIZE = 32          # Batch size for reranker model
DEFAULT_RAG_INITIAL_K = 30                # Initial number of chunks to retrieve for RAG
DEFAULT_RAG_MAX_CONTEXT_TOKENS = 6800     # Max tokens for LLM context window (estimated)

# --- LLM Interaction Defaults ---
_DEFAULT_LLM_API_URL_STATIC = "http://localhost:8080/v1/chat/completions" # Base default before env var
DEFAULT_LLM_MAX_GENERATION_TOKENS = 1536  # Max tokens the LLM should generate
DEFAULT_LLM_TEMPERATURE = 0.5             # LLM temperature for generation
DEFAULT_SYSTEM_PROMPT_RAG = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
Your answer must be grounded in the information from the context.
If the context does not contain enough information to answer the question, state that and do not attempt to answer.
Cite relevant sources from the context by including their citation tag (e.g., [Source N]) at the end of sentences or paragraphs that use information from that source. Use multiple citations if a sentence uses information from multiple sources.

Context:
---
{context}
---
"""
DEFAULT_SYSTEM_PROMPT_EXPLORE = """Based on the following set of retrieved text chunks related to the user's query '{query}', please:
1. Identify 2-4 main themes or topics present across these chunks.
2. Provide a concise summary of the information related to these themes.
3. If you notice any interesting patterns, connections, surprising information, or potential contradictions, please highlight it.
Focus on synthesizing information from the provided context only.

Retrieved Context:
---
{context}
---

Exploration Insights:"""
DEFAULT_EXPLORE_MAX_CHUNKS_TO_SUMMARIZE = 7 # Max chunks to feed LLM for explore mode
DEFAULT_EXPLORE_LLM_SUMMARY_MAX_TOKENS = 768 # Max tokens for LLM's explore summary
DEFAULT_EXPLORE_LLM_SUMMARY_TEMPERATURE = 0.4 # Temperature for LLM explore summary

# --- UMAP & Service Defaults ---
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_METRIC = "cosine"
_DEFAULT_DATA_SERVICE_URL_STATIC = "http://localhost:8001" # Base default before env var

# --- Environment Variable Handling for Overriding Defaults ---
# This section allows configuration values to be overridden by environment variables.
ENV_VAR_PREFIX = "RAGDOLL_" # Prefix for all RAGdoll specific environment variables

ENV_GPU_DEVICE_ID = f"{ENV_VAR_PREFIX}GPU_DEVICE_ID"
ENV_LLM_API_URL = f"{ENV_VAR_PREFIX}LLM_API_URL"
ENV_DATA_SERVICE_URL = f"{ENV_VAR_PREFIX}DATA_SERVICE_URL"
ENV_HF_HUB_OFFLINE = f"{ENV_VAR_PREFIX}HF_HUB_OFFLINE" # For HuggingFace Hub offline mode
ENV_TRANSFORMERS_OFFLINE = "TRANSFORMERS_OFFLINE" # Standard HF var
ENV_LOG_LEVEL = f"{ENV_VAR_PREFIX}LOG_LEVEL" # For configuring application log level

def get_env_var_as_int(var_name: str, default: int) -> int:
    """Safely retrieves an environment variable and casts it to an integer."""
    try: 
        return int(os.environ.get(var_name, str(default)))
    except ValueError: 
        print(f"[Config Warning] Invalid integer value for env var {var_name}. Using default: {default}")
        return default

def get_env_var_as_bool(var_name: str, default: bool) -> bool:
    """Safely retrieves an environment variable and casts it to a boolean."""
    val = os.environ.get(var_name, str(default)).lower()
    return val in ['true', '1', 't', 'y', 'yes']

# Override defaults with environment variables if set
DEFAULT_GPU_DEVICE_ID_PROCESSING = get_env_var_as_int(ENV_GPU_DEVICE_ID, _DEFAULT_GPU_DEVICE_ID_PROCESSING_STATIC)
DEFAULT_CHUNK_PROCESSING_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 1) # This is usually system-dependent

DEFAULT_LLM_API_URL = os.environ.get(ENV_LLM_API_URL, _DEFAULT_LLM_API_URL_STATIC)
DEFAULT_DATA_SERVICE_URL = os.environ.get(ENV_DATA_SERVICE_URL, _DEFAULT_DATA_SERVICE_URL_STATIC)

HF_HUB_OFFLINE_RAGDOLL = get_env_var_as_bool(ENV_HF_HUB_OFFLINE, False) 
HF_HUB_OFFLINE_TRANSFORMERS = get_env_var_as_bool(ENV_TRANSFORMERS_OFFLINE, False)
# If either is set, enforce offline mode for Hugging Face libraries
ACTUAL_HF_OFFLINE_MODE = HF_HUB_OFFLINE_RAGDOLL or HF_HUB_OFFLINE_TRANSFORMERS

DEFAULT_LOG_LEVEL = os.environ.get(ENV_LOG_LEVEL, "INFO").upper()

if ACTUAL_HF_OFFLINE_MODE:
    os.environ["TRANSFORMERS_OFFLINE"] = "1" 
    os.environ["HF_HUB_OFFLINE"] = "1"       
    print("[Config] HuggingFace Hub and Transformers set to OFFLINE mode based on environment variable(s).")

print(f"[Config] Log level configured to: {DEFAULT_LOG_LEVEL}")
print(f"[Config] Default GPU Device ID for processing: {DEFAULT_GPU_DEVICE_ID_PROCESSING}")
print(f"[Config] Default LLM API URL: {DEFAULT_LLM_API_URL}")
print(f"[Config] Default Data Service URL: {DEFAULT_DATA_SERVICE_URL}")
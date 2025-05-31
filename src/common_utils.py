# common_utils.py
import os
import re
from transformers import AutoTokenizer as HFAutoTokenizer

# --- Shared File Path Constants ---
VECTOR_STORE_SUBDIR_NAME = "vicinity_store"
TEXT_CHUNKS_FILENAME = "text_chunks.json"
CHUNK_IDS_FILENAME = "chunk_ids.json"
DETAILED_CHUNK_METADATA_FILENAME = "detailed_chunk_metadata.json"
CHUNK_VECTORS_FILENAME = "chunk_vectors.npy"
VISUALIZATION_DATA_FILENAME = "visualization_plot_data.json"

# --- BGE Tokenizer (for client-side context counting & some chunkers if configured) ---
BGE_TOKENIZER_INSTANCE = None
TOKEN_COUNT_FALLBACK_ACTIVE = False
try:
    BGE_TOKENIZER_INSTANCE = HFAutoTokenizer.from_pretrained("baai/bge-base-en-v1.5")
    print("Common Utils: BGE Tokenizer (baai/bge-base-en-v1.5) loaded successfully.")
except Exception as e_bge_load:
    print(f"Common Utils: WARNING - Error loading BGE tokenizer: {e_bge_load}. Using basic split() for token counting.")
    BGE_TOKENIZER_INSTANCE = lambda text: len(str(text).split()) 
    TOKEN_COUNT_FALLBACK_ACTIVE = True

def count_tokens_robustly(text_to_count: str) -> int:
    text_str = str(text_to_count)
    if TOKEN_COUNT_FALLBACK_ACTIVE or not callable(getattr(BGE_TOKENIZER_INSTANCE, 'encode', None)):
        if callable(BGE_TOKENIZER_INSTANCE): return BGE_TOKENIZER_INSTANCE(text_str)
        return len(text_str.split())
    return len(BGE_TOKENIZER_INSTANCE.encode(text_str, add_special_tokens=False))

def clean_text(text: str) -> str:
    text_str = str(text) if text is not None else ""
    text_str = re.sub(r'\s+', ' ', text_str) # Consolidate multiple whitespace
    text_str = re.sub(r'\x00', '', text_str)  # Remove NULL bytes
    return text_str.strip()

def sanitize_filename_for_id(filename: str) -> str:
    s = str(filename).replace(os.path.sep, "_")
    s = re.sub(r'[^0-9a-zA-Z_.-]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_-.')
    return s if s else "unknown_file_component"

# Default candidate labels for classification (can be overridden)
DEFAULT_CLASSIFICATION_LABELS = [
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

    # Essential Fallbacks
    "General Information", "Miscellaneous", "Other"
]


# --- Default Model Names and Parameters ---
DEFAULT_PIPELINE_EMBEDDING_MODEL = "minishlab/potion-base-8M"
DEFAULT_QUERY_EMBEDDING_MODEL = "minishlab/potion-base-8M" 

DEFAULT_GPU_DEVICE_ID_PROCESSING = 0 
DEFAULT_CHUNK_PROCESSING_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)

# --- CHUNKER CONFIGURATION DEFAULTS ---
# CHUNKER_DEFAULTS['chunker_type_name']['param_name']
# Chonkie chunkers often use 'tokenizer_or_token_counter' or 'embedding_model'
# We'll use our robust BGE tokenizer for token counting where needed by Chonkie basic chunkers
# and specify embedding models for semantic/SDPM/Neural where Chonkie uses them internally.

CHUNKER_DEFAULTS = {
    "semchunk": { # Current basic/rule-based option
        "max_tokens_chunk": 256, "overlap_percent": 15,
    },
    "chonkie_token": { # Chonkie's basic token chunker
        "tokenizer": "gpt2", # Chonkie's TokenChunker takes a tokenizer string or instance
        "chunk_size": 256, "chunk_overlap": 0, # Can be int or float (percentage)
    },
    "chonkie_sentence": { # Chonkie's sentence chunker
        "tokenizer_or_token_counter": "gpt2", # Uses Chonkie's Tokenizer wrapper
        "chunk_size": 256, "chunk_overlap": 0, "min_sentences_per_chunk": 1,
        "min_characters_per_sentence": 12,
    },
    "chonkie_recursive": { # Chonkie's recursive chunker
        "tokenizer_or_token_counter": "gpt2", # Uses Chonkie's Tokenizer wrapper
        "chunk_size": 256, "min_characters_per_chunk": 24,
        # 'rules' typically uses Chonkie's RecursiveRules() default or from_recipe
    },
    "chonkie_sdpm": {
        "embedding_model": "minishlab/potion-base-8M", # For Chonkie's internal sentence similarity
        "chunk_size": 256,    # Target token count for final chunks (using its tokenizer)
        "threshold": "auto",  # Similarity threshold: float (0-1), int (1-100 percentile), or "auto"
        "similarity_window": 1, 
        "min_sentences": 2,
        "skip_window": 5,
        "mode": "window",     # "cumulative" or "window"
        "min_chunk_size": 20, # Min tokens per chunk
        "min_characters_per_sentence": 12,
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
    },
    "chonkie_neural": {
        "model": "mirth/chonky_distilbert_base_uncased_1", # Segmentation model
        "tokenizer": "mirth/chonky_distilbert_base_uncased_1", # Tokenizer for this specific model
        "stride": 128, 
        "min_characters_per_chunk": 30,
        # device_map for NeuralChunker handled by gpu_device_id in pipeline
    }
}
DEFAULT_CHUNKER_TYPE = "chonkie_sdpm"

# Classification
DEFAULT_CLASSIFIER_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
DEFAULT_CLASSIFICATION_LABELS = [
    "Technology", "Business", "Science", "Health", "Finance", "Law", "Politics", 
    "Education", "Environment", "Culture", "Arts", "Sports","Lifestyle", 
    "History", "Travel", "General Information", "Miscellaneous", "Other"
]
DEFAULT_CLASSIFICATION_BATCH_SIZE = 16

# Reranking & RAG
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_TOP_N = 7 
DEFAULT_RERANKER_BATCH_SIZE = 32 
DEFAULT_RAG_INITIAL_K = 30 # Increased for better context window usage
DEFAULT_RAG_MAX_CONTEXT_TOKENS = 6800 

# LLM
DEFAULT_LLM_API_URL = "http://localhost:8080/v1/chat/completions" 
DEFAULT_LLM_MAX_GENERATION_TOKENS = 1536 
DEFAULT_LLM_TEMPERATURE = 0.5 
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
DEFAULT_EXPLORE_MAX_CHUNKS_TO_SUMMARIZE = 7 
DEFAULT_EXPLORE_LLM_SUMMARY_MAX_TOKENS = 768 
DEFAULT_EXPLORE_LLM_SUMMARY_TEMPERATURE = 0.4 

# UMAP & Service
DEFAULT_UMAP_NEIGHBORS = 15; DEFAULT_UMAP_MIN_DIST = 0.1; DEFAULT_UMAP_METRIC = "cosine"
DEFAULT_DATA_SERVICE_URL = "http://localhost:8001"
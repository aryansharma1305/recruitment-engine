import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_path(name: str, default: str) -> str:
    return os.path.expanduser(os.getenv(name, default))


OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
ALLOW_HOSTED_LLM_EXPLANATIONS = os.getenv("ALLOW_HOSTED_LLM_EXPLANATIONS", "false").lower() == "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR = r"C:\Users\DELL\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CANDIDATES_FILE = os.path.join(DEFAULT_DATA_DIR, "candidates.jsonl")
SAMPLE_FILE = os.path.join(DEFAULT_DATA_DIR, "sample_candidates.json")
JD_FILE = os.path.join(DEFAULT_DATA_DIR, "job_description.docx")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "submission.csv")
TOP500_CSV = os.path.join(OUTPUT_DIR, "top500.csv")
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
EMBED_BATCH     = 128

TOP_N_PREFILTER = 5000
TOP_N_RANK      = 500
TOP_N_EXPLAIN   = 100
FINAL_N         = 100

FRAUD_CUTOFF    = 0.55

EXP_MIN     = 3.0
EXP_IDEAL   = 6.5
EXP_MAX     = 12.0

WEIGHTS = {
    "skill_match":       0.23,
    "semantic_sim":      0.14,
    "career_relevance":  0.16,
    "experience":        0.10,
    "behavioral":        0.09,
    "product_impact":    0.10,
    "shipped_system":    0.08,
    "applied_ml_years":  0.05,
    "availability":      0.03,
    "location_logistics": 0.02,
    "education":         0.01,
}

JD_SKILL_BUCKETS = {
    "retrieval_embeddings": {
        "weight": 0.20,
        "terms": {
            "embedding", "embeddings", "sentence transformers", "sentence-transformers",
            "bge", "e5", "semantic search", "dense retrieval", "retrieval",
        },
    },
    "vector_search": {
        "weight": 0.18,
        "terms": {
            "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
            "faiss", "opensearch", "elasticsearch", "hybrid search", "bm25",
        },
    },
    "ranking_eval": {
        "weight": 0.18,
        "terms": {
            "ranking", "reranking", "learning to rank", "learning-to-rank", "ltr",
            "recommendation system", "recommender", "search ranking", "ndcg", "mrr",
            "map", "a/b testing", "ab testing", "offline evaluation", "online evaluation",
        },
    },
    "python_ml": {
        "weight": 0.16,
        "terms": {
            "python", "pytorch", "tensorflow", "scikit-learn", "sklearn", "numpy",
            "pandas", "machine learning", "ml engineer",
        },
    },
    "llm_nlp": {
        "weight": 0.13,
        "terms": {
            "llm", "large language model", "rag", "retrieval augmented generation",
            "fine-tuning", "finetuning", "lora", "qlora", "peft", "transformers",
            "huggingface", "hugging face", "nlp", "natural language processing",
        },
    },
    "production_ml": {
        "weight": 0.10,
        "terms": {
            "mlops", "model deployment", "production ml", "inference", "docker",
            "kubernetes", "aws", "gcp", "azure", "feature store", "monitoring",
        },
    },
    "product_context": {
        "weight": 0.05,
        "terms": {
            "marketplace", "hr tech", "hr-tech", "recruiting", "talent", "product",
            "user engagement", "recruiter", "search product",
        },
    },
}

JD_CORE_SKILLS = [
    "python", "embeddings", "sentence transformers", "bge", "e5",
    "vector database", "hybrid search", "dense retrieval", "semantic search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch",
    "llm", "large language model", "fine-tuning", "lora", "qlora", "peft", "rlhf",
    "transformers", "huggingface", "rag", "retrieval augmented generation",
    "pytorch", "tensorflow", "nlp", "natural language processing", "deep learning",
    "ranking", "recommendation system", "learning to rank", "reranking", "bm25",
    "ndcg", "mrr", "map", "xgboost", "lightgbm",
    "mlops", "model deployment", "docker", "kubernetes", "a/b testing",
    "model evaluation", "feature engineering", "aws", "gcp", "azure",
]

SKILL_ALIASES = {
    "llms": "llm",
    "large language models": "llm",
    "vector db": "vector database",
    "vector databases": "vector database",
    "opensearch": "elasticsearch",
    "bm25": "information retrieval",
    "rag": "retrieval augmented generation",
    "lora": "fine-tuning",
    "qlora": "fine-tuning",
    "peft": "fine-tuning",
    "hugging face": "huggingface",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "k8s": "kubernetes",
    "wandb": "mlops",
    "mlflow": "mlops",
    "bentoml": "model deployment",
    "weights & biases": "mlops",
    "sentence-transformers": "sentence transformers",
    "ab testing": "a/b testing",
    "recsys": "recommendation system",
    "rec sys": "recommendation system",
    "information retrieval": "information retrieval",
}

TITLE_SCORES = {
    "ai engineer":                       100,
    "ml engineer":                       100,
    "machine learning engineer":         100,
    "senior machine learning engineer":  100,
    "nlp engineer":                       95,
    "deep learning engineer":             95,
    "applied scientist":                  90,
    "research engineer":                  90,
    "search engineer":                    90,
    "ranking engineer":                   90,
    "recommendation engineer":            90,
    "senior data scientist":              85,
    "data scientist":                     78,
    "data engineer":                      60,
    "backend engineer":                   52,
    "software engineer":                  48,
    "platform engineer":                  48,
    "junior ml engineer":                 42,
    "junior ai engineer":                 42,
    "full stack engineer":                35,
    "devops engineer":                    30,
    "business analyst":                   12,
    "project manager":                    12,
    "marketing manager":                   4,
    "hr manager":                          4,
    "accountant":                          4,
    "graphic designer":                    4,
    "content writer":                      4,
    "civil engineer":                      4,
    "mechanical engineer":                 4,
    "customer support":                    4,
    "sales executive":                     4,
    "operations manager":                  4,
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "ltimindtree",
}

RELEVANT_EDU_FIELDS = {
    "computer science", "artificial intelligence", "machine learning",
    "data science", "information technology", "statistics",
    "mathematics", "computational", "software engineering",
    "electronics", "electrical engineering",
}

BEHAVIORAL_W = {
    "open_to_work":          0.20,
    "github_activity":       0.18,
    "response_rate":         0.15,
    "interview_completion":  0.12,
    "profile_completeness":  0.12,
    "recency":               0.10,
    "verification":          0.08,
    "notice_period":         0.05,
}

JD_EMBED_TEXT = (
    "Senior AI Engineer. Production embeddings retrieval systems sentence-transformers BGE E5 OpenAI. "
    "Vector databases Pinecone Weaviate Qdrant Milvus FAISS OpenSearch Elasticsearch. "
    "Strong Python. Ranking retrieval matching systems. Hybrid search dense retrieval. "
    "LLM fine-tuning LoRA QLoRA PEFT. Learning-to-rank XGBoost neural ranking. "
    "NLP transformers HuggingFace. Evaluation NDCG MRR MAP A/B testing offline metrics. "
    "MLOps model deployment Docker Kubernetes. Product company not pure research. "
    "Shipped recommendation or ranking system to real users at scale. "
    "5-9 years experience applied ML AI roles."
)

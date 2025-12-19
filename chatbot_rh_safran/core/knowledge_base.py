import os
import logging
import re
import unicodedata
import json
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Pour l'analyse statique uniquement (évite les erreurs Pylance si package absent)
    from sentence_transformers import SentenceTransformer  # type: ignore

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as _e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ sentence-transformers import failed: {_e}")


def _normalize_text(s: str) -> str:
    """Lowercase, remove diacritics and unify whitespace for robust matching."""
    if s is None:
        return ""
    s = str(s)
    # fix common mojibake if any (non-destructive), then normalize
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class KnowledgeBase:
    """
    Classe responsable de charger la base de connaissances RH
    et de fournir des fonctions de recherche intelligente (RAG simple).
    """

    def __init__(self):
        self.df_rh = None
        self.embeddings = None
        self.model = None
        self.user_profiles = []
        self.load_data()
        self.load_simulated_users()

    def load_data(self):
        """Charge les données RH depuis le CSV en gérant encodages et formats variés."""
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "RH_infos.csv"))
        logger = logging.getLogger(__name__)

        # Essais pd.read_csv classiques (UTF-8 -> cp1252 -> latin-1)
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                self.df_rh = pd.read_csv(data_path, encoding=enc)
                break
            except Exception:
                self.df_rh = None
        if self.df_rh is None:
            raise RuntimeError(f"Impossible de lire {data_path} avec les encodages courants.")

        # Fallback : certaines CSV ont chaque ligne entre guillemets -> on parse manuellement en essayant plusieurs encodages
        if "question" not in (c.lower() for c in self.df_rh.columns):
            logger.warning("Colonnes 'question' absentes -> tentative de parsing manuel du CSV (fallback).")
            chosen_text = None
            chosen_enc = None
            raw = open(data_path, "rb").read()
            for enc in ("utf-8", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                except Exception:
                    continue
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                # enlever guillemets externes s'il y en a
                def strip_outer_quotes(s: str) -> str:
                    return s[1:-1] if s.startswith('"') and s.endswith('"') else s
                cleaned = [strip_outer_quotes(ln) for ln in lines]
                header = cleaned[0].split(",")
                norm_headers = [_normalize_text(h) for h in header]
                if "question" in norm_headers:
                    chosen_text = cleaned
                    chosen_enc = enc
                    break
            if chosen_text is None:
                # dernier recours : parse UTF-8 replacer
                text = raw.decode("utf-8", errors="replace")
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                cleaned = [strip_outer_quotes(ln) for ln in lines]
                chosen_text = cleaned
                chosen_enc = "utf-8 (replaced)"

            # reconstruire DataFrame
            split_lines = [ln.split(",") for ln in chosen_text]
            header = [h.strip().strip('"') for h in split_lines[0]]
            rows = [[v.strip().strip('"') for v in row] for row in split_lines[1:]]
            self.df_rh = pd.DataFrame(rows, columns=header)

        # Nettoyage simple des colonnes texte (espaces insécables, apostrophes typographiques, trim)
        self.df_rh.columns = [c.strip() for c in self.df_rh.columns]
        for col in ("question", "reponse", "profil", "domaine"):
            if col in self.df_rh.columns:
                self.df_rh[col] = (
                    self.df_rh[col]
                    .astype(str)
                    .str.replace("\xa0", " ")
                    .str.replace("\u2019", "'")
                    .str.strip()
                )

        # vérifier que les colonnes essentielles existent
        required = {"question", "reponse", "profil", "domaine"}
        missing = required - set(self.df_rh.columns)
        if missing:
            raise RuntimeError(f"Colonnes manquantes dans {data_path}: {missing}")

        # normaliser profil
        self.df_rh["profil"] = self.df_rh["profil"].fillna("").astype(str).str.strip()

        # ajouter colonne normalisée pour recherches rapides (évite recalculs répétitifs)
        self.df_rh["combined_norm"] = (self.df_rh["question"].fillna("") + " " + self.df_rh["reponse"].fillna("")).apply(_normalize_text)

        logger.info("✅ Base de connaissances RH chargée : %d questions (encoding guessed)", len(self.df_rh))

        # Création d'une colonne combinée pour une meilleure recherche
        self.df_rh["combined"] = self.df_rh["question"] + " " + self.df_rh["reponse"]

        # Initialisation du modèle d'embedding (léger et performant)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
                # Génération des embeddings pour chaque entrée
                texts = self.df_rh["combined"].tolist()
                self.embeddings = self.model.encode(texts, convert_to_tensor=True)
                print("✅ Modèle d'embedding chargé et embeddings générés")
            except Exception as e:
                print(f"⚠️  Impossible de charger le modèle d'embedding : {e}")
                print("⚠️  Fallback sur recherche textuelle classique")
                self.model = None
        else:
            print(
                "⚠️  sentence-transformers non disponible — fallback sur recherche textuelle classique"
            )
            self.model = None

    def load_simulated_users(self):
        """Charge les profils utilisateurs simulés."""
        users_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "simulated_users.json"
        )
        with open(users_path, "r", encoding="utf-8") as f:
            self.user_profiles = json.load(f)
        print(f"✅ {len(self.user_profiles)} profils utilisateurs chargés")

    def _compute_profile_score(self, profil_value: str, user_profil: str | None):
        """Score de priorité selon profil utilisateur (valeur plus élevée = plus prioritaire).
        Règles spéciales: pour CDI on privilégie les réponses génériques avant CDI-specific ;
        pour CDD on privilégie les réponses CDD avant génériques.
        """
        GENERIC_PROFILS = {
            "",
            None,
            "all",
            "tous",
            "toutes",
            "tous profils",
            "générique",
            "generique",
            "generic",
        }
        profil_lower = str(profil_value or "").strip().lower()
        if not user_profil:
            return 1 if profil_lower in GENERIC_PROFILS else 0

        user_lower = str(user_profil).lower()
        # CDI: prefer generic > CDI > others
        if user_lower == "cdi":
            if profil_lower in GENERIC_PROFILS:
                return 4
            if profil_lower == "cdi":
                return 3
            if profil_lower == "cdd":
                return 1
            return 0
        # CDD: prefer CDD > generic > others
        if user_lower == "cdd":
            if profil_lower == "cdd":
                return 4
            if profil_lower in GENERIC_PROFILS:
                return 3
            return 1
        # Default behavior: exact match > generic > other
        if profil_lower == user_lower:
            return 3
        if profil_lower in GENERIC_PROFILS:
            return 2
        return 1

    def exact_text_search(self, query, user_profil=None, domaine=None, top_k=3):
        """Recherche textuelle "exacte" (prioritaire avant recherche sémantique)."""
        q_norm = _normalize_text(query or "")
        tokens = [t for t in re.findall(r"\w+", q_norm) if len(t) > 2]
        ngrams = []
        for n in (3, 2):
            ngrams += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngrams += tokens

        candidates = []
        for idx, row in self.df_rh.iterrows():
            if domaine and domaine.lower() not in str(row["domaine"]).lower():
                continue
            combined_text = row.get("combined_norm", _normalize_text(row.get("question", "") + " " + row.get("reponse", "")))
            matched = False
            matched_length = 0
            in_question = False
            for ng in ngrams:
                if ng and ng in combined_text:
                    matched = True
                    matched_length = max(matched_length, len(ng))
                    if ng in _normalize_text(str(row.get("question", ""))):
                        in_question = True
            if not matched:
                continue

            profile_score = self._compute_profile_score(row.get("profil", ""), user_profil)

            candidates.append(
                {
                    "question": row["question"],
                    "reponse": row["reponse"],
                    "profil": row["profil"],
                    "domaine": row["domaine"],
                    "similarity": float(matched_length),
                    "profile_score": profile_score,
                    "in_question": in_question,
                    "source": "Base RH (Exact)",
                }
            )

        candidates.sort(key=lambda x: (x["profile_score"], x["in_question"], x["similarity"]), reverse=True)
        results = []
        for c in candidates[:top_k]:
            c_copy = c.copy()
            c_copy.pop("profile_score", None)
            c_copy.pop("in_question", None)
            results.append(c_copy)
        return results

    def semantic_search(self, query, user_profil=None, domaine=None, top_k=3):
        """
        Recherche sémantique dans la base de connaissances.
        Utilise l'embedding pour trouver les réponses les plus pertinentes.

        NOTE: On exécute d'abord une recherche textuelle exacte (pour capter
        cas comme 'congés payés'), puis la recherche sémantique si rien trouvé.
        """
        # First, attempt exact/text search — if found, return those results immediately.
        exact_results = self.exact_text_search(query, user_profil=user_profil, domaine=domaine, top_k=top_k)
        if exact_results:
            return exact_results

        if self.model is None or self.embeddings is None:
            # Fallback: recherche textuelle simple
            return self.keyword_search(query, user_profil, domaine, top_k)

        # Encode la requête
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calcul des similarités cosinus (simplifié)
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1), self.embeddings.cpu().numpy()
        )[0]

        # On sélectionne d'abord un plus grand ensemble par similarité pour limiter le travail
        candidate_count = min(len(similarities), max(top_k * 5, top_k * 2))
        top_indices = np.argsort(similarities)[::-1][:candidate_count]

        # Définir ce qui compte comme "réponse générique"
        GENERIC_PROFILS = {
            "",
            None,
            "all",
            "tous",
            "toutes",
            "tous profils",
            "générique",
            "generique",
            "generic",
        }

        # Construire candidats avec score de profil pour priorisation
        candidates = []
        for idx in top_indices:
            row = self.df_rh.iloc[idx]

            if domaine and domaine.lower() not in str(row["domaine"]).lower():
                continue

            profil_value = str(row.get("profil", "")).strip()
            profile_score = self._compute_profile_score(profil_value, user_profil)

            candidates.append(
                {
                    "question": row["question"],
                    "reponse": row["reponse"],
                    "profil": row["profil"],
                    "domaine": row["domaine"],
                    "similarity": float(similarities[idx]),
                    "profile_score": profile_score,
                    "source": "Base RH",
                }
            )

        # Tri final: priorité sur profile_score (desc), puis similarity (desc)
        candidates.sort(key=lambda x: (x["profile_score"], x["similarity"]), reverse=True)

        # Retourner seulement top_k résultats (enlevant le champ interne profile_score)
        results = []
        for c in candidates[:top_k]:
            c_copy = c.copy()
            c_copy.pop("profile_score", None)
            results.append(c_copy)

        return results

    def keyword_search(self, query, user_profil=None, domaine=None, top_k=3):
        """
        Recherche par mot-clé (fallback).
        """
        # Filtrage initial
        filtered_df = self.df_rh.copy()

        if user_profil:
            filtered_df = filtered_df[filtered_df["profil"] == user_profil]

        if domaine:
            filtered_df = filtered_df[filtered_df["domaine"] == domaine]

        # Recherche par mots-clés dans la question
        query_words = query.lower().split()
        scores = []

        for idx, row in filtered_df.iterrows():
            score = 0
            question_lower = row["question"].lower()
            for word in query_words:
                if len(word) > 3 and word in question_lower:
                    score += 1
            scores.append((idx, score))

        # Trie par score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Construction des résultats
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:  # Seulement si au moins un mot correspond
                row = filtered_df.iloc[idx]
                results.append(
                    {
                        "question": row["question"],
                        "reponse": row["reponse"],
                        "profil": row["profil"],
                        "domaine": row["domaine"],
                        "similarity": score / len(query_words),  # Score normalisé
                        "source": "Base RH (Keyword)",
                    }
                )

        return results

    def get_user_profile(self, user_id=None, auth_token=None):
        """
        Récupère le profil d'un utilisateur simulé.
        Pour le POC, on simule l'authentification.
        """
        if not user_id and not auth_token:
            return None

        for user in self.user_profiles:
            if user_id and user["user_id"] == user_id:
                return user
            if auth_token and user["simulated_auth_token"] == auth_token:
                return user

        return None

    def get_domains(self):
        """Retourne la liste des domaines RH disponibles."""
        return sorted(self.df_rh["domaine"].unique().tolist())

    def get_profiles(self):
        """Retourne la liste des profils disponibles."""
        return sorted(self.df_rh["profil"].unique().tolist())

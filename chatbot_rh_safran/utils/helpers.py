"""
Module d'utilitaires pour le Chatbot RH Safran POC.
Contient des fonctions de nettoyage, validation, formatage et logique métier.
"""

import re
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# ============================================================================
# FONCTIONS DE NETTOYAGE ET PRÉTRAITEMENT DE TEXTE
# ============================================================================


def normalize_text(text: str, lower: bool = True, remove_accents: bool = True) -> str:
    """
    Normalise un texte pour améliorer la recherche.

    Args:
        text: Texte à normaliser
        lower: Convertir en minuscules
        remove_accents: Supprimer les accents

    Returns:
        Texte normalisé
    """
    if not isinstance(text, str):
        return ""

    # Conversion en minuscules
    if lower:
        text = text.lower()

    # Suppression des accents
    if remove_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in text if not unicodedata.combining(c)])

    # Suppression des caractères spéciaux non désirés (garder lettres, chiffres, espaces, ponctuation basique)
    text = re.sub(r"[^\w\s\-.,!?;:]", " ", text)

    # Réduction des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(
    text: str, min_length: int = 3, stopwords: List[str] = None
) -> List[str]:
    """
    Extrait les mots-clés pertinents d'un texte.

    Args:
        text: Texte d'origine
        min_length: Longueur minimale des mots à considérer
        stopwords: Liste de mots à ignorer

    Returns:
        Liste de mots-clés
    """
    if stopwords is None:
        # Stopwords français basiques
        stopwords = [
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "de",
            "et",
            "ou",
            "mais",
            "où",
            "donc",
            "car",
            "ne",
            "ni",
            "que",
            "qui",
            "quoi",
            "pour",
            "par",
            "sur",
            "sous",
            "dans",
            "avec",
            "sans",
            "est",
            "son",
            "sa",
            "ses",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "notre",
            "votre",
            "leur",
            "leurs",
            "ce",
            "cet",
            "cette",
            "ces",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "au",
            "aux",
            "à",
            "a",
            "as",
            "avoir",
            "être",
            "été",
            "étais",
            "sommes",
            "êtes",
            "sont",
            "ai",
            "avais",
            "avait",
            "avons",
            "avez",
            "avaient",
            "serai",
            "seras",
            "sera",
            "serons",
            "serez",
            "seront",
        ]

    # Normalisation
    clean_text = normalize_text(text, lower=True, remove_accents=True)

    # Extraction des mots
    words = re.findall(r"\b\w+\b", clean_text)

    # Filtrage
    keywords = []
    for word in words:
        if len(word) >= min_length and word not in stopwords and not word.isnumeric():
            keywords.append(word)

    return keywords


def detect_language(text: str) -> str:
    """
    Détection simple de la langue (FR/EN).
    Basé sur la présence de mots caractéristiques.

    Args:
        text: Texte à analyser

    Returns:
        'fr', 'en', ou 'unknown'
    """
    text_lower = text.lower()

    # Mots caractéristiques français
    french_indicators = [
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "et",
        "est",
        "dans",
        "pour",
        "avec",
        "sur",
        "sous",
        "par",
        "mais",
        "ou",
        "où",
        "donc",
        "car",
        "que",
        "qui",
        "quoi",
    ]

    # Mots caractéristiques anglais
    english_indicators = [
        "the",
        "a",
        "an",
        "and",
        "is",
        "in",
        "for",
        "with",
        "on",
        "under",
        "by",
        "but",
        "or",
        "where",
        "so",
        "because",
        "that",
        "which",
        "what",
        "who",
    ]

    french_count = sum(
        1 for word in french_indicators if f" {word} " in f" {text_lower} "
    )
    english_count = sum(
        1 for word in english_indicators if f" {word} " in f" {text_lower} "
    )

    if french_count > english_count:
        return "fr"
    elif english_count > french_count:
        return "en"
    else:
        # Vérification des caractères accentués typiquement français
        if re.search(r"[éèêëàâäôöûüç]", text_lower):
            return "fr"
        return "unknown"


# ============================================================================
# FONCTIONS DE VALIDATION ET CONTRÔLE
# ============================================================================


def validate_rh_question(
    question: str, min_words: int = 2, max_words: int = 30
) -> Dict[str, Any]:
    """
    Valide une question RH selon des critères métier.

    Args:
        question: Question à valider
        min_words: Nombre minimum de mots
        max_words: Nombre maximum de mots

    Returns:
        Dictionnaire avec 'is_valid' et 'message'
    """
    result = {
        "is_valid": True,
        "message": "Question valide",
        "word_count": 0,
        "has_rh_keyword": False,
    }

    # Vérification de la longueur
    words = question.strip().split()
    result["word_count"] = len(words)

    if len(words) < min_words:
        result["is_valid"] = False
        result["message"] = f"Question trop courte (minimum {min_words} mots)"
        return result

    if len(words) > max_words:
        result["is_valid"] = False
        result["message"] = f"Question trop longue (maximum {max_words} mots)"
        return result

    # Vérification des mots-clés RH
    rh_keywords = [
        "congé",
        "salaire",
        "paie",
        "transport",
        "avantage",
        "travail",
        "horaire",
        "absence",
        "maladie",
        "retraite",
        "formation",
        "mutuelle",
        "cantine",
        "ticket",
        "restaurant",
        "pointage",
        "contrat",
        "cdi",
        "cdd",
        "stage",
        "apprenti",
        "intérim",
    ]

    question_lower = question.lower()
    for keyword in rh_keywords:
        if keyword in question_lower:
            result["has_rh_keyword"] = True
            break

    if not result["has_rh_keyword"]:
        result["message"] = "Avertissement : la question ne semble pas liée aux RH"
        # On ne bloque pas, mais on avertit

    return result


def sanitize_user_input(input_text: str, max_length: int = 500) -> str:
    """
    Nettoie et sécurise l'entrée utilisateur.

    Args:
        input_text: Texte d'entrée
        max_length: Longueur maximale autorisée

    Returns:
        Texte nettoyé
    """
    if not input_text:
        return ""

    # Troncature
    if len(input_text) > max_length:
        input_text = input_text[:max_length] + "..."

    # Échappement des caractères dangereux (basique)
    dangerous_patterns = [
        (r"<script.*?>.*?</script>", "[script removed]"),
        (r"javascript:", "[javascript removed]"),
        (r"on\w+\s*=", "[event handler removed]"),
        (r"<.*?>", ""),  # Suppression des balises HTML
    ]

    for pattern, replacement in dangerous_patterns:
        input_text = re.sub(pattern, replacement, input_text, flags=re.IGNORECASE)

    return input_text.strip()


# ============================================================================
# FONCTIONS DE FORMATAGE ET PRÉSENTATION
# ============================================================================


def format_rh_response(response: str, user_profile: Optional[Dict] = None) -> str:
    """
    Formate une réponse RH pour une meilleure présentation.

    Args:
        response: Réponse brute
        user_profile: Profil utilisateur pour personnalisation

    Returns:
        Réponse formatée
    """
    if not response:
        return "Je n'ai pas de réponse à fournir pour le moment."

    # Personnalisation de la salutation
    greeting = ""
    if user_profile:
        name = user_profile.get("name", "").split()[0]  # Prénom seulement
        if name:
            greeting = f"Bonjour {name},\n\n"
        elif user_profile.get("profil"):
            greeting = f"Bonjour collaborateur {user_profile.get('profil')},\n\n"

    # Structuration de la réponse
    formatted = greeting

    # Si la réponse est courte, on la met en évidence
    if len(response) < 150:
        formatted += f"**{response}**"
    else:
        formatted += response

    # Ajout de la signature standard
    formatted += "\n\n---\n"
    formatted += "*Réponse fournie par l'assistant RH virtuel Safran*\n"
    formatted += "*Pour information personnalisée, contactez le service RH au 1234*"

    return formatted


def create_kpi_card(
    title: str, value: Any, delta: Optional[str] = None, icon: str = "📊"
) -> Dict[str, Any]:
    """
    Crée un dictionnaire représentant une carte KPI pour Streamlit.

    Args:
        title: Titre du KPI
        value: Valeur principale
        delta: Variation (optionnel)
        icon: Icône (optionnel)

    Returns:
        Dictionnaire formaté pour affichage
    """
    return {
        "title": f"{icon} {title}",
        "value": value,
        "delta": delta,
        "help": f"KPI: {title}",
    }


def generate_conversation_summary(conversation_history: List[Dict]) -> Dict[str, Any]:
    """
    Génère un résumé statistique d'une conversation.

    Args:
        conversation_history: Historique des messages

    Returns:
        Statistiques de la conversation
    """
    if not conversation_history:
        return {"total_messages": 0, "user_messages": 0, "assistant_messages": 0}

    user_msgs = [msg for msg in conversation_history if msg.get("role") == "user"]
    assistant_msgs = [
        msg for msg in conversation_history if msg.get("role") == "assistant"
    ]

    # Extraction des mots-clés des questions utilisateur
    all_user_text = " ".join([msg.get("content", "") for msg in user_msgs])
    top_keywords = extract_keywords(all_user_text)[:5]

    return {
        "total_messages": len(conversation_history),
        "user_messages": len(user_msgs),
        "assistant_messages": len(assistant_msgs),
        "first_message_time": (
            conversation_history[0].get("timestamp", "N/A")
            if conversation_history
            else "N/A"
        ),
        "last_message_time": (
            conversation_history[-1].get("timestamp", "N/A")
            if conversation_history
            else "N/A"
        ),
        "top_keywords": top_keywords,
    }


# ============================================================================
# FONCTIONS MÉTIER SPÉCIFIQUES SAFRAN
# ============================================================================


def get_profile_specific_info(profile_type: str) -> Dict[str, Any]:
    """
    Retourne les informations spécifiques à un profil employé.

    Args:
        profile_type: Type de profil (CDI, Cadre, CDD, Stagiaire, Intérim)

    Returns:
        Informations du profil
    """
    profiles_info = {
        "CDI": {
            "description": "Contrat à Durée Indéterminée",
            "avantages": [
                "Congés payés",
                "Mutuelle",
                "Transport",
                "Cantine",
                "RTT (si cadre)",
            ],
            "contacts": ["Service RH: 1234", "Manager direct"],
            "notes": "Accès complet aux avantages sociaux",
        },
        "Cadre": {
            "description": "Employé cadre",
            "avantages": [
                "Congés payés",
                "Mutuelle",
                "Transport",
                "Cantine",
                "RTT",
                "Voiture de fonction (si éligible)",
            ],
            "contacts": ["Service RH: 1234", "Direction"],
            "notes": "Horaires flexibles possibles",
        },
        "CDD": {
            "description": "Contrat à Durée Déterminée",
            "avantages": ["Congés payés proportionnels", "Mutuelle", "Transport"],
            "contacts": ["Service RH: 1235"],
            "notes": "Avantages proportionnels à la durée du contrat",
        },
        "Stagiaire": {
            "description": "Stagiaire",
            "avantages": ["Gratification (si > 2 mois)", "Transport", "Cantine"],
            "contacts": ["Tuteur de stage", "Service RH: 1236"],
            "notes": "Contrat spécifique stage",
        },
        "Intérim": {
            "description": "Intérimaire",
            "avantages": ["Salaire horaire", "Transport (selon mission)"],
            "contacts": ["Agence d’intérim", "Service RH: 1237"],
            "notes": "Contrat via agence d’intérim",
        },
    }

    return profiles_info.get(
        profile_type,
        {
            "description": "Profil non spécifié",
            "avantages": [],
            "contacts": ["Service RH: 1234"],
            "notes": "Contactez le service RH pour plus d’informations",
        },
    )


def check_holiday_eligibility(
    profile_type: str, seniority_months: int = 12
) -> Dict[str, Any]:
    """
    Vérifie l'éligibilité aux congés selon le profil.

    Args:
        profile_type: Type de profil
        seniority_months: Ancienneté en mois

    Returns:
        Informations d'éligibilité
    """
    base_days = 25  # Jours de base pour un CDI

    eligibility = {
        "eligible": True,
        "base_days": base_days,
        "additional_days": 0,
        "notes": "",
    }

    if profile_type == "CDI":
        if seniority_months >= 12:
            eligibility["additional_days"] = min(
                (seniority_months - 12) // 12, 5
            )  # +1 jour par an, max 5
        eligibility["notes"] = f"Ancienneté: {seniority_months} mois"

    elif profile_type == "Cadre":
        eligibility["base_days"] = 30
        eligibility["notes"] = "Cadre: 30 jours de base + RTT"

    elif profile_type == "CDD":
        eligibility["base_days"] = max(2, int((seniority_months / 12) * base_days))
        eligibility["notes"] = (
            f'Congés proportionnels: {eligibility["base_days"]} jours'
        )

    elif profile_type == "Stagiaire":
        if seniority_months >= 2:
            eligibility["base_days"] = 2
            eligibility["notes"] = "2 jours de congés pour stage > 2 mois"
        else:
            eligibility["eligible"] = False
            eligibility["notes"] = "Pas de congés pour stage < 2 mois"

    elif profile_type == "Intérim":
        eligibility["eligible"] = False
        eligibility["notes"] = "Congés gérés par l’agence d’intérim"

    return eligibility


# ============================================================================
# FONCTIONS DE DÉBOGAGE ET LOGGING
# ============================================================================


def log_performance(
    start_time: datetime, operation: str, details: str = ""
) -> Dict[str, Any]:
    """
    Log les performances d'une opération.

    Args:
        start_time: Heure de début
        operation: Nom de l'opération
        details: Détails supplémentaires

    Returns:
        Informations de performance
    """
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log_entry = {
        "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "operation": operation,
        "duration_seconds": round(duration, 3),
        "details": details,
        "performance_level": (
            "OK" if duration < 1.0 else "WARNING" if duration < 3.0 else "SLOW"
        ),
    }

    # Affichage console pour le POC
    if log_entry["performance_level"] != "OK":
        print(
            f"⏱️ [PERF] {operation}: {duration:.3f}s - {log_entry['performance_level']}"
        )

    return log_entry


# ============================================================================
# POINT D'ENTRÉE POUR TESTS
# ============================================================================

if __name__ == "__main__":
    """Tests des fonctions utilitaires"""

    # Test de normalisation
    test_text = "Évaluation des congés PAYÉS et transports..."
    print(f"Test normalisation: {normalize_text(test_text)}")

    # Test d'extraction de mots-clés
    keywords = extract_keywords(test_text)
    print(f"Mots-clés extraits: {keywords}")

    # Test de détection de langue
    print(f"Langue détectée: {detect_language('Hello world, how are you?')}")
    print(f"Langue détectée: {detect_language('Bonjour le monde, comment ça va?')}")

    # Test de validation
    validation = validate_rh_question("Je veux des congés")
    print(f"Validation: {validation}")

    print("✅ Tests des helpers terminés")

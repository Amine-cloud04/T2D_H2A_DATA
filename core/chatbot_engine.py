import re
from datetime import datetime
from utils.helpers import (
    normalize_text,
    extract_keywords,
    validate_rh_question,
    format_rh_response,
    get_profile_specific_info,
    sanitize_user_input  # AJOUT IMPORTANT
)
import logging

logger = logging.getLogger(__name__)


class ChatbotEngine:
    """
    Moteur principal du chatbot RH.
    Combine recherche sémantique, règles métier et gestion de contexte.
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.conversation_history = []

        # Règles métier supplémentaires (au-delà de la base de connaissances)
        self.rules = {
            "greeting": {
                "patterns": ["bonjour", "salut", "hello", "coucou", "bonsoir"],
                "response": "Bonjour ! Je suis l'assistant RH virtuel de Safran. Comment puis-je vous aider aujourd'hui ?",
            },
            "thanks": {
                "patterns": ["merci", "parfait", "super", "génial"],
                "response": "Je vous en prie ! N'hésitez pas si vous avez d'autres questions.",
            },
            "contact_hr": {
                "patterns": [
                    "parler à un humain",
                    "contacter rh",
                    "agent humain",
                    "vrai conseiller",
                ],
                "response": "Je peux vous rediriger vers le service RH. Composez le 1234 (interne) ou envoyez un email à rh.support@safran-fictif.com.",
            },
            "fallback": {
                "patterns": [],
                "response": "Je n'ai pas pu trouver une réponse précise à votre question dans ma base. Pour une information personnalisée, veuillez contacter directement le service RH au 1234 ou via le portail RH.",
            },
        }

    def detect_intent(self, query):
        """Détection simple d'intention basée sur des règles."""
        query_lower = query.lower()

        for intent, data in self.rules.items():
            for pattern in data["patterns"]:
                if pattern in query_lower:
                    return intent

        return "query_rh"  # Par défaut, on considère que c'est une question RH

    def extract_domain(self, query):
        """Extrait le domaine RH de la question (simplifié)."""
        query_lower = query.lower()
        domain_keywords = {
            "congés": ["congé", "vacances", "repos", "rtt"],
            "paie": ["salaire", "paie", "bulletin", "versement", "soldes"],
            "transport": ["transport", "bus", "navette", "véhicule"],
            "avantages": ["cantine", "restaurant", "avantage", "ticket", "repas"],
            "temps de travail": [
                "heure",
                "horaire",
                "pointage",
                "présence",
                "travail",
                "temps",
            ],
        }

        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return domain

        return None

    def generate_response(self, query, user_profile=None, include_context=True):
        """
        Génère une réponse à la question de l'utilisateur.
        """
        # DEBUG - pour voir ce qui est reçu
        logger.debug("Query reçue: %s; User Profile: %s", query, user_profile)
        
        # Nettoyage de la requête
        clean_query = sanitize_user_input(query.strip())  # UTILISATION DU HELPER

        # Détection d'intention
        intent = self.detect_intent(clean_query)

        # Réponses aux intentions spéciales
        if intent != "query_rh":
            base_response = self.rules[intent]["response"]

            # Personnalisation si on a un profil
            if user_profile:
                salutation = f"Bonjour {user_profile.get('name', 'collaborateur')}, "
                return salutation + base_response

            return base_response

        # Extraction du domaine pour affiner la recherche
        domain = self.extract_domain(clean_query)

        # Détermination du profil pour le filtrage
        user_profil_type = None
        if user_profile:
            user_profil_type = user_profile.get("profil")
            logger.debug("Recherche avec profil: %s", user_profil_type)

        # Recherche dans la base de connaissances
        search_results = self.kb.semantic_search(
            query=clean_query, user_profil=user_profil_type, domaine=domain, top_k=2
        )
        
        logger.info("Nombre de résultats trouvés: %d", len(search_results))

        SIMILARITY_THRESHOLD = 0.35  # ajuster si besoin
        if search_results:
            best_match = search_results[0]
            if best_match.get("similarity", 1.0) < SIMILARITY_THRESHOLD:
                logger.info("Best match below threshold (%.3f) — using fallback", best_match.get("similarity"))
                search_results = []
        
        # Construction de la réponse
        if search_results:
            best_match = search_results[0]
            logger.info("Meilleure correspondance: %s", best_match['question'][:50])

            # Construction de la réponse contextuelle
            response_parts = []

            # 1. Personnalisation de la salutation
            if user_profile:
                name = user_profile.get('name', '').split()[0]
                if name:
                    response_parts.append(f"Bonjour {name},\n\n")
                elif user_profile.get('profil'):
                    response_parts.append(f"Bonjour collaborateur {user_profile.get('profil')},\n\n")

            # 2. La réponse principale
            response_parts.append(f"**{best_match['reponse']}**")

            # 3. Mention de la source/personnalisation
            if user_profil_type and user_profil_type == best_match["profil"]:
                response_parts.append(
                    f"\n\n*(Cette information est spécifique aux {user_profil_type}s)*"
                )
            elif best_match["profil"] != "CDI":
                response_parts.append(
                    f"\n\n*(Information adaptée pour les {best_match['profil']}s)*"
                )

            # 4. Suggestions de questions liées (si d'autres résultats)
            if len(search_results) > 1:
                response_parts.append("\n\n**Questions liées :**")
                for i, result in enumerate(search_results[1:3], 1):
                    response_parts.append(f"\n{i}. {result['question']}")

            # 5. Rappel de contact
            response_parts.append(
                "\n\n---\n*Pour une information plus personnalisée, contactez le service RH au 1234.*"
            )

            full_response = "".join(response_parts)

        else:
            # Fallback : réponse générique
            logger.warning("Aucun résultat trouvé, utilisation du fallback")
            full_response = self.rules["fallback"]["response"]

            # Ajout de suggestions basées sur le domaine détecté
            if domain:
                domain_questions = self.kb.df_rh[
                    self.kb.df_rh["domaine"].str.contains(domain, case=False)
                ]
                if not domain_questions.empty:
                    full_response += "\n\n**Questions fréquentes sur ce thème :**"
                    for i, (_, row) in enumerate(
                        domain_questions.head(3).iterrows(), 1
                    ):
                        full_response += f"\n{i}. {row['question']}"

        # Journalisation de l'interaction (pour démo sécurité)
        self.log_interaction(
            query=clean_query,
            response=(
                full_response[:100] + "..."
                if len(full_response) > 100
                else full_response
            ),
            user_profile=user_profile,
            domain=domain,
            has_answer=bool(search_results),
        )

        return full_response

    def log_interaction(self, query, response, user_profile, domain, has_answer):
        """Journalise l'interaction pour démontrer la traçabilité."""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": (
                user_profile.get("user_id", "anonymous")
                if user_profile
                else "anonymous"
            ),
            "query": query,
            "domain_detected": domain,
            "response_preview": response[:50],
            "has_answer": has_answer,
            "security_note": "POC - Données simulées",
        }

        self.conversation_history.append(log_entry)

        # Pour le POC, on affiche juste dans la console
        print(
            f"📝 [LOG] {log_entry['timestamp']} - User:{log_entry['user_id']} - Q:'{query}' - Answered:{has_answer}"
        )

    def get_conversation_history(self, user_id=None):
        """Retourne l'historique des conversations (filtré par utilisateur si spécifié)."""
        if user_id:
            return [
                entry
                for entry in self.conversation_history
                if entry["user_id"] == user_id
            ]
        return self.conversation_history
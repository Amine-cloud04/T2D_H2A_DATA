import hashlib
import json
from datetime import datetime, timedelta


class SecuritySimulator:
    """
    Simule des mécanismes de sécurité pour le POC.
    Montre que nous pensons à l'authentification, au contrôle d'accès, etc.
    """

    def __init__(self):
        self.active_sessions = {}
        self.access_log = []

        # Règles d'accès simulées (par profil)
        self.access_rules = {
            "CDI": ["congés", "paie", "transport", "avantages", "temps de travail"],
            "Cadre": [
                "congés",
                "paie",
                "transport",
                "avantages",
                "temps de travail",
                "gestion_équipe",
            ],
            "CDD": ["congés", "paie", "transport", "avantages"],
            "Stagiaire": ["congés", "transport", "avantages"],
            "Intérim": ["transport", "temps de travail"],
        }

    def simulate_login(self, user_id, password="safran2024"):
        """
        Simule une authentification.
        Dans la réalité, ce serait connecté à LDAP/SSO.
        """
        # Pour le POC, on accepte n'importe quel mot de passe
        session_token = hashlib.sha256(
            f"{user_id}{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        session_data = {
            "user_id": user_id,
            "token": session_token,
            "login_time": datetime.now(),
            "expiry_time": datetime.now() + timedelta(hours=8),
            "ip_address": "192.168.1.100",  # Simulé
        }

        self.active_sessions[session_token] = session_data

        # Journalisation
        self.log_access(
            user_id=user_id,
            action="LOGIN",
            resource="CHATBOT",
            status="SUCCESS",
            details=f"Session créée: {session_token}",
        )

        return session_token

    def validate_session(self, session_token):
        """Valide une session simulée."""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]

            # Vérification d'expiration
            if datetime.now() > session["expiry_time"]:
                self.log_access(
                    user_id=session["user_id"],
                    action="SESSION_CHECK",
                    resource="CHATBOT",
                    status="EXPIRED",
                    details="Session expirée",
                )
                del self.active_sessions[session_token]
                return None

            return session

        return None

    def check_access_right(self, user_profile, domain):
        """
        Vérifie si un utilisateur a le droit d'accéder à un domaine RH.
        """
        if not user_profile or "profil" not in user_profile:
            return False

        profil = user_profile["profil"]
        allowed_domains = self.access_rules.get(profil, [])

        # Normalisation du domaine
        domain_lower = domain.lower() if domain else ""

        # Vérification
        for allowed in allowed_domains:
            if allowed in domain_lower or domain_lower in allowed:
                return True

        # Journalisation d'un accès refusé
        if domain:
            self.log_access(
                user_id=user_profile.get("user_id", "unknown"),
                action="ACCESS_CHECK",
                resource=f"DOMAIN:{domain}",
                status="DENIED",
                details=f"Profil {profil} non autorisé pour {domain}",
            )

        return False

    def log_access(self, user_id, action, resource, status, details=""):
        """Journalise les accès pour démontrer la traçabilité."""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "status": status,
            "details": details,
            "security_level": "POC_SIMULATION",
        }

        self.access_log.append(log_entry)

        # Affichage console pour le POC
        print(f"🔐 [SECURITY] {status} - {user_id} - {action} - {resource}")

    def get_security_report(self):
        """Génère un mini-rapport de sécurité pour la démo."""
        total_logins = len([log for log in self.access_log if log["action"] == "LOGIN"])
        denied_access = len(
            [log for log in self.access_log if log["status"] == "DENIED"]
        )
        active_sessions = len(self.active_sessions)

        return {
            "total_logins": total_logins,
            "denied_access_attempts": denied_access,
            "active_sessions": active_sessions,
            "last_activity": (
                self.access_log[-1]["timestamp"] if self.access_log else "None"
            ),
        }

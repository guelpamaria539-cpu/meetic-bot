#!/usr/bin/env python3
"""
Serveur API pour l'app mobile du bot Meetic
Lance sur ton Mac → accède depuis ton téléphone via http://[IP_MAC]:5002
"""

import json
import os
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from anthropic import Anthropic

BASE_DIR = os.path.expanduser("~/whatsapp-mcp-old/whatsapp-mcp")
PERSONAS_DIR = os.path.join(BASE_DIR, "personas")
MEMORY_FILE = os.path.join(BASE_DIR, "meetic_memory.json")
ACTIVE_PERSONA_FILE = os.path.join(BASE_DIR, "active_persona.txt")
MAX_HISTORY = 6000          # Messages max sauvegardés sur disque
CONTEXT_MESSAGES = 30       # Messages récents envoyés à Claude
SUMMARY_EVERY_N = 20        # Générer un résumé tous les N échanges

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

# ─── JSONBIN CONFIG ──────────────────────────────────────────────────────────
JSONBIN_BIN_ID = os.environ.get("JSONBIN_BIN_ID", "6a2703dfda38895dfe9b5f7f")
GROQ_API_KEY_DEFAULT = ""
JSONBIN_API_KEY = os.environ.get("JSONBIN_API_KEY", "$2a$10$It5.OZ/3pjhYaK5cvtKOhuX6uXUkEohjK6sP4REJSTQJbiRAVNsjy")
JSONBIN_URL = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"

def load_memory():
    # Essaie d'abord JSONBin (cloud), sinon fichier local
    try:
        import requests as _req
        resp = _req.get(
            JSONBIN_URL + "/latest",
            headers={"X-Access-Key": JSONBIN_API_KEY},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json().get("record", {})
            if data:
                return data
    except Exception as e:
        print(f"[JSONBin] Erreur lecture: {e}")
    
    # Fallback fichier local
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"historique": {}, "contacts": {}, "aliases": {}}

def save_memory(memory):
    # Sauvegarde sur JSONBin (cloud)
    try:
        import requests as _req
        resp = _req.put(
            JSONBIN_URL,
            headers={
                "Content-Type": "application/json",
                "X-Access-Key": JSONBIN_API_KEY
            },
            json=memory,
            timeout=10
        )
        if resp.status_code == 200:
            print("[JSONBin] Mémoire sauvegardée ✅")
            return
        else:
            print(f"[JSONBin] Erreur sauvegarde: {resp.status_code}")
    except Exception as e:
        print(f"[JSONBin] Erreur: {e}")
    
    # Fallback fichier local
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Local] Erreur sauvegarde: {e}")

def list_personas():
    # D'abord depuis JSONBin
    try:
        memory = load_memory()
        personas = memory.get("personas", {})
        if personas:
            return list(personas.keys())
    except Exception:
        pass
    # Fallback local
    if not os.path.exists(PERSONAS_DIR):
        return []
    return [f.replace(".json", "") for f in os.listdir(PERSONAS_DIR) if f.endswith(".json")]

def load_persona(nom):
    # D'abord depuis JSONBin
    try:
        memory = load_memory()
        personas = memory.get("personas", {})
        if nom.lower() in personas:
            return personas[nom.lower()]
    except Exception:
        pass
    # Fallback local
    path = os.path.join(PERSONAS_DIR, f"{nom.lower()}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_persona(nom, persona_data):
    memory = load_memory()
    if "personas" not in memory:
        memory["personas"] = {}
    memory["personas"][nom.lower()] = persona_data
    save_memory(memory)

def get_active_persona_name():
    try:
        memory = load_memory()
        return memory.get("active_persona", "laurine")
    except Exception:
        pass
    if os.path.exists(ACTIVE_PERSONA_FILE):
        with open(ACTIVE_PERSONA_FILE, "r") as f:
            return f.read().strip().lower()
    return "laurine"

def set_active_persona_name(nom):
    memory = load_memory()
    memory["active_persona"] = nom.lower()
    save_memory(memory)

def get_active_persona():
    nom = get_active_persona_name()
    return load_persona(nom)

def get_all_contacts():
    memory = load_memory()
    persona = get_active_persona()
    all_contacts = {}
    # Depuis la persona
    if persona:
        for nom, data in persona.get("contacts", {}).items():
            all_contacts[nom] = data
    # Depuis la mémoire globale
    for nom, data in memory.get("contacts", {}).items():
        if nom not in all_contacts:
            all_contacts[nom] = data
        else:
            all_contacts[nom].update(data)
    return all_contacts

def get_contact_history(contact_name):
    memory = load_memory()
    return memory.get("historique", {}).get(contact_name, [])

def get_env_api_key():
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return os.getenv("ANTHROPIC_API_KEY", "")

def get_groq_api_key():
    # Variable Railway en priorité, sinon clé par défaut
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_file = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        key = line.split("=", 1)[1].strip()
    return key or GROQ_API_KEY_DEFAULT

def get_ai_provider():
    """Retourne 'groq' si clé Groq dispo, sinon 'anthropic'"""
    if get_groq_api_key():
        return "groq"
    return "anthropic"

def call_ai(system_prompt, user_message, max_tokens=500):
    """Appelle Groq ou Anthropic selon la config"""
    provider = get_ai_provider()
    
    if provider == "groq":
        import requests as _requests
        api_key = get_groq_api_key()
        resp = _requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.8
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    else:
        api_key = get_env_api_key()
        if not api_key:
            raise Exception("Aucune clé API disponible (Groq ou Anthropic)")
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text.strip()

def generate_summary(contact_name, persona, history, api_key):
    """
    Génère un résumé intelligent de la relation tous les SUMMARY_EVERY_N échanges.
    Sauvegardé dans meetic_memory.json sous contacts[nom][resume]
    """
    try:
        hist_text = "\n".join([
            f"{'LUI' if h['role']=='lui' else persona['nom'].upper()}: {h['content']}"
            for h in history[-60:]
        ])
        return call_ai(
            "Tu résumes une relation entre deux personnes sur un site de rencontre. Sois précis et factuel.",
            f"""Résume cette conversation entre {persona['nom']} et son interlocuteur.

{hist_text}

Retourne un résumé structuré avec :
- Ce qu'on sait de lui (vie, personnalité, situation)
- L'état de la relation (niveau de confiance, intérêt mutuel)
- Les moments clés échangés
- Points importants à retenir pour la suite

Max 200 mots. Sois factuel.""",
            max_tokens=600
        )
    except Exception as e:
        print(f"[Résumé] Erreur: {e}")
        return ""

def build_prompt(contact_name, persona, memory):
    contact = get_all_contacts().get(contact_name, {})
    now = datetime.datetime.now()
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    jour = jours[now.weekday()]
    if 6 <= now.hour < 11: moment = "matin"
    elif 11 <= now.hour < 14: moment = "midi"
    elif 14 <= now.hour < 18: moment = "après-midi"
    elif 18 <= now.hour < 22: moment = "soir"
    else: moment = "nuit"

    hist = memory.get("historique", {}).get(contact_name, [])
    
    # Résumé de la relation (généré automatiquement)
    resume = contact.get("resume", "")
    resume_section = ""
    if resume:
        resume_section = f"\nRÉSUMÉ DE LA RELATION :\n{resume}\n"

    # Seulement les CONTEXT_MESSAGES derniers messages envoyés à Claude
    hist_recent = hist[-CONTEXT_MESSAGES:]
    hist_text = ""
    if hist_recent:
        hist_text = f"\nDERNIERS ÉCHANGES ({len(hist_recent)} sur {len(hist)} au total) :\n"
        for h in hist_recent:
            role = "LUI" if h["role"] == "lui" else persona["nom"].upper()
            hist_text += f"[{role}] {h['content']}\n"

    return f"""Tu es {persona['nom']}. Profil :
{persona['profil']}

Date : {jour} {now.strftime('%d/%m/%Y')} à {now.strftime('%H:%M')} ({moment})

Contact : {contact_name}
Registre : {contact.get('registre', 'tutoiement')}
Stade : {contact.get('stade', 'debut')}
Infos accumulées : {contact.get('infos', 'rien encore')}
Points communs : {contact.get('points_communs', '')}
Sincérité observée : {contact.get('feeling', '')}
Niveau d'intérêt : {contact.get('interet', '')}
{resume_section}{hist_text}

RÈGLES : Tu es {persona['nom']}, jamais une IA. Style SMS naturel. Max 2-3 phrases. Pas de signature. Réponds uniquement avec le message."""

# ─── ROUTES API ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    persona = get_active_persona()
    return jsonify({
        "persona": persona["nom"] if persona else "aucune",
        "personas": list_personas(),
        "contacts_count": len(get_all_contacts())
    })

@app.route("/api/personas")
def get_personas():
    result = []
    for nom in list_personas():
        p = load_persona(nom)
        if p:
            result.append({"nom": p["nom"], "slug": nom})
    return jsonify(result)

@app.route("/api/persona/switch", methods=["POST"])
def switch_persona():
    data = request.json
    nom = data.get("nom", "").lower()
    if not load_persona(nom):
        return jsonify({"error": "Persona introuvable"}), 404
    set_active_persona_name(nom)
    # Aussi local si dispo
    try:
        with open(ACTIVE_PERSONA_FILE, "w") as f:
            f.write(nom)
    except Exception:
        pass
    return jsonify({"success": True, "persona": nom})

@app.route("/api/contacts")
def get_contacts():
    contacts = get_all_contacts()
    memory = load_memory()
    result = []
    for nom, data in contacts.items():
        hist = memory.get("historique", {}).get(nom, [])
        last_msg = ""
        last_date = ""
        if hist:
            last = hist[-1]
            last_msg = last.get("content", "")[:60] + "..." if len(last.get("content", "")) > 60 else last.get("content", "")
            last_date = last.get("date", "")
        result.append({
            "nom": nom,
            "stade": data.get("stade", "debut"),
            "plateforme": ", ".join(data.get("plateformes", ["meetic"])),
            "registre": data.get("registre", "tutoiement"),
            "infos": data.get("infos", ""),
            "points_communs": data.get("points_communs", ""),
            "friction": data.get("friction", ""),
            "messages_count": len(hist),
            "last_message": last_msg,
            "last_date": last_date
        })
    result.sort(key=lambda x: x["messages_count"], reverse=True)
    return jsonify(result)

@app.route("/api/contact/<nom>/history")
def get_history(nom):
    hist = get_contact_history(nom)
    persona = get_active_persona()
    persona_nom = persona["nom"] if persona else "Bot"
    result = []
    for h in hist[-50:]:
        result.append({
            "role": h["role"],
            "content": h["content"],
            "date": h.get("date", ""),
            "plateforme": h.get("plateforme", "meetic"),
            "is_bot": h["role"] != "lui"
        })
    return jsonify({"history": result, "persona": persona_nom})

def extract_and_update_profile(contact_name, message_recu, reponse_bot, api_key):
    """
    Après chaque échange, analyse le message reçu et extrait :
    - Nouvelles infos sur la vie de l'interlocuteur
    - Sa sincérité / son niveau d'intérêt
    - Des détails personnels utiles pour Laurine
    Met à jour la fiche contact automatiquement.
    """
    try:
        memory = load_memory()
        contact = get_all_contacts().get(contact_name, {})
        infos_actuelles = contact.get("infos", "")
        points_communs = contact.get("points_communs", "")
        historique = memory.get("historique", {}).get(contact_name, [])

        # Contexte des derniers échanges
        hist_context = "\n".join([
            f"{'LUI' if h['role']=='lui' else 'LAURINE'}: {h['content']}"
            for h in historique[-10:]
        ])

        prompt = f"""Tu analyses une conversation entre Laurine et {contact_name} sur un site de rencontre.

MESSAGE REÇU DE {contact_name.upper()} :
{message_recu}

RÉPONSE DE LAURINE :
{reponse_bot}

HISTORIQUE RÉCENT :
{hist_context}

INFOS DÉJÀ CONNUES SUR LUI :
{infos_actuelles or 'Aucune'}

POINTS COMMUNS DÉJÀ NOTÉS :
{points_communs or 'Aucun'}

Extrais UNIQUEMENT les nouvelles informations utiles révélées dans ce message.
Retourne un JSON avec ces champs (laisse vide si rien de nouveau) :
{{
  "nouvelles_infos": "nouvelles infos sur sa vie (profession, ville, famille, hobbies, expériences...)",
  "points_communs": "nouveaux points communs avec Laurine découverts",
  "sincerite": "bon" | "neutre" | "suspect",
  "niveau_interet": "fort" | "moyen" | "faible",
  "details_cles": "détails importants à retenir (a donné son numéro, a proposé une rencontre, a partagé quelque chose d'intime...)",
  "a_mis_a_jour": true | false
}}

Retourne UNIQUEMENT le JSON, sans texte autour."""

        text = call_ai(
            "Tu es un extracteur d'informations. Retourne uniquement du JSON valide.",
            prompt,
            max_tokens=400
        ).replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        if not data.get("a_mis_a_jour", False):
            return

        # Construire les nouvelles infos en fusionnant avec les anciennes
        nouvelles = data.get("nouvelles_infos", "").strip()
        nouveaux_points = data.get("points_communs", "").strip()
        details = data.get("details_cles", "").strip()

        if nouvelles or details:
            parties = []
            if infos_actuelles:
                parties.append(infos_actuelles)
            if nouvelles:
                parties.append(nouvelles)
            if details:
                parties.append(details)
            contact["infos"] = ". ".join(p.rstrip(".") for p in parties if p)

        if nouveaux_points:
            if points_communs:
                contact["points_communs"] = f"{points_communs}, {nouveaux_points}"
            else:
                contact["points_communs"] = nouveaux_points

        if data.get("sincerite"):
            contact["feeling"] = data["sincerite"]
        if data.get("niveau_interet"):
            contact["interet"] = data["niveau_interet"]

        # Sauvegarder
        if "contacts" not in memory:
            memory["contacts"] = {}
        memory["contacts"][contact_name] = contact
        save_memory(memory)
        print(f"[Profil] {contact_name} mis à jour — sincérité: {data.get('sincerite')} | intérêt: {data.get('niveau_interet')}")

    except Exception as e:
        print(f"[Profil] Erreur extraction {contact_name}: {e}")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    contact_name = data.get("contact")
    message = data.get("message", "").strip()
    context = data.get("context", "").strip()
    if not contact_name or not message:
        return jsonify({"error": "Contact et message requis"}), 400

    persona = get_active_persona()
    if not persona:
        return jsonify({"error": "Aucune persona active"}), 500

    memory = load_memory()
    system_prompt = build_prompt(contact_name, persona, memory)

    # Ajouter le contexte au message si fourni
    message_with_context = message
    if context:
        message_with_context = f"{message}\n\n[CONTEXTE IMPORTANT: {context}]"

    try:
        reply = call_ai(system_prompt, message_with_context, max_tokens=500)
        provider = get_ai_provider()
        return jsonify({"response": reply, "persona": persona["nom"], "provider": provider})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/save", methods=["POST"])
def save_exchange():
    data = request.json
    contact_name = data.get("contact")
    message_recu = data.get("message_recu", "")
    reponse_envoyee = data.get("reponse_envoyee", "")
    if not contact_name:
        return jsonify({"error": "Contact requis"}), 400

    memory = load_memory()
    persona = get_active_persona()
    persona_nom = persona["nom"].lower() if persona else "bot"

    if "historique" not in memory:
        memory["historique"] = {}
    if contact_name not in memory["historique"]:
        memory["historique"][contact_name] = []

    now_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    if message_recu:
        memory["historique"][contact_name].append({
            "role": "lui",
            "content": message_recu,
            "date": now_str,
            "plateforme": "meetic"
        })
    if reponse_envoyee:
        memory["historique"][contact_name].append({
            "role": persona_nom,
            "content": reponse_envoyee,
            "date": now_str,
            "plateforme": "meetic"
        })

    # Limite 6000 messages sur disque
    memory["historique"][contact_name] = memory["historique"][contact_name][-MAX_HISTORY:]

    # Mise à jour stade
    hist_len = len(memory["historique"][contact_name])
    if "contacts" not in memory:
        memory["contacts"] = {}
    if contact_name not in memory["contacts"]:
        memory["contacts"][contact_name] = {}
    if hist_len >= 30:
        memory["contacts"][contact_name]["stade"] = "attachement"
    elif hist_len >= 10:
        memory["contacts"][contact_name]["stade"] = "connaissance"

    save_memory(memory)

    # Traitements en arrière-plan
    if message_recu and reponse_envoyee:
        api_key = get_env_api_key()
        if api_key:
            import threading
            persona = get_active_persona()

            # 1. Extraction profil à chaque échange
            t1 = threading.Thread(
                target=extract_and_update_profile,
                args=(contact_name, message_recu, reponse_envoyee, api_key),
                daemon=True
            )
            t1.start()

            # 2. Résumé tous les SUMMARY_EVERY_N échanges
            nb_echanges = hist_len // 2
            if nb_echanges > 0 and nb_echanges % SUMMARY_EVERY_N == 0 and persona:
                def update_summary():
                    mem = load_memory()
                    hist = mem.get("historique", {}).get(contact_name, [])
                    resume = generate_summary(contact_name, persona, hist, api_key)
                    if resume:
                        if "contacts" not in mem:
                            mem["contacts"] = {}
                        if contact_name not in mem["contacts"]:
                            mem["contacts"][contact_name] = {}
                        mem["contacts"][contact_name]["resume"] = resume
                        save_memory(mem)
                        print(f"[Résumé] {contact_name} — résumé mis à jour ({nb_echanges} échanges)")
                t2 = threading.Thread(target=update_summary, daemon=True)
                t2.start()

    return jsonify({"success": True})

@app.route("/api/persona/create", methods=["POST"])
def create_persona():
    data = request.json
    nom = data.get("nom", "").strip()
    profil = data.get("profil", "").strip()
    if not nom or not profil:
        return jsonify({"error": "Nom et profil requis"}), 400
    persona = {
        "nom": nom,
        "profil": profil,
        "contacts": {}
    }
    save_persona(nom, persona)
    return jsonify({"success": True, "persona": nom})

@app.route("/api/persona/delete", methods=["POST"])
def delete_persona():
    data = request.json
    nom = data.get("nom", "").lower()
    if not nom:
        return jsonify({"error": "Nom requis"}), 400
    if nom == "laurine":
        return jsonify({"error": "Impossible de supprimer Laurine"}), 400
    memory = load_memory()
    personas = memory.get("personas", {})
    if nom not in personas:
        return jsonify({"error": "Persona introuvable"}), 404
    del personas[nom]
    memory["personas"] = personas
    # Si c'était la persona active, repasser sur laurine
    if memory.get("active_persona", "") == nom:
        memory["active_persona"] = "laurine"
    save_memory(memory)
    return jsonify({"success": True})

@app.route("/api/contact/pause", methods=["POST"])
def pause_contact():
    data = request.json
    nom = data.get("nom", "").strip()
    paused = data.get("paused", True)
    if not nom:
        return jsonify({"error": "Nom requis"}), 400
    memory = load_memory()
    if "contacts" not in memory:
        memory["contacts"] = {}
    if nom not in memory["contacts"]:
        memory["contacts"][nom] = {}
    memory["contacts"][nom]["paused"] = paused
    save_memory(memory)
    status = "mis en pause" if paused else "réactivé"
    print(f"[Pause] {nom} {status}")
    return jsonify({"success": True, "paused": paused})

@app.route("/api/contact/delete", methods=["POST"])
def delete_contact():
    data = request.json
    nom = data.get("nom", "").strip()
    if not nom:
        return jsonify({"error": "Nom requis"}), 400
    memory = load_memory()
    # Supprimer du contact list
    if "contacts" in memory and nom in memory["contacts"]:
        del memory["contacts"][nom]
    # Supprimer l'historique
    if "historique" in memory and nom in memory["historique"]:
        del memory["historique"][nom]
    # Supprimer les aliases pointant vers ce contact
    if "aliases" in memory:
        to_delete = [k for k, v in memory["aliases"].items() if v == nom]
        for k in to_delete:
            del memory["aliases"][k]
    # Supprimer aussi de la persona active
    persona = get_active_persona()
    if persona and nom in persona.get("contacts", {}):
        del persona["contacts"][nom]
        save_persona(persona["nom"].lower(), persona)
    save_memory(memory)
    print(f"[Contact] {nom} supprimé")
    return jsonify({"success": True})

@app.route("/api/contact/link", methods=["POST"])
def link_contact_whatsapp():
    data = request.json
    phone = data.get("phone", "").strip()
    contact_name = data.get("contact", "").strip()
    if not phone or not contact_name:
        return jsonify({"error": "Numéro et contact requis"}), 400

    # Nettoyer le numéro (enlever +, espaces)
    phone_clean = phone.replace("+", "").replace(" ", "").replace("-", "")

    memory = load_memory()
    if "aliases" not in memory:
        memory["aliases"] = {}

    # Créer l'alias numéro → nom contact
    memory["aliases"][phone_clean] = contact_name
    memory["aliases"][phone] = contact_name

    # Ajouter whatsapp aux plateformes du contact
    if "contacts" not in memory:
        memory["contacts"] = {}
    if contact_name not in memory["contacts"]:
        memory["contacts"][contact_name] = {}
    plateformes = memory["contacts"][contact_name].get("plateformes", ["meetic"])
    if "whatsapp" not in plateformes:
        plateformes.append("whatsapp")
    memory["contacts"][contact_name]["plateformes"] = plateformes

    # Fusionner historique WhatsApp existant si présent
    if phone_clean in memory.get("historique", {}) and phone_clean != contact_name:
        existing_hist = memory["historique"].pop(phone_clean, [])
        current_hist = memory["historique"].get(contact_name, [])
        memory["historique"][contact_name] = existing_hist + current_hist
        print(f"[Link] Historique WhatsApp de {phone_clean} fusionné avec {contact_name}")

    save_memory(memory)
    print(f"[Link] {phone_clean} lié à {contact_name}")
    return jsonify({"success": True, "alias": phone_clean, "contact": contact_name})

@app.route("/api/contact/add", methods=["POST"])
def add_contact():
    data = request.json
    nom = data.get("nom", "").strip()
    if not nom:
        return jsonify({"error": "Nom requis"}), 400
    memory = load_memory()
    if "contacts" not in memory:
        memory["contacts"] = {}
    memory["contacts"][nom] = {
        "registre": data.get("registre", "tutoiement"),
        "stade": "debut",
        "plateformes": [data.get("plateforme", "meetic")],
        "infos": data.get("infos", ""),
        "points_communs": "",
        "friction": ""
    }
    if "historique" not in memory:
        memory["historique"] = {}
    memory["historique"][nom] = []
    save_memory(memory)
    return jsonify({"success": True})

@app.route("/api/contact/<nom>/update", methods=["POST"])
def update_contact(nom):
    data = request.json
    memory = load_memory()
    if "contacts" not in memory:
        memory["contacts"] = {}
    if nom not in memory["contacts"]:
        memory["contacts"][nom] = {}
    for key in ["infos", "points_communs", "friction", "registre", "stade"]:
        if key in data:
            memory["contacts"][nom][key] = data[key]
    save_memory(memory)
    return jsonify({"success": True})

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "127.0.0.1"
    print("\n" + "="*55)
    print("  APP MOBILE BOT MEETIC")
    print("="*55)
    print(f"\n✅ Serveur démarré !")
    print(f"\n📱 Ouvre cette URL sur ton téléphone (même WiFi) :")
    print(f"   http://{ip}:5002")
    print(f"\n💻 Sur ton Mac :")
    print(f"   http://localhost:5002")
    print(f"\n💡 Assure-toi que ton téléphone et ton Mac")
    print(f"   sont sur le même réseau WiFi.")
    print("\n" + "="*55 + "\n")
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5002)), debug=False)

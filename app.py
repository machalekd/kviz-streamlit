# -*- coding: utf-8 -*-
"""
Vědomostní kvíz – Streamlit (CZ)
---------------------------------
Funkce:
- Témata: Příroda, Historie, Zeměpis, Kultura, Sport, Všeobecné
- Obtížnost: Lehká, Střední, Těžká
- Počet otázek: 10–100
- Limit na otázku: 5–30 s (po vypršení se otázka vyhodnotí jako neodpovězená)
- Možnosti A–D (shuffle), jedna správná
- Skóre + přehled + vysvětlení + export do CSV
- Import vlastní JSON banky otázek
- Generování otázek: Lokální (bez AI) + volitelně AI (OpenAI / Ollama)

Pozn.: Na Streamlit Cloud doporučuji mít v repu soubor requirements.txt:
    streamlit>=1.33
    pandas>=2.0
"""
from __future__ import annotations
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# =============== Datové typy ===============
@dataclass
class QA:
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    category: str
    difficulty: str

# =============== Vestavěná miniaturní banka (ukázková) ===============
BUILTIN_BANK: List[QA] = [
    # Příroda
    QA("Který plyn tvoří největší část zemské atmosféry?", ["Kyslík", "Dusík", "Oxid uhličitý", "Argon"], 1, "Přibližně 78 % je dusík.", "Příroda", "Lehká"),
    QA("Která část rostliny zajišťuje fotosyntézu nejvíce?", ["Kořen", "Květ", "Stonek", "List"], 3, "Fotosyntéza probíhá hlavně v listech.", "Příroda", "Lehká"),
    QA("Jak se nazývá věda o třídění organismů?", ["Ekologie", "Genetika", "Taxonomie", "Etologie"], 2, "Taxonomie se zabývá klasifikací.", "Příroda", "Střední"),
    QA("Který prvek má chemickou značku Fe?", ["Měď", "Železo", "Fluor", "Stříbro"], 1, "Fe je ferrum = železo.", "Příroda", "Lehká"),
    # Historie
    QA("V kterém roce došlo na našem území k sametové revoluci?", ["1968", "1989", "1993", "2004"], 1, "Sametová revoluce proběhla v roce 1989.", "Historie", "Lehká"),
    QA("Kdo byl prvním prezidentem Československa?", ["Tomáš G. Masaryk", "Edvard Beneš", "Klement Gottwald", "Ludvík Svoboda"], 0, "Prvním prezidentem byl T. G. Masaryk.", "Historie", "Lehká"),
    QA("Jak se jmenovala obchodní cesta spojující Čínu a Evropu?", ["Jantarová stezka", "Hedvábná stezka", "Královská cesta", "Cesta koření"], 1, "Hedvábná stezka.", "Historie", "Střední"),
    # Zeměpis
    QA("Která řeka je nejdelší v Evropě?", ["Dunaj", "Volha", "Rýn", "Dněpr"], 1, "Volha je nejdelší evropská řeka.", "Zeměpis", "Střední"),
    QA("Jaké je hlavní město Polska?", ["Varšava", "Krakov", "Gdaňsk", "Poznaň"], 0, "Hlavním městem Polska je Varšava.", "Zeměpis", "Lehká"),
    QA("Nejvyšší hora ČR je…", ["Sněžník", "Praděd", "Sněžka", "Lysá hora"], 2, "Sněžka (1603 m).", "Zeměpis", "Lehká"),
    # Kultura
    QA("Kdo složil symfonii ‚Novosvětská‘?", ["Ludwig van Beethoven", "Antonín Dvořák", "Bedřich Smetana", "Franz Schubert"], 1, "Dvořákova 9. symfonie.", "Kultura", "Střední"),
    QA("Jak se nazývá obrazová technika s malbou do vlhké omítky?", ["Freska", "Akryl", "Tempera", "Pastel"], 0, "Freska se maluje do čerstvé omítky.", "Kultura", "Střední"),
    # Sport
    QA("Kolik hráčů má tým na hřišti v ledním hokeji (včetně brankáře)?", ["5", "6", "7", "8"], 1, "Obvykle 6 hráčů: 5+brankář.", "Sport", "Lehká"),
    QA("Který turnaj je součástí tenisového Grand Slamu?", ["Indian Wells", "Cincinnati", "Wimbledon", "Basilej"], 2, "Wimbledon je jeden ze čtyř Grand Slamů.", "Sport", "Lehká"),
    # Všeobecné
    QA("Kolik minut má jedna hodina?", ["50", "60", "90", "100"], 1, "1 hodina = 60 minut.", "Všeobecné", "Lehká"),
    QA("Jaká veličina se v elektrotechnice značí U?", ["Proud", "Odpor", "Napětí", "Výkon"], 2, "U je napětí (V).", "Všeobecné", "Střední"),
]

TOPICS = ["Příroda", "Historie", "Zeměpis", "Kultura", "Sport", "Všeobecné"]
DIFFS = ["Lehká", "Střední", "Těžká"]

# =============== Pomocné funkce ===============
def shuffle_question(qa: QA) -> QA:
    idxs = list(range(4))
    random.shuffle(idxs)
    options = [qa.options[i] for i in idxs]
    new_correct = idxs.index(qa.correct_index)
    return QA(qa.question, options, new_correct, qa.explanation, qa.category, qa.difficulty)


def synthesize_questions_local(topic: str, difficulty: str, n: int) -> List[QA]:
    """Jednoduché lokální sestavení otázek bez AI, aby to fungovalo hned."""
    pool = [q for q in BUILTIN_BANK if q.category == topic and q.difficulty == difficulty]
    if len(pool) < n:
        pool += [q for q in BUILTIN_BANK if q.category == topic and q.difficulty != difficulty]
    while len(pool) < n:
        base_q = QA(
            question=f"[{difficulty}] ({topic}) Doplňující otázka: Která z možností je správná?",
            options=["Správná", "Možnost 2", "Možnost 3", "Možnost 4"],
            correct_index=0,
            explanation="Ukázková otázka vytvořená generátorem šablon.",
            category=topic,
            difficulty=difficulty,
        )
        pool.append(base_q)
    random.shuffle(pool)
    return [shuffle_question(q) for q in pool[:n]]


def synthesize_questions_ai(topic: str, difficulty: str, n: int, model_pref: str = "openai:gpt-4o-mini") -> List[QA]:
    """Volitelně použije OpenAI (nebo Ollama), pokud je k dispozici. Při chybě spadne zpět na lokální režim."""
    out: List[QA] = []
    try:
        if model_pref.startswith("openai:"):
            import openai  # type: ignore
            client = openai.OpenAI()
            sys_prompt = (
                "Jsi generátor kvízových otázek v češtině. Vrať POUZE validní JSON pole objektů: "
                "{question, options[4], correct_index(0-3), explanation, category, difficulty}. "
                "Dodrž český jazyk a požadovanou obtížnost."
            )
            user_prompt = (
                f"téma: {topic}
obtížnost: {difficulty}
počet: {n}
"
                "Požadavky: žádné opakující se otázky, žádné odpovědi typu 'všechny výše uvedené'."
            )
            resp = client.chat.completions.create(
                model=model_pref.split(":", 1)[1],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            txt = resp.choices[0].message.content or "[]"
            items = json.loads(txt)
            for it in items:
                out.append(QA(
                    question=it['question'],
                    options=it['options'],
                    correct_index=int(it['correct_index']),
                    explanation=it.get('explanation', ''),
                    category=topic,
                    difficulty=difficulty,
                ))
        elif model_pref.startswith("ollama:"):
            import requests  # type: ignore
            model = model_pref.split(":", 1)[1]
            prompt = (
                "Jsi generátor kvízových otázek v češtině. Vrať POUZE validní JSON pole objektů se schématem "
                "{question, options[4], correct_index(0-3), explanation}. "
                f"Téma: {topic}; Obtížnost: {difficulty}; Počet: {n}."
            )
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            payload = r.json()
            items = json.loads(payload.get("response", "[]"))
            for it in items:
                out.append(QA(
                    question=it['question'],
                    options=it['options'],
                    correct_index=int(it['correct_index']),
                    explanation=it.get('explanation', ''),
                    category=topic,
                    difficulty=difficulty,
                ))
        else:
            raise RuntimeError("Nepodporovaný model_pref")
    except Exception as e:
        st.info(f"AI režim selhal ({e}). Přepínám na lokální generování.")
        out = []

    if not out:
        out = synthesize_questions_local(topic, difficulty, n)
    return [shuffle_question(q) for q in out[:n]]

# =============== UI ===============
st.set_page_config(page_title="Vědomostní kvíz", page_icon="❓", layout="centered")
st.title("🧠 Vědomostní kvíz")

with st.sidebar:
    st.header("Nastavení")
    topic = st.selectbox("Téma", TOPICS, index=5)
    diff = st.selectbox("Obtížnost", DIFFS, index=0)
    n_questions = st.number_input("Počet otázek", min_value=10, max_value=100, value=10, step=1)
    time_limit = st.number_input("Limit na otázku (s)", min_value=5, max_value=30, value=20, step=1)

    st.divider()
    st.subheader("Generování otázek")
    gen_mode = st.radio("Režim", ["Lokální", "AI"], horizontal=True)
    model_pref = st.text_input("Model (AI)", value="openai:gpt-4o-mini", help="openai:gpt-4o-mini nebo ollama:llama3")
    st.caption("AI režim vyžaduje klíč (OpenAI) nebo běžící Ollama server.")

    start = st.button("▶️ Spustit kvíz", use_container_width=True)

# =============== Session state ===============
ss = st.session_state
ss.setdefault("questions", [])
ss.setdefault("q_index", 0)
ss.setdefault("score", 0)
ss.setdefault("answers", [])
ss.setdefault("deadline", None)
ss.setdefault("running", False)

# =============== Start ===============
if start:
    if gen_mode == "AI":
        ss["questions"] = synthesize_questions_ai(topic, diff, int(n_questions), model_pref=model_pref)
    else:
        ss["questions"] = synthesize_questions_local(topic, diff, int(n_questions))
    ss["q_index"] = 0
    ss["score"] = 0
    ss["answers"] = []
    ss["running"] = True
    ss["deadline"] = time.time() + int(time_limit)
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# =============== Hlavní logika kvízu ===============
if ss["running"] and ss["questions"]:
    q: QA = ss["questions"][ss["q_index"]]

    # Odpočet – bezpečně, protože autorefresh nemusí být v každé verzi
    remaining = max(0, int(ss["deadline"] - time.time())) if ss["deadline"] else 0
    try:
        st.progress((remaining / time_limit) if time_limit else 0.0)
    except Exception:
        pass
    st.write(f"⏳ Zbývá: **{remaining} s** / limit {int(time_limit)} s")

    st.subheader(f"Otázka {ss['q_index'] + 1} / {len(ss['questions'])}")
    st.markdown(f"**{q.question}**")

    labels = [f"A) {q.options[0]}", f"B) {q.options[1]}", f"C) {q.options[2]}", f"D) {q.options[3]}"]
    choice = st.radio(
        "Vyber odpověď:",
        options=list(range(4)),
        format_func=lambda i: labels[i],
        index=None,
        key=f"choice_{ss['q_index']}"
    )

    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Potvrdit odpověď", use_container_width=True)
    with col2:
        skip = st.button("Přeskočit", use_container_width=True)

    # Autorefresh pro odpočet (pokud je k dispozici)
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=1000, key="timer")

    # Vynucené odeslání po vypršení
    force_submit = remaining <= 0

    if submit or skip or force_submit:
        is_correct = (choice is not None) and (int(choice) == q.correct_index)
        ss["score"] += 1 if is_correct else 0
        ss["answers"].append({
            "question": q.question,
            "chosen": None if choice is None else q.options[int(choice)],
            "correct": q.options[q.correct_index],
            "correct_bool": bool(is_correct),
            "explanation": q.explanation,
            "category": q.category,
            "difficulty": q.difficulty,
            "time_limit_s": int(time_limit),
            "time_left_s": remaining,
        })
        # Další otázka
        ss["q_index"] += 1
        if ss["q_index"] >= len(ss["questions"]):
            ss["running"] = False
        else:
            ss["deadline"] = time.time() + int(time_limit)
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

# =============== Výsledky ===============
if (not ss["running"]) and ss["questions"]:
    st.success("Hotovo! Tady jsou výsledky.")
    correct_count = sum(1 for a in ss["answers"] if a["correct_bool"]) if ss["answers"] else 0
    st.metric("Správně", f"{correct_count} / {len(ss['questions'])}")
    st.metric("Skóre", f"{ss['score']}")

    df = pd.DataFrame(ss["answers"])
    df_show = df[[
        "question", "chosen", "correct", "correct_bool", "explanation", "category", "difficulty", "time_left_s"
    ]]
    df_show = df_show.rename(columns={
        "question": "Otázka",
        "chosen": "Tvoje odpověď",
        "correct": "Správná odpověď",
        "correct_bool": "Správně?",
        "explanation": "Vysvětlení",
        "category": "Téma",
        "difficulty": "Obtížnost",
        "time_left_s": "Zbývající čas (s)",
    })
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Stáhnout výsledky (CSV)", csv, file_name="kviz_vysledky.csv", mime="text/csv")

    st.divider()
    if st.button("🔁 Hrát znovu se stejnými nastaveními", use_container_width=True):
        ss.update({"running": True, "q_index": 0, "score": 0, "answers": [], "deadline": time.time() + int(time_limit)})
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    if st.button("🏠 Zpět na start (vybrat nové nastavení)", use_container_width=True):
        ss.update({"questions": [], "running": False, "q_index": 0, "score": 0, "answers": [], "deadline": None})
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

# =============== Import vlastní banky ===============
with st.expander("📥 Import vlastní banky otázek (JSON)"):
    st.caption("Formát: pole objektů {question, options[4], correct_index, explanation, category, difficulty}.")
    up = st.file_uploader("Nahraj JSON soubor", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            data = json.load(up)
            items = data if isinstance(data, list) else data.get("items", [])
            loaded: List[QA] = []
            for it in items:
                loaded.append(QA(
                    question=it['question'],
                    options=it['options'],
                    correct_index=int(it['correct_index']),
                    explanation=it.get('explanation', ''),
                    category=it.get('category', TOPICS[0]),
                    difficulty=it.get('difficulty', DIFFS[0]),
                ))
            BUILTIN_BANK.clear()
            BUILTIN_BANK.extend(loaded)
            st.success(f"Načteno {len(loaded)} otázek. Lokální režim teď bude čerpat z tvého souboru.")
        except Exception as e:
            st.error(f"Nepodařilo se načíst JSON: {e}")

with st.expander("ℹ️ Tipy"):
    st.markdown(
        "- **Shuffle** odpovědí je zapnutý pro každou otázku.
"
        "- **Časovač** automaticky vyhodnotí otázku po vypršení limitu.
"
        "- **AI režim**: pokud selže, aplikace plynule přepne na lokální generování.
"
        "- Můžeš nahrát **vlastní JSON banku** a používat Lokální režim bez AI."
    )

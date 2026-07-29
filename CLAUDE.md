# CLAUDE.md — voice-engine

הוראות קבועות ל-Claude Code ברפו הזה. נטען בכל סשן. עד עכשיו סשן שנפתח כאן
התחיל עיוור — הקובץ הזה הוא מרכז הידע של הרפו: קרא אותו לפני חקירת העץ.

## שפה ואזור זמן

- כל טקסט שמופנה למשתמש — **בעברית**. קוד, מזהים, לוגים וטקסט קומיטים —
  באנגלית, לפי מוסכמות הרפו.
- כל תאריך/שעה שמוצגים למשתמש — באזור הזמן **America/New_York** (המשתמש
  בניו-יורק). אחסון ב-UTC כרגיל; המרה לניו-יורק בגבול התצוגה בלבד.

## מה זה הרפו

שירות יצירת האודיו של **smrtVoice** (חלק מפלטפורמת smrtesy): ‏Python 3.11 +
FastAPI + Huey על Redis, פרוס על **Railway** (‏`railway.json` + `Dockerfile`).
הזרימה: ‏smrtesy‏ (Node, הריפו `mrtesy-app`) שולח עבודה ב-`POST /jobs` →
ה-API מתייק ל-Redis → ה-worker מריץ TTS דרך ה-adapters → התוצר נשמר →
webhook חוזר ל-smrtesy‏ (`/api/voice/webhook`).

## מפת הקוד (`src/voice_engine/`)

| איפה | מה |
|---|---|
| `main.py`, `config.py` | כניסת ה-FastAPI + קונפיג מ-env |
| `api/` | הראוטים: `jobs.py`, `parse.py`, `voices.py`, `health.py`; אימות ב-`auth.py` |
| `workers/` | `huey_app.py` (התור), `tasks.py`, `orchestrator.py` — העיבוד בפועל |
| `adapters/` | ספקי קול מתחלפים: `resemble.py` (פרודקשן), `chatterbox.py` (עתידי), `factory.py`, חוזה משותף ב-`base.py` |
| `preprocessor/` | עיבוד טקסט לפני TTS — **קורא ל-Anthropic API בתשלום** (`llm_client.py`, `prompts.py`, `processor.py`, `fidelity.py`) |
| `parsers/` | פירוק תסריטים: `script.py`, `google_docs.py`, `directions.py` |
| `dictionaries/` | הגייה: `hebrew_names.py`, `chabad_pronunciation.py`, `pronunciations.py`, `resemble_tags.py`, `emotion_directions.py` |
| `audio/` | `analyzer.py`, `splitter.py`, `postprocess.py`, `utils.py` |
| `db/` | גישת Supabase: jobs, lines, takes, scripts, projects, characters, lexicon, webhook_outbox |
| `storage/` | Supabase Storage (`storage_manager.py`, `supabase_client.py`) |
| `platform/webhooks.py` | שליחת ה-webhooks חזרה ל-smrtesy |
| `models/`, `lib/` | מודלי Pydantic (domain/events/requests/responses); לוגר, שגיאות, עזרי עברית |

טסטים: `tests/unit` + `tests/integration` (‏pytest). הרצה מקומית: ‏README —
‏uvicorn לשרת + `huey_consumer` ל-worker בטרמינל שני.

## כללים

- **אישור עלות לפני כל פעולה שמפעילה API בתשלום.** ה-preprocessor קורא
  ל-Anthropic וה-adapters ל-Resemble — הרצה ידנית / באץ' שמחייבת את המשתמש
  דורשת: מה ירוץ, הערכת עלות (לפריט + סה"כ), והמתנה ל"כן" מפורש על העלות.
  עבודת הסוכן עצמה רצה על המנוי — לא דורשת אישור.
- **רעננות המפה:** שינוי מבני כאן — תיקייה/מודול/ראוט שנוסף, הוזז או נמחק —
  מעדכן את טבלת מפת-הקוד שבקובץ הזה **באותו קומיט**. Verified: 2026-07-29.

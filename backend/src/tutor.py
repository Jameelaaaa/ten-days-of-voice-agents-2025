import logging
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (  # type: ignore
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation  # type: ignore
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # type: ignore

logger = logging.getLogger("day4_tutor")
load_dotenv(".env.local")

# Paths and content
DATA_DIR = Path(__file__).parent.parent / "shared-data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONTENT_PATH = DATA_DIR / "day4_tutor_content.json"

DEFAULT_CONTENT = [
    {
        "id": "variables",
        "title": "Variables",
        "summary": "Variables store values so you can reuse and change them later in a program.",
        "sample_question": "What is a variable and why is it useful?"
    },
    {
        "id": "loops",
        "title": "Loops",
        "summary": "Loops let you repeat an action multiple times, which helps avoid writing repeated code.",
        "sample_question": "Explain the difference between a for loop and a while loop."
    }
]

if not CONTENT_PATH.exists():
    with open(CONTENT_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONTENT, f, indent=2, ensure_ascii=False)


# Tutor Agent
class TutorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, patient tutor. Greet the user, ask which mode they want (learn, quiz, teach_back), "
                "and when asked to teach, generate a long, natural explanation in your own words using very simple language. "
                "Do not repeat any stored summary verbatim; instead, compose a fresh explanation with at least three sentences and one short example. "
                "Be encouraging and teacher-like. Keep messages plain text (no asterisks or decorations)."
            )
        )
        
        # Initialize order state
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None
        }
        
        # Create orders directory if it doesn't exist
        self.orders_dir = Path("orders")
        self.orders_dir.mkdir(exist_ok=True)

        # Load content
        with open(CONTENT_PATH, "r", encoding="utf-8") as f:
            self.content = json.load(f)

        # State
        self.mode = None
        self.current = self.content[0] if self.content else None
        self._session = None  # Will be set after session starts
        self._tts_instances = {}  # Will hold pre-created TTS instances

        # Voice mapping for modes (Murf Falcon voice names)
        # learn -> Matthew, quiz -> Alicia, teach_back -> Ken
        self.voice_for_mode = {
            "learn": "en-US-matthew",
            "quiz": "en-US-alicia",
            "teach_back": "en-US-ken"
        }

    def _get_concept(self, concept_id: str = None):
        if concept_id:
            for c in self.content:
                if c["id"] == concept_id:
                    return c
        return self.current

    
    @function_tool
    async def list_concepts(self, context: RunContext):
        """Return available concepts (id: title)."""
        return "\n".join([f"{c['id']}: {c['title']}" for c in self.content])

    @function_tool
    async def set_mode(self, context: RunContext, mode: str = None, concept_id: str = None):
        """
        Set learning mode and optionally choose a concept.
        It switches the TTS voice to match the mode.
        """
        if mode and mode not in ("learn", "quiz", "teach_back"):
            return "Mode must be one of: learn, quiz, teach_back."

        if mode:
            self.mode = mode
            
            # Switch the TTS voice for this mode
            new_voice = self.voice_for_mode.get(self.mode, "en-US-matthew")
            
            if self._session and self._tts_instances:
                try:
                    # Switch to the TTS instance for this mode
                    self._session._tts = self._tts_instances[self.mode]
                    logger.info(f"Switched TTS voice to {new_voice} for mode {self.mode}")
                except Exception as e:
                    logger.error(f"Failed to switch TTS voice: {e}")
        
        if concept_id:
            c = self._get_concept(concept_id)
            if not c:
                return "Concept not found. Use list_concepts() to see valid ids."
            self.current = c

        return f"Mode set to {self.mode}. Concept: {self.current['title']}. Voice changed to {self.voice_for_mode.get(self.mode)}."

    @function_tool
    async def explain_concept(self, context: RunContext):
        """
        Just give a small hint, and let the AI explain everything in a simple way.
        """
        if not self.current:
            return "No concept selected."

        title = self.current.get("title", "Concept")
        seed = self.current.get("summary", "")
        instruction = (
            f"Explain '{title}' in very simple, everyday English. "
            "Do not repeat exact stored summary. Use at least three short sentences and include one tiny example. "
            "Keep tone supportive and teacher-like."
        )
        return {"seed": seed, "instruction": instruction}

    @function_tool
    async def ask_quiz_question(self, context: RunContext):
        """Return a short quiz prompt for the LLM to ask in teacher style."""
        if not self.current:
            return "No concept selected."
        return self.current.get("sample_question", "Can you explain this concept in your own words?")

    @function_tool
    async def assess_teach_back(self, context: RunContext, user_response: str = ""):
        """
        This is a small tool that listens to the user explain something back, checks for important words, and then gives helpful teacher-like feedback. It does not store the score anywhere.
        """
        if not self.current:
            return "No concept selected."

        ref_words = set((self.current.get("summary") or "").lower().split())
        resp_words = set((user_response or "").lower().split())
        overlap = len(ref_words & resp_words)
        total = max(1, len(ref_words))
        score = int((overlap / total) * 100)

        # feedback
        if score >= 80:
            feedback = (
                "Excellent explanation. You covered the main ideas clearly and used your own words. "
                "For improvement, add a tiny code or real-life example next time."
            )
        elif score >= 50:
            feedback = (
                "Good explanation. You included important points but missed a couple of details. "
                "Try: definition, why it matters, one short example."
            )
        elif score >= 25:
            feedback = (
                "Nice effort. You captured some parts. Focus on the main idea and give a short example."
            )
        else:
            feedback = (
                "Good start. You're learning — keep practising. Try: one-sentence definition, why we use it, then an example."
            )

        sample_answer = ""
        cid = self.current["id"]
        if cid == "variables":
            sample_answer = "Sample: A variable is a named place to store a value, like age = 20. We use it to save and reuse data."
        elif cid == "loops":
            sample_answer = "Sample: A loop repeats actions, for example printing each item in a list. Use it to avoid repeating code."

        next_step = "Would you like to try again, or shall I give a short quiz?"
        return f"Score: {score}% — {feedback} {sample_answer} {next_step}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Create the agent first
    agent = TutorAgent()

    # Create TTS instances for each mode with different voices
    agent._tts_instances = {
        "learn": murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        "quiz": murf.TTS(
            voice="en-US-alicia",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        "teach_back": murf.TTS(
            voice="en-US-ken",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
    }

    # Start with "learn" mode TTS (Matthew voice)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=agent._tts_instances["learn"],  # Default to learn mode
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Give agent access to the session so it can switch TTS
    agent._session = session

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
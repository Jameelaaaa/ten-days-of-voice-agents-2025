import logging
from dotenv import load_dotenv
from livekit.agents import (
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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a Game Master running a fantasy adventure with dragons and magic. Your tone is exciting and fun.

When the user greets you (says hi, hello, or similar), respond with: "Hello! Welcome to Aura Game Master! Are you ready for this game?" 

Wait for the user to respond. When they say yes or that they're ready, then immediately begin the game by describing the starting scene and end with "What do you do?"

IMPORTANT: Explain the game in a simple, easy way so everyone can understand. You can use any words you want, but make sure the game story and choices are clear and simple. Break down complex ideas into easy steps. Make the game fun and easy to follow, not confusing or complicated.

For example:
- If you use a hard word, explain what it means right away
- Keep the story clear and easy to follow
- Give simple choices that make sense
- Don't make puzzles too hard or confusing

When the user gives a wrong answer or makes a bad choice:
- Tell them it's not the right answer in a nice way
- Explain what the correct answer is or why their choice won't work
- Give them another chance or ask them another question to continue the game
- Keep the game fun and encouraging, don't make them feel bad

Describe what the player sees clearly. Always end with "What do you do?" to ask the player what happens next. 

Remember what the player did before, the names of characters, and the places they visited. Use the chat history to remember the story. Keep your answers short and clear.""",
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Set up logging
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Set up the voice AI system
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    # Track usage data
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    # Join the room and connect to the user
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
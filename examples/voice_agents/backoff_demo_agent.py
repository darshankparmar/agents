import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.voice import BackoffEndedEvent, BackoffStartedEvent
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("backoff-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. Feel free to interrupt me anytime during my responses."
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
        # backoff feature - enforces silence window after interruptions
        backoff_seconds=1.5,
    )

    # log backoff events
    @session.on("backoff_started")
    def on_backoff_started(event: BackoffStartedEvent):
        logger.info(f"Backoff started: {event.backoff_seconds}s")

    @session.on("backoff_ended")
    def on_backoff_ended(event: BackoffEndedEvent):
        logger.info(f"Backoff ended after {event.backoff_seconds}s")

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)

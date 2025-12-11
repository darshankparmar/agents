#!/usr/bin/env python3
"""
BackoffSeconds Demo Agent

This example demonstrates the new backoffSeconds feature in LiveKit Agents.
The backoff feature enforces a configurable silence window after user interruptions,
improving conversational UX by preventing the agent from speaking over users
who have just interrupted.

Key features demonstrated:
- Configurable backoff duration (backoff_seconds)
- Restart policy options ("restart" vs "ignore")
- Event monitoring for backoff lifecycle
- Different conversation scenarios (gaming, assistant, healthcare)
"""

import logging
from typing import Literal

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import BackoffEndedEvent, BackoffStartedEvent
from livekit.plugins import cartesia, deepgram, google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("backoff-demo-agent")

load_dotenv()


class BackoffDemoAgent(Agent):
    def __init__(self, scenario: str = "assistant") -> None:
        # Customize instructions based on scenario
        instructions_map = {
            "gaming": (
                "You are a gaming assistant. Respond quickly and concisely. "
                "Keep responses under 10 words when possible. Be energetic and direct."
            ),
            "assistant": (
                "You are a helpful AI assistant. Speak naturally and conversationally. "
                "Keep responses clear and engaging. You can be interrupted at any time."
            ),
            "healthcare": (
                "You are a healthcare information assistant. Speak slowly and clearly. "
                "Allow plenty of time for users to ask questions or clarify. "
                "Be patient and thorough in your responses."
            ),
            "support": (
                "You are a customer support agent. Be helpful and professional. "
                "Listen carefully to customer concerns and provide clear solutions."
            ),
        }

        super().__init__(instructions=instructions_map.get(scenario, instructions_map["assistant"]))
        self.scenario = scenario

    async def on_enter(self):
        logger.info(f"BackoffDemo agent entered - scenario: {self.scenario}")

        # Generate different initial messages based on scenario
        if self.scenario == "gaming":
            self.session.say("Ready to game! What's up?")
        elif self.scenario == "healthcare":
            self.session.say(
                "Hello, I'm here to help with your healthcare questions. Please take your time and feel free to interrupt me if you need clarification."
            )
        elif self.scenario == "support":
            self.session.say("Hi! I'm here to help you today. What can I assist you with?")
        else:
            self.session.say(
                "Hello! I'm your AI assistant. Feel free to interrupt me anytime - I'll wait for you to finish before responding."
            )

    @function_tool
    async def demonstrate_backoff(self, context: RunContext, scenario: str):
        """Demonstrate different backoff scenarios.

        Args:
            scenario: The scenario to demonstrate (gaming, assistant, healthcare, support)
        """
        logger.info(f"Demonstrating backoff for scenario: {scenario}")

        if scenario == "gaming":
            return "Gaming mode uses 0 second backoff for instant responses. Try interrupting me!"
        elif scenario == "healthcare":
            return "Healthcare mode uses 2.5 second backoff to ensure patients can fully express themselves. This gives you plenty of time to clarify or add more information after interrupting."
        elif scenario == "support":
            return "Support mode uses 1.5 second backoff to balance responsiveness with careful listening. This helps ensure I understand your complete question."
        else:
            return "Assistant mode uses 1 second backoff as a balanced default. This prevents me from talking over you while keeping the conversation flowing naturally."

    @function_tool
    async def explain_backoff_feature(self, context: RunContext):
        """Explain how the backoff feature works."""
        return (
            "The backoff feature creates a silence window after you interrupt me. "
            "During this time, I won't speak even if I have a response ready, "
            "but I'm still listening to everything you say. "
            "This prevents me from talking over you and gives you time to finish your thought."
        )

    @function_tool
    async def test_long_response(self, context: RunContext):
        """Generate a long response to test interruption behavior."""
        return (
            "This is a deliberately long response to test the backoff feature. "
            "You can interrupt me at any point during this speech, and I'll pause "
            "for the configured backoff duration before responding to your interruption. "
            "The backoff ensures I don't immediately start talking over you when you interrupt. "
            "Try interrupting me now to see how it works! "
            "I'll wait for the backoff period to complete before I respond to whatever you say."
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm models for better performance."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def get_backoff_config(scenario: str) -> tuple[float, Literal["restart", "ignore"]]:
    """Get backoff configuration for different scenarios."""
    configs = {
        "gaming": (0.0, "restart"),  # No backoff for gaming
        "assistant": (1.0, "restart"),  # Balanced default
        "healthcare": (2.5, "ignore"),  # Longer pause, don't restart on rapid interruptions
        "support": (1.5, "restart"),  # Moderate pause with restart
    }
    return configs.get(scenario, configs["assistant"])


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the backoff demo agent."""

    scenario = "healthcare"
    backoff_seconds, backoff_restart_policy = get_backoff_config(scenario)

    logger.info(
        f"Starting backoff demo - scenario: {scenario}, backoff: {backoff_seconds}s, policy: {backoff_restart_policy}"
    )

    # Configure session with backoff settings
    session = AgentSession(
        # Core models
        stt=deepgram.STT(),
        llm=google.LLM(),
        # llm="openai/gpt-4.1-mini",
        tts=cartesia.TTS(voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),  # Sonic voice
        # Turn detection and VAD~
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Backoff configuration - THE NEW FEATURE!
        backoff_seconds=backoff_seconds,
        backoff_restart_policy=backoff_restart_policy,
    )

    # Set up event monitoring for backoff events
    @session.on("backoff_started")
    def on_backoff_started(event: BackoffStartedEvent):
        logger.info(f"🔇 Backoff started: {event.backoff_seconds}s silence window")
        print(f"[BACKOFF] Started {event.backoff_seconds}s silence window at {event.created_at}")

    @session.on("backoff_ended")
    def on_backoff_ended(event: BackoffEndedEvent):
        logger.info(f"🔊 Backoff ended: Agent can speak again after {event.backoff_seconds}s")
        print(f"[BACKOFF] Ended after {event.backoff_seconds}s at {event.created_at}")

    # Monitor other relevant events
    @session.on("user_input_transcribed")
    def on_user_input(event):
        if event.is_final:
            logger.info(f"User said: {event.transcript}")

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        logger.info(f"Agent state: {event.old_state} -> {event.new_state}")

    # Log configuration for debugging
    logger.info("Session configuration:")
    logger.info(f"  - Scenario: {scenario}")
    logger.info(f"  - Backoff seconds: {backoff_seconds}")
    logger.info(f"  - Backoff restart policy: {backoff_restart_policy}")
    logger.info(f"  - Allow interruptions: {session.options.allow_interruptions}")
    logger.info(f"  - Resume false interruption: {session.options.resume_false_interruption}")

    # Start the session
    await session.start(
        agent=BackoffDemoAgent(scenario=scenario),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            audio_output=room_io.AudioOutputOptions(),
        ),
    )


if __name__ == "__main__":
    # Add custom CLI arguments for testing different scenarios
    import argparse
    import sys

    # Parse scenario argument if provided
    parser = argparse.ArgumentParser(description="BackoffSeconds Demo Agent")
    parser.add_argument(
        "--scenario",
        choices=["gaming", "assistant", "healthcare", "support"],
        default="assistant",
        help="Conversation scenario to demonstrate",
    )

    # Parse known args to avoid conflicts with livekit CLI
    args, unknown = parser.parse_known_args()

    # Set scenario in environment or room metadata would be better,
    # but for demo purposes we'll log it
    print("\n🎯 BackoffSeconds Demo Agent")
    print(f"📋 Scenario: {args.scenario}")

    scenario_configs = {
        "gaming": "0.0s backoff (immediate response)",
        "assistant": "1.0s backoff (balanced)",
        "healthcare": "2.5s backoff (careful listening)",
        "support": "1.5s backoff (moderate pause)",
    }

    print(f"⏱️  Configuration: {scenario_configs[args.scenario]}")
    print("🎮 Try interrupting the agent to see the backoff feature in action!")
    print("📊 Watch the console for [BACKOFF] events\n")

    # Restore sys.argv for livekit CLI
    sys.argv = [sys.argv[0]] + unknown

    cli.run_app(server)

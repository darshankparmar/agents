from dotenv import load_dotenv  
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli  
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.stt.stt import SpeechEventType
from livekit.plugins import groq, deepgram, cartesia  
  
load_dotenv()  

class GroqTurnDetectionAgent(Agent):  
    def __init__(self):  
        super().__init__(  
            instructions="You are a helpful voice assistant."  
        )  
          
        self._turn_llm = groq.LLM(
            model="llama-3.1-8b-instant",
            temperature=0
        )  
        self._pending_transcript = ""  
      
    # async def on_enter(self):  
    #     self.session.generate_reply()  
      
    async def check_turn_completion(self, transcript: str) -> bool:
        turn_llm_ctx = ChatContext.empty()

        turn_llm_ctx.add_message(
            role="system",
            content="""
    You classify if the user has finished speaking.
    Answer ONLY: yes or no.
    Do NOT add anything else.

    Rules:
    - Full question or statement → yes
    - Short fragments → no
    - No explanations.
    """
        )

        turn_llm_ctx.add_message(
            role="user",
            content=f'Transcript: "{transcript}"\nIs the turn complete?'
        )

        chunks = []
        async for c in self._turn_llm.chat(chat_ctx=turn_llm_ctx).to_str_iterable():
            chunks.append(c)

        response = "".join(chunks).strip().lower()
        print("Turn LLM raw:", response)

        response = response.replace(".", "").strip()

        return response == "yes"
      
    async def stt_node(self, audio, model_settings):  
        """Override STT node to intercept transcripts."""  
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:  
                transcript = event.alternatives[0].text  
                print("Received transcript:", transcript)  
                  
                # Accumulate transcript  
                self._pending_transcript += (" " if self._pending_transcript else "") + transcript  
                self._pending_transcript = self._pending_transcript.lstrip()  
                  
                if await self.check_turn_completion(self._pending_transcript):  
                    print("Turn complete, committing...")  
                    self.session.commit_user_turn()  
                    self._pending_transcript = ""  
                else:  
                    print("Still listening...")  
                    await self.session.say("Still I am listening...")
              
            yield event  
  
async def entrypoint(ctx: JobContext):  
    await ctx.connect()  
      
    agent = GroqTurnDetectionAgent()  
    session = AgentSession(  
        turn_detection="manual",  
        stt=deepgram.STT(),  
        tts=cartesia.TTS(),  
        llm=groq.LLM(model="llama-3.3-70b-versatile"),  
    )  
      
    await session.start(agent=agent, room=ctx.room)  
  
if __name__ == "__main__":  
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

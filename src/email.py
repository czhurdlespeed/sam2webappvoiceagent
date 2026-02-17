import json
import os

import logfire
import resend
from openai import AsyncOpenAI

from .usersession import UserAgentUsage


def confirm_conversation_history(chat_history: list[dict]) -> bool:
    for dictionary in chat_history:
        if "role" in dictionary and dictionary["role"] == "user":
            return True
    return False


async def generate_summary(chat_history: list[dict]) -> str:
    logfire.info("Generating summary for chat history", chat_history=chat_history)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.responses.create(
        model="gpt-4.1-nano",
        input=f"""You are a helpful assistant that generates a summary of a conversation you had with the user. Write as if you are talking directly to the user about the conversation. This serves as a refresher for the user.
        Start the summary with an h2 tag and the title "Conversation Summary". Don't greet the user in the summary. Point out the key topics and insights from the conversation and where the user can go to get more information by sharing the relevant web links or titles. 
        The conversation history is as follows: {json.dumps(chat_history)}. Return the summary as html that will be embedded in an email. Don't ask the user any questions in the summary.""",
        max_output_tokens=1000,
    )
    return response.output_text


def template_email(
    user_agent_usage: UserAgentUsage,
    summary: str,
    *,
    total_seconds_used: int,
    session_count: int,
) -> str:
    return f"""
    <p>Hey {user_agent_usage.name.split(" ")[0]} 👋,</p>

    <p>Thank you for visiting <a href="https://nocodefinetuning.calvinwetzel.dev">SAM 2 No Code Finetuning</a>! I enjoyed talking with you and hope I was able to help ☺️! If you have any other questions
    or need any further assistance, please follow up by emailing <a href="mailto:info@calvinwetzel.dev">info@calvinwetzel.dev</a>.</p>

    <p>Checkout the summary of our conversation below 👇:</p>

    <p>Usage Stats:</p>
    <ul>
    <li><b>Total time spent:</b> {total_seconds_used} seconds</li>
    <li><b>Your total number of sessions:</b> {session_count}</li>
    <li><b>Agent assistance credits remaining:</b> {max(0, int(os.getenv("SESSION_TIME_LIMIT_SECONDS", "120")) - total_seconds_used)} seconds</li>
    </ul>

    {summary}
    
    <p>Happy training 👾,</p>

    <p>Calvin, your friendly neighborhood AI Agent 🤖</p>
    """


def send_email(
    user_agent_usage: UserAgentUsage, html: str, *, session_count: int
) -> None:
    try:
        params: resend.Emails.SendParams = resend.Emails.SendParams(
            **{"from": "SAM 2 No Code Finetuning Assistant <info@calvinwetzel.dev>"},
            to=[f"{user_agent_usage.name} <{user_agent_usage.email}>"],
            bcc="Calvin Wetzel <info@calvinwetzel.dev>",
            subject=f"SAM 2 No Code Finetuning Assistant - {user_agent_usage.name.split(' ')[0]}'s - Agent Chat Session {session_count} Summary",
            html=html,
        )
        email = resend.Emails.send(params)
        logfire.info("Email sent successfully ✉️ ✅", email=email)
    except Exception as e:
        logfire.error("Failed to send email", error=e, exc_info=True)
